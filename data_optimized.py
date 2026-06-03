import random
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Iterator

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from scipy.io import loadmat


def parse_mars_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extrait (pid, camid) depuis un nom MARS: 0001C1T0001F001.jpg
    Gère aussi les junk (00-1...) qui doivent être ignorés.
    """
    # Junk frames: pid=-1 encodé comme "00-1..."
    if name.startswith("00-1"):
        cam_match = re.search(r"C(?P<cam>\d)", name)
        cam = int(cam_match.group("cam")) if cam_match else None
        return "-1", cam

    m = re.match(r"(?P<pid>\d{4})C(?P<cam>\d)T\d{4}F\d{3}", name)
    if not m:
        return None, None
    return m.group("pid"), int(m.group("cam"))


def load_split_json(path: str, data_root: str = "", strict: bool = False,
                    allow_pid_fallback: bool = True, 
                    keep_distractors: bool = True) -> List[Tuple[str, str, int]]:
    """
    Charge (path, pid, camid) depuis un fichier liste ou un dossier.
    
    Args:
        path: Chemin vers le fichier .txt ou dossier
        data_root: Racine pour les chemins relatifs
        strict: Lève une erreur si fichier manquant
        allow_pid_fallback: Tente de retrouver l'image dans bbox_train/<pid>/
        keep_distractors: Inclut les images avec pid==0000
    """
    p = Path(path)
    if data_root and not p.is_absolute():
        p = Path(data_root) / p
    root = Path(data_root) if data_root else p.parent
    items = []
    
    # Mode dossier (MARS bbox_train/bbox_test)
    if p.exists() and p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_path in sorted(x for x in p.rglob("*") if x.suffix.lower() in exts):
            pid, camid = parse_mars_name(img_path.stem)
            pid = pid or "0"
            camid = camid or 0
            items.append((str(img_path), pid, camid))
        
        if not items:
            raise FileNotFoundError(f"Aucune image trouvée dans {p}")
        return items
    
    if not p.exists():
        raise FileNotFoundError(f"Fichier split non trouvé: {p}")
    
    # Mode fichier texte
    if p.suffix == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                rel_path = parts[0]
                pid_hint = parts[1] if len(parts) > 1 else None
                
                # Parsing du nom de fichier
                stem = Path(rel_path).stem
                pid, camid = parse_mars_name(stem)
                
                if pid_hint:
                    pid = pid_hint
                pid = pid or pid_hint or "0"
                
                if camid is None:
                    m = re.search(r"[cC](\d+)", stem)
                    camid = int(m.group(1)) if m else 0
                
                full_path = root / rel_path
                
                # Fallback MARS standard
                if allow_pid_fallback and not full_path.exists() and data_root:
                    fname = Path(rel_path).name
                    # 1) bbox_train/<pid>/file.jpg
                    alt_train = Path(data_root) / "bbox_train" / pid / fname
                    # 2) bbox_test/<pid>/file.jpg
                    alt_test = Path(data_root) / "bbox_test" / pid / fname
                    full_path = alt_train if alt_train.exists() else alt_test
                
                # Gestion des fichiers manquants
                if not full_path.exists():
                    msg = f"[{p.name}:{line_num}] Image manquante: {full_path}"
                    if strict:
                        raise FileNotFoundError(msg)
                    logging.warning(msg)
                    continue
                
                # Filtrage junk/distractors
                if pid == "-1":
                    continue
                if pid == "0000" and not keep_distractors:
                    continue
                
                items.append((str(full_path), pid, camid))
    
    return items


class ReidDataset(Dataset):
    """
    Dataset ReID avec support two-view pour augmentation.
    """
    
    def __init__(self, items: List[Tuple[str, str, int]], 
                 pid2label: Optional[dict] = None,
                 transform: Optional[T.Compose] = None,
                 two_view: bool = False):
        self.items = items
        self.pid2label = pid2label
        self.transform = transform
        self.two_view = two_view
        
        # Cache pour les images déjà chargées (utile pour petits datasets)
        self._cache = {}
        self._cache_size = 1000
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int):
        path, pid, camid = self.items[idx]
        
        # Chargement avec cache
        if path in self._cache:
            img = self._cache[path]
        else:
            try:
                img = Image.open(path).convert("RGB")
                if len(self._cache) < self._cache_size:
                    self._cache[path] = img.copy()
            except Exception as e:
                logging.error(f"Erreur chargement {path}: {e}")
                img = Image.new("RGB", (128, 256))
        
        # Transformation
        if self.transform:
            if self.two_view and self.pid2label is not None:
                # Deux vues augmentées de la même image
                img1 = self.transform(img)
                img2 = self.transform(img)
                label = self.pid2label[pid]
                return img1, img2, label, camid, path
            elif self.pid2label is not None:
                img_t = self.transform(img)
                label = self.pid2label[pid]
                return img_t, label, camid, path
            else:
                # Mode évaluation
                return self.transform(img), pid, camid, path


class PKSampler(Sampler):
    """
    Sampler P-personnes x K-images avec épuisement sans remise.
    
    Garantit que chaque identité est vue exactement le même nombre de fois
    par époque (contrairement à random.choices avec remise).
    """
    
    def __init__(self, data_source: List[Tuple], P: int, K: int, 
                 steps_per_epoch: Optional[int] = None):
        self.data_source = data_source
        self.P = P
        self.K = K
        
        # Construction de l'index par PID
        self.idx_by_pid = defaultdict(list)
        for idx, item in enumerate(data_source):
            pid = item[1]
            self.idx_by_pid[pid].append(idx)
        
        self.pids = list(self.idx_by_pid.keys())
        
        # Nombre de steps
        if steps_per_epoch:
            self.num_batches = steps_per_epoch
        else:
            self.num_batches = max(1, len(self.pids) // P)
        
        # Validation
        min_samples = min(len(v) for v in self.idx_by_pid.values())
        if min_samples < K:
            logging.warning(
                f"Certaines identités n'ont que {min_samples} samples < K={K}. "
                f"Utilisation de sampling avec remise pour celles-ci."
            )
        
        # Buffers circulaires pour chaque PID
        self._reset_buffers()
    
    def _reset_buffers(self):
        """Réinitialise les buffers avec shuffle."""
        self.buffers = {}
        for pid in self.pids:
            indices = self.idx_by_pid[pid].copy()
            random.shuffle(indices)
            self.buffers[pid] = indices
    
    def __iter__(self) -> Iterator[List[int]]:
        self._reset_buffers()
        
        for _ in range(self.num_batches):
            batch = []
            selected_pids = random.sample(self.pids, self.P)
            
            for pid in selected_pids:
                need = self.K
                available = len(self.buffers[pid])
                
                # Premier essai: prendre sans remise
                take = min(need, available)
                if take > 0:
                    batch.extend(self.buffers[pid][:take])
                    self.buffers[pid] = self.buffers[pid][take:]
                    need -= take
                
                # Si besoin de plus: remise avec reshuffle
                while need > 0:
                    self.buffers[pid] = self.idx_by_pid[pid].copy()
                    random.shuffle(self.buffers[pid])
                    take = min(need, len(self.buffers[pid]))
                    batch.extend(self.buffers[pid][:take])
                    self.buffers[pid] = self.buffers[pid][take:]
                    need -= take
            
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


def build_transforms(height: int, width: int, is_train: bool = True) -> T.Compose:
    """
    Construit les transformations avec augmentation cohérente.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ])
    else:
        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def build_query_gallery_from_mars(test_list: str, query_idx_path: str,
                                   data_root: str = "",
                                   keep_distractors: bool = True,
                                   use_all_as_query: bool = False) -> Tuple[List, List]:
    """
    Construit query/gallery depuis les fichiers MARS officiels.
    """
    test_items = load_split_json(test_list, data_root, keep_distractors=keep_distractors)
    
    if use_all_as_query:
        # Utiliser toutes les images comme queries et gallery (cross-camera complet)
        return test_items, test_items
    
    idx_path = Path(query_idx_path)
    if not idx_path.is_absolute() and data_root:
        idx_path = Path(data_root) / idx_path
    
    if not idx_path.exists():
        raise FileNotFoundError(f"query_IDX non trouvé: {idx_path}")
    
    mat = loadmat(idx_path)
    if "query_IDX" not in mat:
        raise KeyError(f"'query_IDX' absent de {idx_path}")
    
    # MATLAB est 1-based
    idxs = mat["query_IDX"].squeeze().astype(int)
    q_set = set(idxs)
    
    # Validation des indices
    invalid = [i for i in idxs if i < 1 or i > len(test_items)]
    if invalid:
        logging.warning(f"Indices invalides ignorés: {invalid[:10]}...")
        q_set = {i for i in idxs if 1 <= i <= len(test_items)}
    
    query_items = [test_items[i-1] for i in idxs if 1 <= i <= len(test_items)]
    gallery_items = [item for j, item in enumerate(test_items, 1) if j not in q_set]
    
    logging.info(f"MARS split: {len(query_items)} queries, {len(gallery_items)} gallery")
    return query_items, gallery_items


def _load_names_list(name_txt: str, data_root: str = "") -> List[str]:
    """Charge une liste de noms depuis un fichier texte."""
    p = Path(name_txt)
    if data_root and not p.is_absolute():
        p = Path(data_root) / p
    
    if not p.exists():
        raise FileNotFoundError(f"Liste de noms non trouvée: {p}")
    
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def _load_track_info(track_mat_path: str, data_root: str = ""):
    """Charge les informations de tracks depuis un .mat"""
    p = Path(track_mat_path)
    if data_root and not p.is_absolute():
        p = Path(data_root) / p
    
    if not p.exists():
        raise FileNotFoundError(f"Track mat non trouvé: {p}")
    
    mat = loadmat(p)
    key = next((k for k in mat.keys() if not k.startswith("__") and "info" in k), None)
    
    if key is None:
        raise KeyError(f"Aucune clé 'info' trouvée dans {p}")
    
    return mat[key]


def load_tracks_from_mat(track_mat_path: str, name_txt_path: str,
                         data_root: str = "", subset: str = "bbox_train",
                         strict: bool = False, allow_pid_fallback: bool = False,
                         keep_distractors: bool = True) -> List[Tuple]:
    """
    Charge les images depuis tracks_*_info.mat + name.txt
    """
    names = _load_names_list(name_txt_path, data_root)
    tracks = _load_track_info(track_mat_path, data_root)
    
    root = Path(data_root) if data_root else Path(name_txt_path).parent
    items = []
    
    for track_idx, row in enumerate(tracks):
        if len(row) < 4:
            logging.warning(f"Track {track_idx}: format invalide, ignoré")
            continue
        
        start_idx, end_idx = int(row[0]), int(row[1])
        pid = str(int(row[2])).zfill(4)
        camid = int(row[3])
        
        for idx in range(start_idx, end_idx + 1):
            if idx < 1 or idx > len(names):
                continue
            
            rel_path = names[idx - 1]
            full_path = root / rel_path
            
            # Fallback
            if allow_pid_fallback and not full_path.exists() and data_root:
                full_path = Path(data_root) / subset / pid / Path(rel_path).name
            
            if not full_path.exists():
                msg = f"Image manquante dans track: {full_path}"
                if strict:
                    raise FileNotFoundError(msg)
                logging.warning(msg)
                continue
            
            if pid == "-1":
                continue
            if pid == "0000" and not keep_distractors:
                continue
            
            items.append((str(full_path), pid, camid))
    
    return items


def build_query_gallery_from_tracks(test_tracks_mat: str, test_name_txt: str,
                                     query_idx_path: str, data_root: str = "",
                                     subset: str = "bbox_test", strict: bool = False,
                                     allow_pid_fallback: bool = False,
                                     keep_distractors: bool = True,
                                     use_all_as_query: bool = False) -> Tuple[List, List]:
    """
    Construit query/gallery depuis tracks_test_info.mat
    """
    names = _load_names_list(test_name_txt, data_root)
    tracks = _load_track_info(test_tracks_mat, data_root)
    
    q_path = Path(query_idx_path)
    if not q_path.is_absolute() and data_root:
        q_path = Path(data_root) / q_path
    
    q_mat = loadmat(q_path)
    idxs = q_mat["query_IDX"].squeeze().astype(int)
    q_set = set(idxs)
    
    root = Path(data_root) if data_root else Path(test_name_txt).parent
    
    def collect(track_indices: set) -> List[Tuple]:
        items = []
        for t_idx in track_indices:
            if t_idx < 1 or t_idx > tracks.shape[0]:
                continue
            
            row = tracks[t_idx - 1]
            start_idx, end_idx = int(row[0]), int(row[1])
            pid = str(int(row[2])).zfill(4)
            camid = int(row[3])
            
            for idx in range(start_idx, end_idx + 1):
                if idx < 1 or idx > len(names):
                    continue
                
                rel_path = names[idx - 1]
                full_path = root / rel_path
                
                if allow_pid_fallback and not full_path.exists() and data_root:
                    full_path = Path(data_root) / subset / pid / Path(rel_path).name
                
                if not full_path.exists():
                    if strict:
                        raise FileNotFoundError(f"Missing: {full_path}")
                    continue
                
                if pid == "0000" and not keep_distractors:
                    continue
                
                items.append((str(full_path), pid, camid))
        return items
    
    all_idxs = set(range(1, tracks.shape[0] + 1))
    if use_all_as_query:
        q_items = collect(all_idxs)
        g_items = q_items
    else:
        q_items = collect(q_set)
        g_items = collect(all_idxs - q_set)
    
    logging.info(f"Track split: {len(q_items)} queries, {len(g_items)} gallery")
    return q_items, g_items
