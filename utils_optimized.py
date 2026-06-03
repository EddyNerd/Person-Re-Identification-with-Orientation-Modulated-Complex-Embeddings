import os
import random
import logging
import sys
import json
import csv
import time
import importlib.util
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, cast
from contextlib import contextmanager
import warnings

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Device Management
# =============================================================================

class DeviceManager:
    """Gestion centralisée du device avec vérifications."""
    
    def __init__(self, device_str: str = "auto"):
        self.device = self._resolve(device_str)
        self.is_cuda = self.device.type == "cuda"
        self.is_mps = self.device.type == "mps"
        
    def _resolve(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        # Force explicit device; fail fast if CUDA demandé mais indisponible
        if device_str.lower().startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)
    
    def configure_reproducibility(self, seed: int, deterministic: bool = True, 
                                   cudnn_benchmark: bool = False):
        """Configure tous les seeds pour la reproductibilité."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if self.is_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
            if deterministic:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        
    def get_io_flags(self, pin_memory: bool = True, use_amp: bool = True) -> Tuple[bool, bool]:
        """Retourne (pin_memory, use_amp) adaptés au device."""
        pm = pin_memory and self.is_cuda
        amp = use_amp and self.is_cuda
        return pm, amp
    
    def verify_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Vérifie que le tensor est sur le bon device."""
        if tensor.device != self.device:
            raise RuntimeError(
                f"{name} sur {tensor.device}, attendu {self.device}"
            )
        return tensor


class ExternalOrientationProvider:
    """Load per-image orientation vectors from a CSV prediction file."""

    CLASS_NAME_TO_ANGLE = {
        "front": 0.0,
        "front-left": 42.5,
        "left side front": 75.0,
        "left side back": 105.0,
        "back-left": 137.5,
        "back": 180.0,
        "back-right": -137.5,
        "right side back": -105.0,
        "right side front": -75.0,
        "front-right": -42.5,
        # Legacy aliases kept for older coarse-5 exports.
        "face": 0.0,
        "three_quarters_front": 45.0,
        "profile": 90.0,
        "three_quarters_back": 135.0,
    }

    def __init__(
        self,
        csv_path: str,
        path_field: str = "image_path",
        angle_field: str = "pred_angle_deg",
        class_field: str = "pred_class_logits",
        confidence_field: str = "confidence",
        min_confidence: float = 0.0,
    ):
        self.csv_path = Path(str(csv_path)).expanduser().resolve()
        self.path_field = str(path_field)
        self.angle_field = str(angle_field)
        self.class_field = str(class_field)
        self.confidence_field = str(confidence_field)
        self.min_confidence = float(min_confidence)
        self.by_path: Dict[str, torch.Tensor] = {}
        self.by_name: Dict[str, torch.Tensor] = {}
        self._load()

    @staticmethod
    def _norm_path(value: str) -> str:
        return str(Path(value).expanduser().resolve()).replace("\\", "/").lower()

    @staticmethod
    def _angle_to_vec(angle_deg: float) -> torch.Tensor:
        rad = np.deg2rad(float(angle_deg))
        return torch.tensor([float(np.cos(rad)), float(np.sin(rad))], dtype=torch.float32)

    def _row_to_vec(self, row: Dict[str, str]) -> Optional[torch.Tensor]:
        angle_val = str(row.get(self.angle_field, "") or "").strip()
        if angle_val:
            try:
                return self._angle_to_vec(float(angle_val))
            except ValueError:
                pass

        class_val = str(row.get(self.class_field, "") or "").strip().lower()
        if class_val in self.CLASS_NAME_TO_ANGLE:
            return self._angle_to_vec(self.CLASS_NAME_TO_ANGLE[class_val])

        return None

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"External orientation CSV not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                path_val = str(row.get(self.path_field, "") or "").strip()
                if not path_val:
                    continue

                if self.confidence_field:
                    conf_val = str(row.get(self.confidence_field, "") or "").strip()
                    if conf_val:
                        try:
                            if float(conf_val) < self.min_confidence:
                                continue
                        except ValueError:
                            pass

                vec = self._row_to_vec(row)
                if vec is None:
                    continue

                key = self._norm_path(path_val)
                if key not in self.by_path:
                    self.by_path[key] = vec

                name_key = Path(path_val).name.lower()
                if name_key not in self.by_name:
                    self.by_name[name_key] = vec

    def get_batch(
        self,
        paths: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not paths:
            return None

        vectors: List[torch.Tensor] = []
        found = 0
        for raw_path in paths:
            key = self._norm_path(str(raw_path))
            vec = self.by_path.get(key)
            if vec is None:
                vec = self.by_name.get(Path(str(raw_path)).name.lower())
            if vec is None:
                vec = torch.zeros(2, dtype=torch.float32)
            else:
                found += 1
            vectors.append(vec)

        if found == 0:
            return None

        return torch.stack(vectors, dim=0).to(device=device, dtype=dtype)


class _MEBOWNode(dict):
    """Small config node compatible with the official MEBOW HRNet code."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _mebow_node(value):
    if isinstance(value, dict):
        return _MEBOWNode({key: _mebow_node(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_mebow_node(item) for item in value]
    return value


def _build_mebow_cfg() -> _MEBOWNode:
    """Return the official COCO-MEBOW HRNet-W32 inference architecture config."""
    return cast(_MEBOWNode, _mebow_node(
        {
            "MODEL": {
                "USE_FEATUREMAP": True,
                "NAME": "pose_hrnet",
                "INIT_WEIGHTS": False,
                "PRETRAINED": "",
                "NUM_JOINTS": 17,
                "TARGET_TYPE": "gaussian",
                "IMAGE_SIZE": [192, 256],
                "HEATMAP_SIZE": [48, 64],
                "SIGMA": 2,
                "EXTRA": {
                    "PRETRAINED_LAYERS": [
                        "conv1",
                        "bn1",
                        "conv2",
                        "bn2",
                        "layer1",
                        "transition1",
                        "stage2",
                        "transition2",
                        "stage3",
                        "transition3",
                        "stage4",
                    ],
                    "FINAL_CONV_KERNEL": 1,
                    "STAGE2": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4],
                        "NUM_CHANNELS": [32, 64],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE3": {
                        "NUM_MODULES": 4,
                        "NUM_BRANCHES": 3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4],
                        "NUM_CHANNELS": [32, 64, 128],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE4": {
                        "NUM_MODULES": 3,
                        "NUM_BRANCHES": 4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4, 4],
                        "NUM_CHANNELS": [32, 64, 128, 256],
                        "FUSE_METHOD": "SUM",
                    },
                },
            }
        }
    ))


_MEBOW_POSE_MODULE = None


def _load_mebow_pose_module():
    """Load the vendored official MEBOW pose_hrnet.py without polluting imports."""
    global _MEBOW_POSE_MODULE
    if _MEBOW_POSE_MODULE is not None:
        return _MEBOW_POSE_MODULE

    repo_root = Path(__file__).resolve().parent / "third_party" / "mebow_official"
    pose_path = repo_root / "lib" / "models" / "pose_hrnet.py"
    if not pose_path.exists():
        raise FileNotFoundError(
            f"Official MEBOW model file not found: {pose_path}. "
            "Clone https://github.com/ChenyanWu/MEBOW into third_party/mebow_official."
        )

    spec = importlib.util.spec_from_file_location("_official_mebow_pose_hrnet", str(pose_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official MEBOW pose_hrnet from {pose_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MEBOW_POSE_MODULE = module
    return module


def _resolve_local_path(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()

    module_dir = Path(__file__).resolve().parent
    candidates = [
        (Path.cwd() / path).resolve(),
        (module_dir / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


class StripeOrientationEstimator:
    """MEBOW HBOE estimator with ellipse-derived stripe orientations.

    The official MEBOW network predicts a 72-bin body-orientation distribution
    with 5-degree bins on the full person crop.  The original visualization code
    draws this angle with a +90 degree offset; that vector is the concrete
    front/torso-facing direction used here for FiLM conditioning.

    The person is approximated by a top-view ellipse with 10 anatomical
    sectors. This is the visible-sector geometry path shared with FiLM.

    We select the most camera-visible sectors using the same camera-facing
    geometry as the demo script, order them left-to-right in the image, then
    return anatomy-relative sector normals for FiLM conditioning.
    """

    BODY_SECTORS_SEMANTIC10: Tuple[Tuple[str, str, float, float], ...] = (
        ("front", "front", -25.0, 25.0),
        ("front-left", "front-left", 25.0, 60.0),
        ("left side front", "left-front", 60.0, 90.0),
        ("left side back", "left-back", 90.0, 120.0),
        ("back-left", "back-left", 120.0, 155.0),
        ("back", "back", 155.0, 205.0),
        ("back-right", "back-right", 205.0, 240.0),
        ("right side back", "right-back", 240.0, 270.0),
        ("right side front", "right-front", 270.0, 300.0),
        ("front-right", "front-right", 300.0, 335.0),
    )

    BODY_SECTORS_SEMANTIC8_MERGED_SIDES: Tuple[Tuple[str, str, float, float], ...] = (
        ("front", "front", -25.0, 25.0),
        ("front-left", "front-left", 25.0, 60.0),
        ("left", "left", 60.0, 120.0),
        ("back-left", "back-left", 120.0, 155.0),
        ("back", "back", 155.0, 205.0),
        ("back-right", "back-right", 205.0, 240.0),
        ("right", "right", 240.0, 300.0),
        ("front-right", "front-right", 300.0, 335.0),
    )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        num_classes: int = 5,
        stripe_height: int = 256,
        stripe_width: int = 128,
        sector_mode: str = "semantic10",
        output_frame: str = "relative",
    ):
        self.checkpoint_path = _resolve_local_path(str(checkpoint_path))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MEBOW HBOE checkpoint not found: {self.checkpoint_path}. "
                "Download the official model_hboe.pth and place it under "
                "third_party/mebow_official/models/."
            )

        self.device = torch.device(device)
        self.num_classes = 72
        self.stripe_height = int(stripe_height)
        self.stripe_width = int(stripe_width)
        self.ellipse_a = 1.0
        self.ellipse_b = 0.7
        self.sector_mode = str(sector_mode).strip().lower()
        self.output_frame = str(output_frame).strip().lower()
        if self.output_frame not in {"relative", "absolute_visible"}:
            raise ValueError(
                "Unsupported output_frame for StripeOrientationEstimator: "
                f"{self.output_frame}. Expected 'relative' or 'absolute_visible'."
            )
        self.body_sectors = self._resolve_body_sectors(self.sector_mode)
        self.ellipse_num_segments = len(self.body_sectors)

        pose_hrnet = _load_mebow_pose_module()
        self.model = pose_hrnet.get_pose_net(_build_mebow_cfg(), is_train=False).to(self.device)
        ckpt = self._load_checkpoint(self.checkpoint_path, self.device)
        state = self._extract_state_dict(ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        angles_deg = torch.arange(self.num_classes, dtype=torch.float32) * (360.0 / self.num_classes)
        raw_angles = torch.deg2rad(angles_deg)
        # Kept for compatibility with visualization scripts: this is the raw
        # MEBOW bin angle before the official +90 degree drawing convention.
        self.bin_basis = torch.stack([torch.cos(raw_angles), torch.sin(raw_angles)], dim=1).to(self.device)

        official_angles = torch.deg2rad(angles_deg + 90.0)
        self.official_bin_basis = torch.stack(
            [torch.cos(official_angles), torch.sin(official_angles)],
            dim=1,
        ).to(self.device)
        self.sector_relative_normals, self.sector_local_points = self._build_semantic_sector_geometry(
            self.body_sectors,
            self.ellipse_a,
            self.ellipse_b,
            self.device,
        )

    @classmethod
    def _resolve_body_sectors(
        cls,
        sector_mode: str,
    ) -> Tuple[Tuple[str, str, float, float], ...]:
        if sector_mode == "semantic10":
            return cls.BODY_SECTORS_SEMANTIC10
        if sector_mode in {"semantic8_merge_sides", "semantic8", "merged_sides"}:
            return cls.BODY_SECTORS_SEMANTIC8_MERGED_SIDES
        raise ValueError(
            "Unsupported sector_mode for StripeOrientationEstimator: "
            f"{sector_mode}. Expected 'semantic10' or 'semantic8_merge_sides'."
        )

    @staticmethod
    def _load_checkpoint(path: Path, device: torch.device):
        try:
            return torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            # Older PyTorch versions do not expose the weights_only argument.
            return torch.load(str(path), map_location=device)

    @staticmethod
    def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("MEBOW checkpoint must be a state_dict or contain a state_dict-like entry")
        return {
            (key[7:] if str(key).startswith("module.") else key): value
            for key, value in checkpoint.items()
        }

    @staticmethod
    def _build_semantic_sector_geometry(
        body_sectors: Tuple[Tuple[str, str, float, float], ...],
        a: float,
        b: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        centers = torch.tensor(
            [(start + end) * 0.5 for _name, _short, start, end in body_sectors],
            dtype=torch.float32,
            device=device,
        )
        rel_rad = torch.deg2rad(centers)
        rel_normals = torch.stack([torch.cos(rel_rad), torch.sin(rel_rad)], dim=1)

        # Demo convention: local +Y is the anatomical front.  A body-relative
        # normal of 0 deg therefore corresponds to the local ellipse normal
        # angle 90 deg in the standard x/y ellipse frame.
        local_normal_rad = torch.deg2rad(centers + 90.0)
        t = torch.atan2(float(b) * torch.sin(local_normal_rad), float(a) * torch.cos(local_normal_rad))
        local_points = torch.stack([float(a) * torch.cos(t), float(b) * torch.sin(t)], dim=1)
        return F.normalize(rel_normals, dim=1, eps=1e-6), local_points

    def _mebow_probs_to_body_orientation(self, hoe_probs: torch.Tensor) -> torch.Tensor:
        orientation = torch.matmul(
            hoe_probs.float(),
            self.official_bin_basis.to(device=hoe_probs.device, dtype=hoe_probs.dtype),
        )
        low_conf = orientation.norm(dim=1) < 1e-6
        if low_conf.any():
            bins = torch.argmax(hoe_probs[low_conf], dim=1)
            orientation[low_conf] = self.official_bin_basis[bins].to(
                device=orientation.device,
                dtype=orientation.dtype,
            )
        return F.normalize(orientation, dim=1, eps=1e-6)

    def predict_body_orientation(
        self,
        images: torch.Tensor,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Predict the global MEBOW body-orientation vector for full person crops."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                mebow_input = F.interpolate(
                    images.to(device=self.device, dtype=torch.float32),
                    size=(self.stripe_height, self.stripe_width),
                    mode="bilinear",
                    align_corners=False,
                )
                _pose_heatmaps, hoe_probs = self.model(mebow_input)
                orientation = self._mebow_probs_to_body_orientation(hoe_probs)
                return orientation.to(device=images.device, dtype=output_dtype)
        finally:
            if was_training:
                self.model.train()

    def _ellipse_visible_stripe_orientations(
        self,
        body_orientation: torch.Tensor,
        num_stripes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        body_orientation = F.normalize(body_orientation.float(), dim=1, eps=1e-6)
        sector_relative_normals = self.sector_relative_normals.to(
            device=body_orientation.device,
            dtype=body_orientation.dtype,
        )
        sector_local_points = self.sector_local_points.to(
            device=body_orientation.device,
            dtype=body_orientation.dtype,
        )

        # body_orientation already includes the official MEBOW +90 degree
        # convention and is treated as the anatomical front direction.
        fx = body_orientation[:, 0].unsqueeze(1)
        fy = body_orientation[:, 1].unsqueeze(1)
        rx = sector_relative_normals[:, 0].unsqueeze(0)
        ry = sector_relative_normals[:, 1].unsqueeze(0)
        world_normals = torch.stack(
            [
                fx * rx - fy * ry,
                fy * rx + fx * ry,
            ],
            dim=2,
        )

        # Visible body normals are mirrored with respect to the image vertical
        # axis before matching against the fixed camera-facing normal.
        visible_world_normals = torch.stack(
            [-world_normals[:, :, 0], world_normals[:, :, 1]],
            dim=2,
        )

        # The camera view direction is +Y (bottom of image looking upward).
        # The visible surface normal points back toward the camera, so it is -Y.
        visible_surface_normal = torch.tensor(
            [0.0, -1.0],
            device=body_orientation.device,
            dtype=body_orientation.dtype,
        )

        # Select the num_stripes sectors whose mirrored normals point most
        # toward the camera, then sort those selected sectors left-to-right in
        # the image. This reproduces the semantic10 demo behavior.
        visibility_scores = torch.matmul(visible_world_normals, visible_surface_normal)
        selected_unsorted = torch.topk(visibility_scores, k=int(num_stripes), dim=1, largest=True).indices

        # In the demo, the ellipse is rotated by front-90.  The x coordinate of
        # each local ellipse point after that rotation is:
        # x' = x * sin(front) + y * cos(front).
        local_x = sector_local_points[:, 0].unsqueeze(0)
        local_y = sector_local_points[:, 1].unsqueeze(0)
        projected_x = local_x * fy + local_y * fx
        selected_x = torch.gather(projected_x, dim=1, index=selected_unsorted)
        order = torch.argsort(selected_x, dim=1)
        segment_idx = torch.gather(selected_unsorted, dim=1, index=order)

        gather_idx = segment_idx.unsqueeze(2).expand(-1, -1, 2)
        abs_bank = visible_world_normals
        selected_abs_normals = torch.gather(abs_bank, dim=1, index=gather_idx)
        selected_abs_normals = F.normalize(selected_abs_normals, dim=2, eps=1e-6)

        rel_bank = sector_relative_normals.unsqueeze(0).expand(body_orientation.size(0), -1, -1)
        selected_body_relative_normals = torch.gather(rel_bank, dim=1, index=gather_idx)
        selected_body_relative_normals = F.normalize(selected_body_relative_normals, dim=2, eps=1e-6)
        return selected_abs_normals, selected_body_relative_normals

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        num_stripes: int,
        output_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return both absolute-visible and relative stripe vectors as (B,C,4)."""
        if images.dim() != 4:
            raise ValueError(f"Expected images as (B,3,H,W), got {tuple(images.shape)}")
        if int(num_stripes) <= 0:
            return None
        if int(num_stripes) > self.ellipse_num_segments:
            raise ValueError(
                f"num_stripes={int(num_stripes)} cannot exceed "
                f"ellipse_num_segments={self.ellipse_num_segments}"
            )

        _bsz, _channels, _height, width = images.shape
        n_stripes = int(num_stripes)
        if width <= 0:
            return None

        mebow_input = F.interpolate(
            images,
            size=(self.stripe_height, self.stripe_width),
            mode="bilinear",
            align_corners=False,
        ).to(device=self.device, dtype=torch.float32)
        _pose_heatmaps, hoe_probs = self.model(mebow_input)
        body_orientation = self._mebow_probs_to_body_orientation(hoe_probs)
        abs_orientation, rel_orientation = self._ellipse_visible_stripe_orientations(body_orientation, n_stripes)
        orientation = torch.cat([abs_orientation, rel_orientation], dim=2)
        return orientation.to(device=images.device, dtype=output_dtype)
    
    def to_device(self, tensor: Union[torch.Tensor, List, Tuple], 
                  non_blocking: bool = True) -> Union[torch.Tensor, List, Tuple]:
        """Envoie vers le device avec vérification de type."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device, non_blocking=non_blocking)
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(self.to_device(t, non_blocking) for t in tensor)
        return tensor


# =============================================================================
# Logging
# =============================================================================

class MetricsLogger:
    """Logger structuré pour les métriques d'entraînement."""
    
    def __init__(self, log_dir: str, run_name: str, csv_flush_interval: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_flush_interval = max(1, int(csv_flush_interval))
        
        # Logger Python
        self.logger = self._setup_logger()
        
        # Fichiers CSV
        self.train_csv = self.log_dir / f"train_{run_name}.csv"
        self.eval_csv = self.log_dir / f"eval_{run_name}.csv"
        
        # Headers
        self.train_header = [
            "epoch", "batch", "loss", "lr", "time", "mem_alloc", "mem_reserved"
        ]
        self.eval_header = [
            "epoch", "mAP", "Rank-1", "Rank-5", "Rank-10",
            "mAP_stripe", "Rank-1_stripe", "eval_time"
        ]
        self._train_buffer = []
        self._eval_buffer = []
        
        self._init_csv_files()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"doc_{self.run_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console (stdout pour éviter que PowerShell remonte un NativeCommandError)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # Fichier
        fh = logging.FileHandler(self.log_dir / f"{self.run_name}.log",
                                  encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Ne pas propager vers le root logger (qui écrit par défaut sur stderr)
        logger.propagate = False
        
        return logger
    
    def _init_csv_files(self):
        """Crée les fichiers CSV avec headers si nécessaire."""
        if not self.train_csv.exists():
            with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.train_header)
        
        if not self.eval_csv.exists():
            with open(self.eval_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.eval_header)

    @staticmethod
    def _append_csv_rows(path: Path, rows):
        if not rows:
            return
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    def flush_train(self):
        """Ecrit les lignes train bufferisees sur disque."""
        if not self._train_buffer:
            return
        rows = self._train_buffer
        self._train_buffer = []
        self._append_csv_rows(self.train_csv, rows)

    def flush_eval(self):
        """Ecrit les lignes eval bufferisees sur disque."""
        if not self._eval_buffer:
            return
        rows = self._eval_buffer
        self._eval_buffer = []
        self._append_csv_rows(self.eval_csv, rows)

    def flush(self):
        """Force l'ecriture des buffers CSV."""
        self.flush_train()
        self.flush_eval()
    
    def log_train_step(self, epoch: int, batch: int, loss: float, lr: float, 
                       step_time: float):
        """Log un step d'entraînement."""
        mem_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        
        row = [epoch, batch, f"{loss:.6f}", f"{lr:.8f}", f"{step_time:.3f}",
               f"{mem_alloc:.2f}", f"{mem_reserved:.2f}"]
        
        self._train_buffer.append(row)
        if len(self._train_buffer) >= self.csv_flush_interval:
            self.flush_train()
    
    def log_epoch(self, epoch: int, avg_loss: float, lr: float, epoch_time: float,
                  metrics: Optional[Dict] = None):
        """Log de fin d'époque."""
        msg = f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s"
        if metrics:
            msg += f" | mAP: {metrics.get('mAP', 0):.2%}"
        self.logger.info(msg)
        self.flush_train()
    
    def log_eval(self, epoch: int, metrics: Dict, eval_time: float):
        """Log résultats d'évaluation."""
        row = [
            epoch,
            f"{metrics.get('mAP', 0):.6f}",
            f"{metrics.get('Rank-1', 0):.6f}",
            f"{metrics.get('Rank-5', 0):.6f}",
            f"{metrics.get('Rank-10', 0):.6f}",
            f"{metrics.get('mAP_stripe', 0):.6f}",
            f"{metrics.get('Rank-1_stripe', 0):.6f}",
            f"{eval_time:.1f}"
        ]
        
        self._eval_buffer.append(row)
        self.flush_eval()
        
        self.logger.info(
            f"Eval @ Epoch {epoch} | Global mAP: {metrics.get('mAP', 0):.2%} | "
            f"Stripe mAP: {metrics.get('mAP_stripe', 0):.2%} | "
            f"Time: {eval_time:.1f}s"
        )
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)

    def close(self):
        self.flush()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


# =============================================================================
# Métriques de Distance (Optimisées)
# =============================================================================

@torch.no_grad()
def cosine_distmat_chunked(qf: torch.Tensor, gf: torch.Tensor, 
                           device: torch.device, chunk_size: int = 256) -> torch.Tensor:
    """
    Calcule la matrice de distance cosinus par chunks.
    Optimisé avec pré-allocation et normalisation lazy.
    """
    qf = qf.float()
    gf = gf.float()
    
    Nq, Ng = qf.size(0), gf.size(0)
    dist = torch.empty((Nq, Ng), dtype=torch.float32, device="cpu")
    
    # Normalisation une seule fois
    qf_norm = F.normalize(qf, dim=1)
    gf_norm = F.normalize(gf, dim=1)
    
    for i in range(0, Nq, chunk_size):
        end_i = min(Nq, i + chunk_size)
        q_chunk = qf_norm[i:end_i].to(device, non_blocking=True)
        
        for j in range(0, Ng, chunk_size):
            end_j = min(Ng, j + chunk_size)
            g_chunk = gf_norm[j:end_j].to(device, non_blocking=True)
            
            # Distance cosinus: 1 - cosine_similarity
            sim = q_chunk @ g_chunk.t()
            dist[i:end_i, j:end_j] = (1.0 - sim).cpu()
    
    return dist


@torch.no_grad()
def cosine_topk_chunked(qf: torch.Tensor, gf: torch.Tensor,
                        device: torch.device, k: int,
                        chunk_q: int = 128, chunk_g: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k distance sans matrice complète.
    Utilise un heap running pour économiser mémoire.
    """
    k = min(k, gf.size(0))
    Nq, Ng = qf.size(0), gf.size(0)
    
    # Normalisation
    qf_norm = F.normalize(qf.float(), dim=1)
    gf_norm = F.normalize(gf.float(), dim=1)
    
    dist_topk = torch.empty((Nq, k), dtype=torch.float32)
    idx_topk = torch.empty((Nq, k), dtype=torch.long)
    
    for qs in range(0, Nq, chunk_q):
        qe = min(Nq, qs + chunk_q)
        q = qf_norm[qs:qe].to(device)
        
        # Heap local pour ce bloc de queries
        best_vals = torch.full((q.size(0), k), float("inf"), device=device)
        best_idxs = torch.zeros((q.size(0), k), dtype=torch.long, device=device)
        
        for gs in range(0, Ng, chunk_g):
            ge = min(Ng, gs + chunk_g)
            g = gf_norm[gs:ge].to(device)
            
            # Similarité pour ce chunk
            d_chunk = 1.0 - (q @ g.t())  # (chunk_q, chunk_g)
            
            # Merge avec le heap courant
            all_vals = torch.cat([best_vals, d_chunk], dim=1)  # (chunk_q, k + chunk_g)
            all_idxs = torch.cat([
                best_idxs,
                torch.arange(gs, ge, device=device).expand(q.size(0), -1)
            ], dim=1)
            
            # Top-k sur la dimension fusionnée
            best_vals, pos = torch.topk(all_vals, k, dim=1, largest=False, sorted=True)
            best_idxs = torch.gather(all_idxs, 1, pos)
        
        dist_topk[qs:qe] = best_vals.cpu()
        idx_topk[qs:qe] = best_idxs.cpu()
    
    return dist_topk, idx_topk


# =============================================================================
# CMC/mAP Metrics (Corrigés et optimisés)
# =============================================================================

def compute_cmc_map(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
                    q_camids: np.ndarray, g_camids: np.ndarray,
                    ranks: Tuple[int, ...] = (1, 5, 10),
                    remove_same_cam: bool = False,
                    logger: Optional[logging.Logger] = None,
                    progress_label: str = "cmc_map",
                    progress_every: int = 5000) -> Dict[str, float]:
    """
    Calcule CMC et mAP avec gestion correcte des caméras.
    """
    num_q = distmat.shape[0]
    max_rank = max(ranks)
    
    cmc = np.zeros(max_rank, dtype=np.float64)
    aps = []
    t0 = time.time()
    did_log_start = False

    if logger is not None and num_q > 0:
        logger.info(
            f"{progress_label}: starting metric aggregation on {num_q} queries"
        )
        did_log_start = True
    
    for i in range(num_q):
        # Tri par distance croissante
        order = np.argsort(distmat[i])
        
        # Filtrage same-cam si demandé
        if remove_same_cam:
            keep = ~((g_pids[order] == q_pids[i]) & (g_camids[order] == q_camids[i]))
            order = order[keep]
        
        # Matches binaires
        matches = (g_pids[order] == q_pids[i]).astype(np.int32)
        
        if matches.sum() == 0:
            continue  # Pas de ground truth pour cette query
        
        # CMC: première position où on trouve un match
        first_match = np.where(matches == 1)[0][0]
        if first_match < max_rank:
            cmc[first_match:] += 1
        
        # AP: aire sous la courbe precision-recall
        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(len(matches)) + 1)
        recall = cum_matches / matches.sum()
        
        # Interpolation pour mAP
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precision, recall):
            if r > prev_recall:
                ap += p * (r - prev_recall)
                prev_recall = r
        
        aps.append(ap)

        if logger is not None and progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == num_q):
            elapsed = time.time() - t0
            qps = (i + 1) / max(1e-9, elapsed)
            eta = (num_q - (i + 1)) / max(1e-9, qps)
            logger.info(
                f"{progress_label}: {i + 1}/{num_q} queries "
                f"({100.0 * (i + 1) / num_q:.2f}%) | "
                f"valid={len(aps)} | elapsed={elapsed:.1f}s | eta={eta:.1f}s"
            )
    
    if not aps:
        if logger is not None and did_log_start:
            logger.info(f"{progress_label}: completed with 0 valid queries")
        res = {f"Rank-{r}": 0.0 for r in ranks}
        res.update({"mAP": 0.0})
        return res
    
    cmc = cmc / len(aps)
    if logger is not None and did_log_start:
        logger.info(
            f"{progress_label}: completed in {time.time() - t0:.1f}s "
            f"with {len(aps)}/{num_q} valid queries"
        )
    
    res = {f"Rank-{r}": float(cmc[min(r, max_rank)-1]) for r in ranks}
    res.update({"mAP": float(np.mean(aps))})
    return res


def compute_cmc_map_topk(dist_topk: torch.Tensor, idx_topk: torch.Tensor,
                         q_pids: np.ndarray, g_pids: np.ndarray,
                         q_camids: np.ndarray, g_camids: np.ndarray,
                         ranks: Tuple[int, ...] = (1, 5, 10),
                         remove_same_cam: bool = False,
                         assume_sorted: bool = True) -> Dict[str, float]:
    """
    CMC/mAP sur une liste top-k pré-calculée.
    """
    dist = dist_topk.cpu().numpy()
    idx = idx_topk.cpu().numpy()

    if assume_sorted:
        if dist.shape[1] > 1 and not np.all(dist[:, :-1] <= dist[:, 1:] + 1e-7):
            raise ValueError("compute_cmc_map_topk expected sorted top-k distances")
        idx_sorted = idx
    else:
        order_local = np.argsort(dist, axis=1)
        idx_sorted = np.take_along_axis(idx, order_local, axis=1)
    
    # Récupérer les PIDs et CamIDs correspondants
    g_pids_expanded = g_pids[idx_sorted]
    g_camids_expanded = g_camids[idx_sorted]
    
    max_rank = min(max(ranks), dist.shape[1])
    cmc = np.zeros(max_rank, dtype=np.float64)
    aps = []
    
    for i in range(len(q_pids)):
        matches = (g_pids_expanded[i] == q_pids[i]).astype(np.int32)
        
        if remove_same_cam:
            same_cam_mask = (g_camids_expanded[i] == q_camids[i])
            matches = matches & (~same_cam_mask)
        
        if matches.sum() == 0:
            continue
        
        first_match = np.where(matches == 1)[0]
        if len(first_match) > 0 and first_match[0] < max_rank:
            cmc[first_match[0]:] += 1
        
        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(len(matches)) + 1)
        recall = cum_matches / matches.sum()
        
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precision, recall):
            if r > prev_recall:
                ap += p * (r - prev_recall)
                prev_recall = r
        
        aps.append(ap)
    
    if not aps:
        res = {f"Rank-{r}": 0.0 for r in ranks}
        res.update({"mAP": 0.0})
        return res
    
    cmc = cmc / len(aps)
    res = {f"Rank-{r}": float(cmc[min(r, max_rank)-1]) for r in ranks}
    res.update({"mAP": float(np.mean(aps))})
    return res


# =============================================================================
# Helpers divers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Crée le dossier si nécessaire et retourne le Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: Union[str, Path], model_state: dict, 
                    optimizer_state: Optional[dict] = None,
                    scaler_state: Optional[dict] = None,
                    scheduler_state: Optional[dict] = None,
                    epoch: int = 0, metrics: Optional[Dict] = None,
                    cfg_dict: Optional[dict] = None,
                    pid2label: Optional[dict] = None,
                    best_metric: Optional[float] = None):
    """
    Sauvegarde sécurisée de checkpoint (JSON-safe).
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "metrics": metrics or {},
        "config": cfg_dict or {},
        "pid2label": pid2label or {},
    }
    
    if optimizer_state:
        checkpoint["optimizer_state"] = optimizer_state
    if scaler_state:
        checkpoint["scaler_state"] = scaler_state
    if scheduler_state:
        checkpoint["scheduler_state"] = scheduler_state
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    
    # Sauvegarde atomique
    path = Path(path)
    temp_path = path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> dict:
    """Charge un checkpoint avec vérification."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trouvé: {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint


class EarlyStopping:
    """Early stopping avec sauvegarde du meilleur modèle."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "max" pour mAP, "min" pour loss
        self.counter = 0
        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        if self.mode == "max":
            improved = metric > self.best_value + self.min_delta
        else:
            improved = metric < self.best_value - self.min_delta
        
        if improved:
            self.best_value = metric
            self.counter = 0
            return True  # Meilleur modèle
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Timer:
    """Context manager pour mesurer le temps."""
    
    def __init__(self, name: str = "Operation", logger: Optional[MetricsLogger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = 0.0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        self.elapsed = time.time() - self.start_time
        msg = f"{self.name} terminé en {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
