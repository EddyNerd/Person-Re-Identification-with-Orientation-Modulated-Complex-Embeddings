import argparse
import os
import sys
import gc
import json
from pathlib import Path
from copy import deepcopy
import itertools
from typing import List, Callable, TypeVar

import torch
from torch.utils.data import DataLoader
from dataclasses import replace

# Imports optimisés
from config_optimized import CFG, apply_preset, apply_matching_code, PRESET_TO_RECOMMENDED_EVAL_CODES  # type: ignore
from utils_optimized import DeviceManager, MetricsLogger, ensure_dir, Timer, save_checkpoint, load_checkpoint  # type: ignore
from data_optimized import (
    load_split_json, ReidDataset, PKSampler, build_transforms,
    build_query_gallery_from_mars, load_tracks_from_mat, 
    build_query_gallery_from_tracks
)
from model_optimized import DOCModel  # type: ignore
from trainer import Trainer
from evaluator import Evaluator  # type: ignore


T = TypeVar("T")


def parse_list(s: str, dtype: Callable[[str], T] = str) -> List[T]:
    """Parse une liste depuis string comma-separated."""
    if not s or s.strip() == "":
        return []
    return [dtype(x.strip()) for x in s.split(",")]


def build_data_loaders(cfg: CFG, dm: DeviceManager, logger):
    """
    Construit tous les loaders nécessaires.
    
    Returns:
        cfg (éventuellement ajusté), train_loader, (val_q_loader, val_g_loader) ou None, pid2label
    """
    pin_memory, _ = dm.get_io_flags(cfg.pin_memory, cfg.use_amp)
    
    # === Données d'entraînement ===
    if cfg.train_tracks_mat:
        train_items = load_tracks_from_mat(
            cfg.train_tracks_mat, cfg.train_list, cfg.data_root,
            subset="bbox_train", strict=cfg.strict_paths,
            allow_pid_fallback=cfg.allow_pid_fallback,
            keep_distractors=cfg.keep_distractors
        )
    else:
        train_items = load_split_json(
            cfg.train_list, cfg.data_root, strict=cfg.strict_paths,
            allow_pid_fallback=cfg.allow_pid_fallback,
            keep_distractors=cfg.keep_distractors
        )
    
    logger.info(f"Loaded {len(train_items)} training samples")
    
    # Mapping PID -> label
    pids = sorted(list(set([x[1] for x in train_items])))
    pid2label = {pid: i for i, pid in enumerate(pids)}
    num_classes = len(pids)
    
    # Dataset et loader train
    train_tf = build_transforms(cfg.height, cfg.width, is_train=True)
    train_ds = ReidDataset(train_items, pid2label, train_tf, two_view=cfg.use_L_aug)
    
    sampler = PKSampler(train_items, cfg.P, cfg.K, cfg.steps_per_epoch)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )
    
    # === Données de validation ===
    val_loaders = None
    
    if not cfg.train_only:  # construit les loaders de validation sauf si train_only=True
        val_q_items = []
        val_g_items = []
        
        # Méthode 1: Query/Gallery lists explicites
        if cfg.val_query_list and cfg.val_gallery_list:
            val_q_items = load_split_json(
                cfg.val_query_list, cfg.data_root, strict=cfg.strict_paths,
                allow_pid_fallback=cfg.allow_pid_fallback,
                keep_distractors=cfg.keep_distractors
            )
            val_g_items = load_split_json(
                cfg.val_gallery_list, cfg.data_root, strict=cfg.strict_paths,
                allow_pid_fallback=cfg.allow_pid_fallback,
                keep_distractors=cfg.keep_distractors
            )
        
        # Méthode 2: Tracks MAT
        elif cfg.test_tracks_mat and cfg.test_list and cfg.query_idx_path:
            val_q_items, val_g_items = build_query_gallery_from_tracks(
                cfg.test_tracks_mat, cfg.test_list, cfg.query_idx_path,
                cfg.data_root, subset="bbox_test", strict=cfg.strict_paths,
                allow_pid_fallback=cfg.allow_pid_fallback,
                keep_distractors=cfg.keep_distractors,
                use_all_as_query=cfg.use_all_as_query,
            )
        
        # Méthode 3: MARS standard
        elif cfg.test_list and cfg.query_idx_path:
            # Si on veut les frames complètes des tracklets query et que tracks sont dispo, basculer sur la méthode tracks
            if cfg.use_query_track_frames and cfg.test_tracks_mat:
                val_q_items, val_g_items = build_query_gallery_from_tracks(
                    cfg.test_tracks_mat, cfg.test_list, cfg.query_idx_path,
                    cfg.data_root, subset="bbox_test", strict=cfg.strict_paths,
                    allow_pid_fallback=cfg.allow_pid_fallback,
                    keep_distractors=cfg.keep_distractors,
                    use_all_as_query=cfg.use_all_as_query,
                )
            else:
                val_q_items, val_g_items = build_query_gallery_from_mars(
                    cfg.test_list, cfg.query_idx_path, cfg.data_root,
                    keep_distractors=cfg.keep_distractors,
                    use_all_as_query=cfg.use_all_as_query,
                )
        
        if val_q_items and val_g_items:
            logger.info(f"Validation: {len(val_q_items)} queries, {len(val_g_items)} gallery")
            
            val_tf = build_transforms(cfg.height, cfg.width, is_train=False)
            
            val_q_ds = ReidDataset(val_q_items, None, val_tf)
            val_g_ds = ReidDataset(val_g_items, None, val_tf)
            
            val_q_loader = DataLoader(
                val_q_ds, batch_size=128, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=pin_memory
            )
            val_g_loader = DataLoader(
                val_g_ds, batch_size=128, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=pin_memory
            )
            
            val_loaders = (val_q_loader, val_g_loader)
            
            # Forcer l'évaluation en top-k pour éviter toute matrice complète
            if cfg.eval_topk <= 0:
                cfg = replace(cfg, eval_topk=min(1000, len(val_g_items)))
                logger.info(f"[Eval] eval_topk fixé à {cfg.eval_topk} pour éviter la matrice complète")
            elif cfg.eval_topk > len(val_g_items):
                cfg = replace(cfg, eval_topk=len(val_g_items))
                logger.info(f"[Eval] eval_topk borné à la taille gallery ({cfg.eval_topk})")
            # Aligner stripe_topk si nécessaire
            if getattr(cfg, "stripe_topk", 0) <= 0 or cfg.stripe_topk > len(val_g_items):
                cfg = replace(cfg, stripe_topk=cfg.eval_topk)
    
    return cfg, train_loader, val_loaders, pid2label, num_classes


def run_single_training(cfg: CFG, run_name: str = "single_run"):
    """
    Entraînement single run complet.
    """
    # Assainir l'état mémoire avant de démarrer un nouveau run
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    # Normaliser out_dir en chemin absolu pour éviter les checkpoints introuvables
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        # Ancre les chemins de sortie sur le projet (cwd), pas sur data_root,
        # pour éviter d'écrire les checkpoints dans le dossier du dataset.
        base_root = Path(".").resolve()
        out_dir = base_root / out_dir
        cfg = replace(cfg, out_dir=str(out_dir))

    # Setup
    out_dir = ensure_dir(cfg.out_dir)
    logger = MetricsLogger(str(out_dir), run_name)
    
    dm = DeviceManager(cfg.device)
    dm.configure_reproducibility(cfg.seed, cfg.deterministic, cfg.cudnn_benchmark)
    
    logger.info(f"=== Training Run: {run_name} ===")
    logger.info(f"Device: {dm.device}")
    if dm.device.type != "cuda":
        logger.warning("Training/eval will run on CPU. Set cfg.device='cuda' or pass --device cuda to force GPU.")
    logger.info(f"Config: {cfg}")
    
    # Data
    with Timer("Data loading", logger):
        cfg, train_loader, val_loaders, pid2label, num_classes = build_data_loaders(cfg, dm, logger)
    
    # Pas de matrice complète : neutralise tout chemin/reservoir éventuel
    if not bool(getattr(cfg, "always_full_distance_train", False)):
        cfg = replace(cfg, save_full_dist="", force_full_dist=False)
    
    # Model & Training
    trainer = Trainer(cfg, dm, logger)
    trainer.setup(num_classes)
    setattr(trainer, "pid2label", pid2label)  # Pour sauvegarde
    
    # Evaluator si données de validation
    evaluator = None
    if val_loaders:
        evaluator = Evaluator(cfg, dm, logger)
    
    # Entraînement
    with Timer("Training", logger):
        history = trainer.train(train_loader, val_loaders, evaluator)

    # Sauvegarde de l'historique (loss & métriques)
    history_path = Path(cfg.out_dir) / "history.json"
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved training history to {history_path}")
    except Exception as e:
        logger.warning(f"Could not save training history to {history_path}: {e}")
    
    return history


def run_mix_plan(args):
    """
    Plan de mix: entraîne plusieurs presets, évalue avec plusieurs codes.
    """
    presets = parse_list(args.mix_presets) or ["A1", "EXP5_FiLM_Hermitian"]
    root_path = Path(args.root).expanduser().resolve()
    results_file = root_path / "mixplan_results.csv"
    
    print(f"=== MIX PLAN: Presets {presets} ===")
    
    for preset in presets:
        print(f"\n--- Training Preset {preset} ---")
        
        # 1. Entraînement
        cfg = CFG()
        cfg = replace(cfg, data_root=str(root_path))
        cfg = apply_preset(cfg, preset)
        cfg = replace(cfg, out_dir=str(root_path / f"runs_{preset}"))

        history = run_single_training(cfg, run_name=preset)
        best_map = history["best_metric"]
        
        # 2. Inférence ablation
        print(f"--- Inference Ablation on {preset} ---")
        
        # Chargement du meilleur checkpoint
        ckpt_path = f"{cfg.out_dir}/best_model.pth"
        if not Path(ckpt_path).exists():
            ckpt_path = f"{cfg.out_dir}/last_model.pth"
        
        if not Path(ckpt_path).exists():
            print(f"[Warn] No checkpoint found for {preset}, skipping ablation")
            continue
        
        # Setup évaluation
        dm = DeviceManager(cfg.device)
        logger = MetricsLogger(cfg.out_dir, f"{preset}_eval")
        
        cfg, _train_loader, val_loaders, _pid2label, num_classes = build_data_loaders(cfg, dm, logger)
        if not val_loaders:
            print(f"[Warn] No validation data for {preset}")
            continue
        
        # Test avec différents codes
        preset_key = (preset or "").strip().upper()
        if "/" in preset_key:
            preset_key = preset_key.split("/")[0]
        codes = parse_list(args.mix_eval_codes) or PRESET_TO_RECOMMENDED_EVAL_CODES.get(preset_key, ["G0", "O0"])

        for code in codes:
            try:
                eval_cfg = apply_matching_code(cfg, code)
                evaluator = Evaluator(eval_cfg, dm, logger)

                metrics = evaluator.evaluate_checkpoint(
                    ckpt_path, val_loaders[0], val_loaders[1], num_classes
                )
            except Exception as e:
                print(f"[Error] {preset}+{code}: {e}")
                continue

            # Sauvegarde résultats
            row = {
                "train_preset": preset,
                "eval_code": code,
                "train_mAP": f"{best_map:.2%}",
                "eval_mAP": f"{metrics.get('mAP', 0):.2%}",
                "eval_mAP_stripe": f"{metrics.get('mAP_stripe', 0):.2%}",
                "eval_R1": f"{metrics.get('Rank-1', 0):.2%}",
                "eval_R1_stripe": f"{metrics.get('Rank-1_stripe', 0):.2%}",
            }
            
            with open(results_file, "a", newline="") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            
            print(f"  {preset} + {code}: mAP={metrics.get('mAP', 0):.2%} "
                  f"mAP_s={metrics.get('mAP_stripe', 0):.2%}")

        # Nettoyage explicite entre presets pour éviter accumulation CUDA/hooks
        gc.collect()
        if dm.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_grid_search(args):
    """
    Grid search sur les hyperparamètres.
    """
    root_path = Path(args.root).expanduser().resolve()
    cfg = CFG()
    cfg = replace(cfg, data_root=str(root_path))
    cfg = apply_preset(cfg, args.preset)
    
    results_file = root_path / "grid_search_results.csv"
    
    # Grilles
    w_set_list = parse_list(args.grid_w_set, float) or [1.0]
    w_att_list = parse_list(args.grid_w_att, float) or [0.5]
    c_list = parse_list(args.grid_C, int) or [cfg.C_stripes]
    r_list = parse_list(args.grid_R, int) or [cfg.R_rows]
    
    combinations = list(itertools.product(w_set_list, w_att_list, c_list, r_list))
    print(f"=== GRID SEARCH: {len(combinations)} configurations ===")
    
    for i, (ws, wa, cs, rr) in enumerate(combinations):
        run_name = f"grid_{i:03d}_wset{ws}_watt{wa}_C{cs}_R{rr}"
        print(f"\n[{i+1}/{len(combinations)}] {run_name}")
        
        # Override config
        grid_cfg = deepcopy(cfg)
        grid_cfg = replace(grid_cfg,
            w_setNCE=ws,
            w_attach=wa,
            C_stripes=cs,
            R_rows=max(1, rr),
            skip_stripe_eval=False,
            out_dir=f"{args.root}/grid_search"
        )
        
        # Entraînement
        try:
            history = run_single_training(grid_cfg, run_name)
            best_map = history["best_metric"]
        except Exception as e:
            print(f"  ERROR: {e}")
            best_map = 0.0
        
        # Log
        row = {
            "run": run_name,
            "w_setNCE": ws,
            "w_attach": wa,
            "C_stripes": cs,
            "R_rows": rr,
            "best_mAP": f"{best_map:.2%}",
        }
        
        with open(results_file, "a", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="DOC Training")
    
    # General
    parser.add_argument("--root", default="data", help="Data root directory")
    parser.add_argument("--preset", default="EXP5_FiLM_Hermitian", help="Training preset")
    parser.add_argument("--config", default="", help="Config JSON file to load")
    
    # Modes
    parser.add_argument("--mix_plan", action="store_true", help="Run mix plan")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate checkpoint")
    parser.add_argument("--checkpoint", default="", help="Checkpoint for eval_only")
    
    # Mix plan args
    parser.add_argument("--mix_presets", default="A1,EXP5_FiLM_Hermitian", help="Presets to train")
    parser.add_argument("--mix_eval_codes", default="", help="Override eval codes")
    
    # Grid search args
    parser.add_argument("--grid_w_set", default="0.5,1.0,2.0", help="w_setNCE values")
    parser.add_argument("--grid_w_att", default="0.1,0.5,1.0", help="w_attach values")
    parser.add_argument("--grid_C", default="", help="C values (empty = use preset)")
    parser.add_argument("--grid_R", default="", help="R values (empty = use preset)")
    
    args = parser.parse_args()

    # Racine normalisée
    root_path = Path(args.root).expanduser().resolve()
    
    # Chargement config personnalisée
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        base_cfg = CFG.from_dict(config_dict)
    else:
        base_cfg = CFG()

    base_cfg = replace(base_cfg, data_root=str(root_path))
    
    # Mode évaluation seule
    if args.eval_only:
        if not args.checkpoint:
            print("ERROR: --checkpoint required for --eval_only")
            sys.exit(1)

        eval_codes = parse_list(args.mix_eval_codes) or [""]

        for code in eval_codes:
            dm = DeviceManager(base_cfg.device)

            cfg = apply_preset(base_cfg, args.preset)
            if code:
                cfg = apply_matching_code(cfg, code)

            # Logger dédié dans le dossier du preset
            run_name = f"eval_{args.preset}" + (f"_{code}" if code else "")
            logger = MetricsLogger(cfg.out_dir, run_name)
            logger.info(f"Device: {dm.device}")
            if dm.device.type != "cuda":
                logger.warning("Evaluation running on CPU. Set --device cuda (or CFG.device='cuda') to force GPU.")

            cfg, _, val_loaders, _, num_classes = build_data_loaders(cfg, dm, logger)
            # Neutraliser toute matrice complète
            if not bool(getattr(cfg, "always_full_distance_train", False)):
                cfg = replace(cfg, save_full_dist="", force_full_dist=False)
            if not val_loaders:
                print(f"ERROR: No validation data found for {run_name}")
                continue

            evaluator = Evaluator(cfg, dm, logger)
            metrics = evaluator.evaluate_checkpoint(
                args.checkpoint, val_loaders[0], val_loaders[1], num_classes
            )

            print(f"\n=== Evaluation Results ({run_name}) ===")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return
    
    # Modes d'entraînement
    if args.mix_plan:
        run_mix_plan(args)
    elif args.grid_search:
        run_grid_search(args)
    else:
        # Single run
        cfg = apply_preset(base_cfg, args.preset)
        run_single_training(cfg, run_name=f"single_{args.preset}")


if __name__ == "__main__":
    if os.name == "nt":
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    main()
