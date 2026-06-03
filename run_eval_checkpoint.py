import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from config_optimized import CFG  # type: ignore
from evaluator import Evaluator  # type: ignore
from train_main import build_data_loaders  # type: ignore
from utils_optimized import DeviceManager, MetricsLogger  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint in a fresh process")
    ap.add_argument("--cfg_json", required=True, help="Path to serialized CFG JSON")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    ap.add_argument("--metrics_out", required=True, help="Output JSON file for metrics")
    ap.add_argument("--run_name", default="eval_checkpoint", help="Run name for logs")
    ap.add_argument("--device", default="", help="Optional device override, e.g. cuda or cpu")
    args = ap.parse_args()

    cfg_path = Path(args.cfg_json)
    ckpt_path = Path(args.checkpoint)
    metrics_out = Path(args.metrics_out)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # This script is eval-only: disable training-time auto-forcing logic.
    cfg_dict["always_full_distance_train"] = False

    cfg = CFG.from_dict(cfg_dict)
    if args.device.strip():
        cfg = replace(cfg, device=args.device.strip())

    # Eval process should only evaluate.
    cfg = replace(cfg, train_only=False, eval_during_train=False)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(str(out_dir), f"{args.run_name}_eval")

    dm = DeviceManager(cfg.device)
    dm.configure_reproducibility(cfg.seed, cfg.deterministic, cfg.cudnn_benchmark)

    logger.info("=== Evaluation Process Start ===")
    logger.info(f"Config JSON: {cfg_path}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Metrics output: {metrics_out}")
    logger.info(f"Device resolved: {dm.device}")
    use_uniform_ot = bool(getattr(cfg, "use_uniform_ot_marginals", False))
    if bool(getattr(cfg, "use_cell_ot_matching", False)):
        ot_marginal_source = "cell_gamma"
    elif bool(getattr(cfg, "use_stripe_ot_matching", False)) and use_uniform_ot:
        ot_marginal_source = "uniform"
    elif bool(getattr(cfg, "use_stripe_ot_matching", False)) and bool(getattr(cfg, "use_gamma_weights_for_matching", False)):
        ot_marginal_source = "gnn_gamma_mean"
    elif bool(getattr(cfg, "use_stripe_ot_matching", False)):
        ot_marginal_source = "beta"
    else:
        ot_marginal_source = "n/a"
    logger.info(
        "Eval options: "
        f"force_full_dist_stripe={bool(getattr(cfg, 'force_full_dist_stripe', False))} | "
        f"use_cell_ot={bool(getattr(cfg, 'use_cell_ot_matching', False))} | "
        f"use_stripe_ot={bool(getattr(cfg, 'use_stripe_ot_matching', False))} | "
        f"ot_eps={float(getattr(cfg, 'ot_epsilon', 0.1))} | "
        f"ot_iters={int(getattr(cfg, 'ot_num_iters', 100))} | "
        f"crg_lambda={float(getattr(cfg, 'crg_lambda', 0.5))} | "
        f"uniform_ot_marginals={use_uniform_ot} | "
        f"ot_marginal_source={ot_marginal_source} | "
        f"cross_view={bool(getattr(cfg, 'use_cross_view_consistency', False))} | "
        f"cross_view_row={bool(getattr(cfg, 'use_cross_view_row_consistency', False))} | "
        f"cross_view_row_weight={float(getattr(cfg, 'cross_view_row_weight', 0.0)):.3f}"
    )

    start = time.time()
    cfg, _train_loader, val_loaders, _pid2label, num_classes = build_data_loaders(cfg, dm, logger)
    if not val_loaders:
        raise RuntimeError("No validation loaders available for evaluation")

    q_loader, g_loader = val_loaders
    evaluator = Evaluator(cfg, dm, logger)
    metrics = evaluator.evaluate_checkpoint(str(ckpt_path), q_loader, g_loader, num_classes=num_classes)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    elapsed = max(0, int(time.time() - start))
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Evaluation completed in {h:02d}:{m:02d}:{s:02d}")
    logger.info(f"Metrics written to: {metrics_out}")

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    import os

    if os.name == "nt":
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    main()
