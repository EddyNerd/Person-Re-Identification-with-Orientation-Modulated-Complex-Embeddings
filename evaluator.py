
import torch
import time
import torch.nn.functional as F
from dataclasses import replace
from typing import Tuple, Dict, Optional, Any
import numpy as np
from pathlib import Path

import utils_optimized as _utils  # type: ignore

cosine_topk_chunked = getattr(_utils, "cosine_topk_chunked")
compute_cmc_map = getattr(_utils, "compute_cmc_map")
compute_cmc_map_topk = getattr(_utils, "compute_cmc_map_topk")
Timer = getattr(_utils, "Timer")
ExternalOrientationProvider = getattr(_utils, "ExternalOrientationProvider")
StripeOrientationEstimator = getattr(_utils, "StripeOrientationEstimator")


class Evaluator:
    """
    ?valuateur dédié pour la Re-ID avec support stripe et global.
    """
    
    def __init__(self, cfg, device_manager, logger=None):
        self.cfg = cfg
        self.device = device_manager.device
        self.dm = device_manager
        self.logger = logger
        self.orientation_provider: Optional[Any] = None
        self.stripe_orientation_estimator: Optional[Any] = None

        use_external_orientation = (
            bool(getattr(self.cfg, "use_residual_film_orientation", False))
            and str(getattr(self.cfg, "film_orientation_source", "view_prototype")) == "external"
        )
        if use_external_orientation:
            csv_path = str(getattr(self.cfg, "film_external_orientation_csv", "") or "").strip()
            if not csv_path:
                raise ValueError(
                    "film_orientation_source='external' requires film_external_orientation_csv in config"
                )
            _op = ExternalOrientationProvider(
                csv_path=csv_path,
                path_field=str(getattr(self.cfg, "film_external_path_field", "image_path")),
                angle_field=str(getattr(self.cfg, "film_external_angle_field", "pred_angle_deg")),
                class_field=str(getattr(self.cfg, "film_external_class_field", "pred_class_logits")),
                confidence_field=str(getattr(self.cfg, "film_external_confidence_field", "confidence")),
                min_confidence=float(getattr(self.cfg, "film_external_min_confidence", 0.0)),
            )
            self.orientation_provider = _op
            if self.logger:
                self.logger.info(
                    f"Loaded external orientation provider from {_op.csv_path} "
                    f"with {len(_op.by_path)} indexed paths"
                )

        use_stripe_estimator = (
            bool(getattr(self.cfg, "use_residual_film_orientation", False))
            and str(getattr(self.cfg, "film_orientation_source", "view_prototype")) == "stripe_estimator"
        )
        if use_stripe_estimator:
            ckpt_path = str(getattr(self.cfg, "film_stripe_estimator_checkpoint", "") or "").strip()
            if not ckpt_path:
                raise ValueError(
                    "film_orientation_source='stripe_estimator' requires film_stripe_estimator_checkpoint in config"
                )
            _se = StripeOrientationEstimator(
                checkpoint_path=ckpt_path,
                device=self.device,
                num_classes=int(getattr(self.cfg, "film_stripe_estimator_num_classes", 5)),
                stripe_height=int(getattr(self.cfg, "film_stripe_estimator_height", 256)),
                stripe_width=int(getattr(self.cfg, "film_stripe_estimator_width", 128)),
                sector_mode=str(getattr(self.cfg, "film_stripe_estimator_sector_mode", "semantic10")),
            )
            self.stripe_orientation_estimator = _se
            if self.logger:
                self.logger.info(
                    f"Loaded stripe orientation estimator from {_se.checkpoint_path}"
                )
        
        # Import différé pour éviter circular imports
        from model_optimized import DOCModel  # type: ignore

    def _resolve_ot_modes(self) -> Tuple[bool, bool]:
        use_cell = bool(getattr(self.cfg, "use_cell_ot_matching", False))
        stripe_ot = getattr(self.cfg, "use_stripe_ot_matching", None)
        if stripe_ot is None:
            # Legacy behavior: old use_cell_ot_matching toggled stripe OT.
            return False, use_cell
        return use_cell, bool(stripe_ot)

    def _needs_cross_view_cells(self) -> bool:
        return (
            bool(getattr(self.cfg, "use_cross_view_consistency", False))
            and bool(getattr(self.cfg, "use_cross_view_row_consistency", False))
            and float(getattr(self.cfg, "cross_view_row_weight", 0.0)) > 0.0
        )


    def _resolve_cache_root(self) -> Path:
        """Resolve local cache root for reusable evaluation matrices."""
        cache_dir = str(getattr(self.cfg, "cache_dir", "") or "").strip()
        if cache_dir:
            root = Path(cache_dir).expanduser()
            if not root.is_absolute():
                root = Path(getattr(self.cfg, "out_dir", ".")).expanduser().resolve() / root
        else:
            root = Path(getattr(self.cfg, "out_dir", ".")).expanduser().resolve() / "reid_matrices"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_full_dist_stripe_path(self) -> Optional[Path]:
        """Resolve output path for full stripe distance matrix."""
        override = getattr(self.cfg, "full_dist_stripe_run_override", "")
        run_name = override.strip() if override.strip() else Path(getattr(self.cfg, "out_dir", "run")).name
        path = self._resolve_cache_root() / run_name / "full_dist_stripe.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_full_dist_global_path(self) -> Optional[Path]:
        """Resolve output path for full global distance matrix."""
        override = getattr(self.cfg, "full_dist_global_run_override", "")
        run_name = override.strip() if override.strip() else Path(getattr(self.cfg, "out_dir", "run")).name
        path = self._resolve_cache_root() / run_name / "full_dist_global.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _maybe_reuse_full_dist(self, save_path: Optional[Path], shape: Tuple[int, int], kind: str) -> Optional[np.ndarray]:
        """
        Reuse an existing full distance matrix on disk when enabled and compatible.
        """
        if not bool(getattr(self.cfg, "reuse_full_dist_if_exists", True)):
            return None
        if save_path is None or not save_path.exists():
            return None
        try:
            dist = np.load(save_path, mmap_mode="r")
            if tuple(dist.shape) != tuple(shape):
                if self.logger:
                    self.logger.warning(
                        f"Existing {kind} full distance matrix shape mismatch at {save_path}: "
                        f"found {tuple(dist.shape)}, expected {tuple(shape)}. Recomputing."
                    )
                return None
            if self.logger:
                self.logger.info(f"Reusing existing {kind} full distance matrix from {save_path}")
            return dist
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Failed to reuse existing {kind} full distance matrix at {save_path}: {exc}")
            return None

    def _cleanup_cuda_cache(self) -> None:
        """Best-effort CUDA cleanup after a failed or heavy chunked computation."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format duration in seconds as HH:MM:SS."""
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _gpu_mem_str(self) -> str:
        """Return a short CUDA memory snapshot for logs."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "gpu_mem=n/a"
        try:
            alloc = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            max_alloc = torch.cuda.max_memory_allocated(self.device) / 1e9
            return f"gpu_mem alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={max_alloc:.2f}GB"
        except Exception:
            return "gpu_mem=unavailable"

    def _compute_stripe_full_dist_memmap_impl(
        self,
        qE,
        qModAbs,
        qRel_src,
        qO,
        qU,
        qGamma,
        gE,
        gModAbs,
        gRel_src,
        gO,
        gU,
        gGamma,
        save_path: Path,
        *,
        compute_device: torch.device,
        chunk_q: int,
        chunk_g: int,
        rel_are_normalized: bool,
    ) -> np.ndarray:
        """Chunked full stripe distance matrix computation on a specific device."""
        from numpy.lib.format import open_memmap
        from loss_optimized import set_to_set_similarity  # type: ignore
        use_complex = bool(getattr(self.cfg, "use_complex_hermitian_embedding", False))
        use_orientation_ot = bool(getattr(self.cfg, "use_orientation_guided_ot", False))

        Nq = qE.size(0)
        Ng = gE.size(0)
        use_non_blocking = compute_device.type == "cuda"

        if self.logger:
            self.logger.info(
                f"Stripe full-dist compute on {compute_device} "
                f"(chunk_q={chunk_q}, chunk_g={chunk_g})"
            )

        dist_mm = open_memmap(save_path, mode="w+", dtype=np.float32, shape=(Nq, Ng))
        total_bytes = Nq * Ng * np.dtype(np.float32).itemsize
        q_chunk_count = max(1, (Nq + chunk_q - 1) // chunk_q)
        stage_start = time.time()

        for q_chunk_idx, i0 in enumerate(range(0, Nq, chunk_q), start=1):
            i1 = min(Nq, i0 + chunk_q)

            Eq_chunk = qE[i0:i1].to(compute_device, non_blocking=use_non_blocking)
            Rq_chunk = (
                qRel_src[i0:i1].to(compute_device, non_blocking=use_non_blocking)
                if qRel_src is not None else None
            )
            Oq_chunk = qO[i0:i1].to(compute_device, non_blocking=use_non_blocking)
            Uq_chunk = qU[i0:i1].to(compute_device, non_blocking=use_non_blocking) if qU is not None else None
            Gq_chunk = qGamma[i0:i1].to(compute_device, non_blocking=use_non_blocking) if qGamma is not None else None
            Zq_chunk = None
            if use_complex and hasattr(self, "_last_extracted_Z_q") and self._last_extracted_Z_q is not None:
                Zq_chunk = self._last_extracted_Z_q[i0:i1].to(compute_device, non_blocking=use_non_blocking)
            Orient_q_chunk = None
            if use_orientation_ot and hasattr(self, "_last_extracted_orient_q") and self._last_extracted_orient_q is not None:
                Orient_q_chunk = self._last_extracted_orient_q[i0:i1].to(compute_device, non_blocking=use_non_blocking)

            for j0 in range(0, Ng, chunk_g):
                j1 = min(Ng, j0 + chunk_g)

                Eg_chunk = gE[j0:j1].to(compute_device, non_blocking=use_non_blocking)
                Rg_chunk = (
                    gRel_src[j0:j1].to(compute_device, non_blocking=use_non_blocking)
                    if gRel_src is not None else None
                )
                Og_chunk = gO[j0:j1].to(compute_device, non_blocking=use_non_blocking)
                Ug_chunk = gU[j0:j1].to(compute_device, non_blocking=use_non_blocking) if gU is not None else None
                Gg_chunk = gGamma[j0:j1].to(compute_device, non_blocking=use_non_blocking) if gGamma is not None else None
                Zg_chunk = None
                if use_complex and hasattr(self, "_last_extracted_Z_g") and self._last_extracted_Z_g is not None:
                    Zg_chunk = self._last_extracted_Z_g[j0:j1].to(compute_device, non_blocking=use_non_blocking)
                Orient_g_chunk = None
                if use_orientation_ot and hasattr(self, "_last_extracted_orient_g") and self._last_extracted_orient_g is not None:
                    Orient_g_chunk = self._last_extracted_orient_g[j0:j1].to(compute_device, non_blocking=use_non_blocking)

                use_cell_ot, use_stripe_ot = self._resolve_ot_modes()
                S = set_to_set_similarity(
                    Eq_chunk, Rq_chunk, Oq_chunk,
                    Eg_chunk, Rg_chunk, Og_chunk,
                    sim_mode=self.cfg.sim_mode,
                    alpha=self.cfg.alpha_mix,
                    temp=self.cfg.set_match_temp,
                    use_wp=self.cfg.use_wp_in_agg,
                    use_zscore=self.cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(self.cfg, "zscore_kappa", 2.5),
                    crg_lambda=float(getattr(self.cfg, "crg_lambda", 0.5)),
                    Z_q=Zq_chunk,
                    Z_g=Zg_chunk,
                    Orient_q=Orient_q_chunk,
                    Orient_g=Orient_g_chunk,
                    use_orientation_guided_ot=use_orientation_ot,
                    orientation_ot_cost_weight=float(getattr(self.cfg, "orientation_ot_cost_weight", 0.05)),
                    orientation_ot_mass_weight=float(getattr(self.cfg, "orientation_ot_mass_weight", 1.0)),
                    rel_are_normalized=rel_are_normalized,
                    use_cell_ot=use_cell_ot,
                    use_stripe_ot=use_stripe_ot,
                    uniform_ot_marginals=getattr(self.cfg, "use_uniform_ot_marginals", False),
                    ot_epsilon=getattr(self.cfg, "ot_epsilon", 0.1),
                    ot_num_iters=getattr(self.cfg, "ot_num_iters", 100),
                    ot_margi_eps=getattr(self.cfg, "ot_margi_eps", 1e-9),
                    use_cross_view_consistency=bool(getattr(self.cfg, "use_cross_view_consistency", False)),
                    cross_view_alpha=float(getattr(self.cfg, "cross_view_alpha", 0.7)),
                    cross_view_pos_lambda=float(getattr(self.cfg, "cross_view_pos_lambda", 0.75)),
                    cross_view_phi_scale=float(getattr(self.cfg, "cross_view_phi_scale", 8.0)),
                    cross_view_phi_bias=float(getattr(self.cfg, "cross_view_phi_bias", 0.5)),
                    cross_view_norm_transitive=bool(getattr(self.cfg, "cross_view_norm_transitive", True)),
                    Uq_cells=Uq_chunk,
                    Ug_cells=Ug_chunk,
                    Gamma_q=Gq_chunk,
                    Gamma_g=Gg_chunk,
                    use_cross_view_row_consistency=bool(getattr(self.cfg, "use_cross_view_row_consistency", False)),
                    cross_view_row_weight=float(getattr(self.cfg, "cross_view_row_weight", 0.0)),
                    cross_view_row_pos_lambda=float(getattr(self.cfg, "cross_view_row_pos_lambda", 0.75)),
                    use_view_prototype_propagation=bool(getattr(self.cfg, "use_view_prototype_propagation", False)),
                    view_prototype_path=str(getattr(self.cfg, "view_prototype_path", "")),
                    view_propagation_lambda=float(getattr(self.cfg, "view_propagation_lambda", 0.15)),
                    view_prototype_temp=float(getattr(self.cfg, "view_prototype_temp", 10.0)),
                    use_view_prototype_span=bool(getattr(self.cfg, "use_view_prototype_span", False)),
                    view_prototype_span_lambda=float(getattr(self.cfg, "view_prototype_span_lambda", 1e-3)),
                    view_transition_self=float(getattr(self.cfg, "view_transition_self", 1.0)),
                    view_transition_neighbor1=float(getattr(self.cfg, "view_transition_neighbor1", 0.7)),
                    view_transition_neighbor2=float(getattr(self.cfg, "view_transition_neighbor2", 0.2)),
                    use_view_uncertainty_gate=bool(getattr(self.cfg, "use_view_uncertainty_gate", True)),
                )

                dist_mm[i0:i1, j0:j1] = (1.0 - S).float().cpu().numpy()
                del Eg_chunk, Rg_chunk, Og_chunk, Ug_chunk, Gg_chunk, Zg_chunk, Orient_g_chunk, S

            del Eq_chunk, Rq_chunk, Oq_chunk, Uq_chunk, Gq_chunk, Zq_chunk, Orient_q_chunk
            dist_mm.flush()

            if self.logger and (q_chunk_idx % 10 == 0 or i1 == Nq):
                pct = 100.0 * i1 / max(1, Nq)
                done_bytes = i1 * Ng * np.dtype(np.float32).itemsize
                elapsed = max(time.time() - stage_start, 1e-6)
                rows_per_sec = i1 / elapsed
                pairs_per_sec = (i1 * Ng) / elapsed
                rows_left = max(0, Nq - i1)
                eta_sec = rows_left / max(rows_per_sec, 1e-6)
                self.logger.info(
                    "  stripe full-dist progress: "
                    f"{q_chunk_idx}/{q_chunk_count} q-chunks | "
                    f"rows {i1}/{Nq} ({pct:.2f}%) | "
                    f"approx matrix written {done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB | "
                    f"speed {rows_per_sec:.1f} rows/s ({pairs_per_sec / 1e6:.2f}M pairs/s) | "
                    f"elapsed {self._fmt_duration(elapsed)} | eta {self._fmt_duration(eta_sec)} | "
                    f"{self._gpu_mem_str()}"
                )

            if compute_device.type == "cuda":
                self._cleanup_cuda_cache()

        return dist_mm

    def _compute_global_full_dist_memmap_impl(
        self,
        qf: torch.Tensor,
        gf: torch.Tensor,
        save_path: Path,
        *,
        compute_device: torch.device,
        chunk_q: int,
        chunk_g: int,
    ) -> np.ndarray:
        """Chunked full global distance matrix computation on a specific device."""
        from numpy.lib.format import open_memmap

        Nq = qf.size(0)
        Ng = gf.size(0)
        use_non_blocking = compute_device.type == "cuda"

        if self.logger:
            self.logger.info(
                f"Global full-dist compute on {compute_device} "
                f"(chunk_q={chunk_q}, chunk_g={chunk_g})"
            )

        dist_mm = open_memmap(save_path, mode="w+", dtype=np.float32, shape=(Nq, Ng))
        total_bytes = Nq * Ng * np.dtype(np.float32).itemsize
        q_chunk_count = max(1, (Nq + chunk_q - 1) // chunk_q)
        stage_start = time.time()

        qf_norm = F.normalize(qf.float(), dim=1)
        gf_norm = F.normalize(gf.float(), dim=1)

        for q_chunk_idx, i0 in enumerate(range(0, Nq, chunk_q), start=1):
            i1 = min(Nq, i0 + chunk_q)
            q_chunk = qf_norm[i0:i1].to(compute_device, non_blocking=use_non_blocking)

            for j0 in range(0, Ng, chunk_g):
                j1 = min(Ng, j0 + chunk_g)
                g_chunk = gf_norm[j0:j1].to(compute_device, non_blocking=use_non_blocking)

                sim = torch.matmul(q_chunk, g_chunk.t())
                dist_mm[i0:i1, j0:j1] = (1.0 - sim).float().cpu().numpy()
                del g_chunk, sim

            del q_chunk
            dist_mm.flush()

            if self.logger and (q_chunk_idx % 10 == 0 or i1 == Nq):
                pct = 100.0 * i1 / max(1, Nq)
                done_bytes = i1 * Ng * np.dtype(np.float32).itemsize
                elapsed = max(time.time() - stage_start, 1e-6)
                rows_per_sec = i1 / elapsed
                pairs_per_sec = (i1 * Ng) / elapsed
                rows_left = max(0, Nq - i1)
                eta_sec = rows_left / max(rows_per_sec, 1e-6)
                self.logger.info(
                    "  global full-dist progress: "
                    f"{q_chunk_idx}/{q_chunk_count} q-chunks | "
                    f"rows {i1}/{Nq} ({pct:.2f}%) | "
                    f"approx matrix written {done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB | "
                    f"speed {rows_per_sec:.1f} rows/s ({pairs_per_sec / 1e6:.2f}M pairs/s) | "
                    f"elapsed {self._fmt_duration(elapsed)} | eta {self._fmt_duration(eta_sec)} | "
                    f"{self._gpu_mem_str()}"
                )

            if compute_device.type == "cuda":
                self._cleanup_cuda_cache()

        return dist_mm

    def compute_global_full_dist_memmap(self, qf: torch.Tensor, gf: torch.Tensor, save_path: Path) -> np.ndarray:
        """Compute full global distance matrix in chunks and write to a .npy memmap."""
        attempts = [(self.device, 128, 256)]
        if self.device.type == "cuda":
            attempts.extend([
                (self.device, 64, 128),
                (torch.device("cpu"), 32, 64),
            ])

        for attempt_idx, (compute_device, chunk_q, chunk_g) in enumerate(attempts, start=1):
            try:
                return self._compute_global_full_dist_memmap_impl(
                    qf, gf, save_path,
                    compute_device=compute_device,
                    chunk_q=chunk_q,
                    chunk_g=chunk_g,
                )
            except RuntimeError as exc:
                msg = str(exc)
                is_cuda_related = compute_device.type == "cuda" and any(
                    token in msg.lower()
                    for token in ("cuda", "cublas", "cudnn", "out of memory")
                )
                last_attempt = attempt_idx == len(attempts)

                if not is_cuda_related or last_attempt:
                    raise

                next_device, next_q, next_g = attempts[attempt_idx]
                if self.logger:
                    self.logger.warning(
                        f"Global full-dist failed on {compute_device} with "
                        f"chunk_q={chunk_q}, chunk_g={chunk_g}: {exc}. "
                        f"Retrying on {next_device} with chunk_q={next_q}, chunk_g={next_g}."
                    )
                self._cleanup_cuda_cache()

        raise RuntimeError("Unexpected failure while computing full global distance matrix.")

    def compute_stripe_full_dist_memmap(
        self,
        qE,
        qModAbs,
        qRel,
        qO,
        gE,
        gModAbs,
        gRel,
        gO,
        save_path: Path,
        qU=None,
        qGamma=None,
        gU=None,
        gGamma=None,
    ) -> np.ndarray:
        """
        Compute full stripe distance matrix in chunks and write to a .npy memmap.
        Returns a numpy memmap-backed array of shape (Nq, Ng).
        """
        rel_are_normalized = False
        qRel_src = qRel
        gRel_src = gRel
        if qRel is not None and gRel is not None and qRel.dim() == 2 and gRel.dim() == 2:
            qRel_src = F.normalize(qRel.float(), dim=1)
            gRel_src = F.normalize(gRel.float(), dim=1)
            rel_are_normalized = True

        attempts = [(self.device, 64, 256)]
        if self.device.type == "cuda":
            attempts.extend([
                (self.device, 32, 128),
                (torch.device("cpu"), 16, 64),
            ])

        for attempt_idx, (compute_device, chunk_q, chunk_g) in enumerate(attempts, start=1):
            try:
                return self._compute_stripe_full_dist_memmap_impl(
                    qE, qModAbs, qRel_src, qO, qU, qGamma, gE, gModAbs, gRel_src, gO, gU, gGamma, save_path,
                    compute_device=compute_device,
                    chunk_q=chunk_q,
                    chunk_g=chunk_g,
                    rel_are_normalized=rel_are_normalized,
                )
            except RuntimeError as exc:
                msg = str(exc)
                is_cuda_related = compute_device.type == "cuda" and any(
                    token in msg.lower()
                    for token in ("cuda", "cublas", "cudnn", "out of memory")
                )
                last_attempt = attempt_idx == len(attempts)

                if not is_cuda_related or last_attempt:
                    raise

                next_device, next_q, next_g = attempts[attempt_idx]
                if self.logger:
                    self.logger.warning(
                        f"Stripe full-dist failed on {compute_device} with "
                        f"chunk_q={chunk_q}, chunk_g={chunk_g}: {exc}. "
                        f"Retrying on {next_device} with chunk_q={next_q}, chunk_g={next_g}."
                    )
                self._cleanup_cuda_cache()

        raise RuntimeError("Unexpected failure while computing full stripe distance matrix.")

    @torch.no_grad()
    def extract_features(self, model, loader, desc: str = "Extracting", return_cells: bool = False) -> Tuple:
        """
        Extraction des features avec gestion mémoire optimisée.
        
        Returns:
            global_feats: (N, D) sur CPU
            stripe_E: (N, C, D) ou None
            stripe_R: (N, D) rel_vec ou None
            stripe_O: (N, C) ou None
            stripe_U: (N, C, R, D) if return_cells else omitted
            stripe_Gamma: (N, C, R) if return_cells else omitted
            pids: List[str]
            camids: List[int]
        """
        if self.logger:
            self.logger.info(f"{desc}...")
        
        model.eval()
        
        total_items = len(getattr(loader, "dataset", []))
        total_batches = len(loader)
        stage_start = time.time()
        write_pos = 0
        global_buf = None
        stripe_E_buf = None
        stripe_mod_abs_buf = None
        stripe_Z_buf = None  # Buffer pour embeddings complexes Hermitian
        stripe_orient_buf = None
        stripe_R_buf = None
        stripe_O_buf = None
        stripe_U_buf = None
        stripe_Gamma_buf = None
        all_pids = []
        all_camids = []

        def _ensure_capacity(buf: Optional[torch.Tensor], min_rows: int) -> torch.Tensor:
            """Grow a CPU tensor buffer if required while preserving existing rows."""
            assert buf is not None, "_ensure_capacity: buffer must be initialized before growing"
            if min_rows <= buf.size(0):
                return buf
            new_rows = max(min_rows, int(buf.size(0) * 1.5) + 1)
            grown = torch.empty((new_rows, *buf.shape[1:]), dtype=buf.dtype)
            grown[:buf.size(0)].copy_(buf)
            return grown
        
        use_complex = bool(getattr(self.cfg, "use_complex_hermitian_embedding", False))
        do_stripes = not getattr(self.cfg, "skip_stripe_eval", False)
        need_rel_vec = do_stripes or bool(getattr(self.cfg, "use_rel_vec", False))
        if bool(getattr(self.cfg, "use_relvec_global_fusion", False)):
            need_rel_vec = True
        need_stripe_extract = do_stripes or need_rel_vec
        keep_cells = bool(return_cells) and do_stripes
        desc_lower = str(desc).lower()
        cache_role = "gallery" if "gallery" in desc_lower else "query"
        
        for batch_idx, batch in enumerate(loader):
            # Support both single-view and two-view batches
            if len(batch) == 5:  # e.g., imgs, imgs2, pids, camids, idx
                imgs, _, pids, camids, paths = batch
            elif len(batch) == 4:
                imgs, pids, camids, paths = batch
            else:
                imgs, pids, camids = batch[0], batch[1], batch[2]
                paths = None
            
            imgs = imgs.to(self.device, non_blocking=True)
            
            # Forward
            _logits, g_feat, feat_map = model.forward_global(imgs)
            del _logits

            orientation_vec = None
            if self.stripe_orientation_estimator is not None:
                orientation_vec = self.stripe_orientation_estimator.predict_batch(
                    imgs,
                    num_stripes=int(getattr(self.cfg, "C_stripes", 5)),
                    output_dtype=feat_map.dtype,
                )
            elif self.orientation_provider is not None and paths is not None:
                orientation_vec = self.orientation_provider.get_batch(
                    list(paths),
                    device=imgs.device,
                    dtype=feat_map.dtype,
                )

            # Pre-initialize stripe variables; assigned below when need_stripe_extract is True.
            Ehat: Optional[torch.Tensor] = None
            Beta: Optional[torch.Tensor] = None
            Uhat: Optional[torch.Tensor] = None
            Gamma: Optional[torch.Tensor] = None
            Ehat_mod_abs: Optional[torch.Tensor] = None
            Ehat_mod_rel: Optional[torch.Tensor] = None
            rel_vec: Optional[torch.Tensor] = None
            Omega: Optional[torch.Tensor] = None
            Z: Optional[torch.Tensor] = None

            # Extraction stripes si demandé (ou requise pour rel_vec/fusion).
            if need_stripe_extract:
                if need_rel_vec:
                    if use_complex:
                        Ehat, Beta, Uhat, Gamma, Ehat_mod_abs, Ehat_mod_rel = model.extract_stripes_adaptive(
                            feat_map,
                            self.cfg.C_stripes,
                            self.cfg.R_rows,
                            self.cfg,
                            orientation_vec=orientation_vec,
                            return_cells=True,
                            return_complex=True,
                            cache_role=cache_role,
                        )
                    else:
                        Ehat, Beta, Uhat, Gamma = model.extract_stripes_adaptive(
                            feat_map,
                            self.cfg.C_stripes,
                            self.cfg.R_rows,
                            self.cfg,
                            orientation_vec=orientation_vec,
                            return_cells=True,
                            cache_role=cache_role,
                        )
                    assert Ehat is not None
                    assert Beta is not None
                    assert Uhat is not None
                    assert Gamma is not None
                    rel_vec = model.compute_hierarchical_relation_vector(Ehat, Beta, Uhat, Gamma, self.cfg)
                    if bool(getattr(self.cfg, "use_relvec_global_fusion", False)):
                        assert rel_vec is not None
                        alpha = float(getattr(self.cfg, "relvec_global_alpha", 0.5))
                        g_feat = model.fuse_global_with_relvec(
                            g_feat.float(), rel_vec.float(), alpha=alpha, normalize_out=True
                        )
                else:
                    if use_complex:
                        Ehat, Beta, Ehat_mod_abs, Ehat_mod_rel = model.extract_stripes_adaptive(
                            feat_map,
                            self.cfg.C_stripes,
                            self.cfg.R_rows,
                            self.cfg,
                            orientation_vec=orientation_vec,
                            return_complex=True,
                            cache_role=cache_role,
                        )
                    else:
                        Ehat, Beta = model.extract_stripes_adaptive(
                            feat_map,
                            self.cfg.C_stripes,
                            self.cfg.R_rows,
                            self.cfg,
                            orientation_vec=orientation_vec,
                            cache_role=cache_role,
                        )

                if do_stripes:
                    assert Ehat is not None
                    assert Beta is not None
                    Omega = model.compute_omega(
                        Ehat, Beta, self.cfg.omega_mode, Uhat=Uhat, Gamma=Gamma, cfg=self.cfg
                    )
                    # Récupérer Z complexe si disponible (branche Hermitian)
                    Z = None
                    if use_complex and bool(getattr(self.cfg, "use_complex_hermitian_Z", False)):
                        Z_q, Z_g = model.get_last_complex_embeddings()
                        Z = Z_g if cache_role == "gallery" else Z_q
                        if Z is not None:
                            Z = Z.detach()  # Détacher pour éviter retention graph

            bsz = int(g_feat.size(0))
            next_pos = write_pos + bsz
            orientation_to_store = None
            if do_stripes and orientation_vec is not None:
                if orientation_vec.dim() == 2:
                    orientation_to_store = orientation_vec.unsqueeze(1).expand(
                        -1, int(getattr(self.cfg, "C_stripes", 5)), -1
                    )
                else:
                    orientation_to_store = orientation_vec

            # First batch initializes CPU buffers; later batches append by slicing.
            if global_buf is None:
                init_rows = max(total_items, next_pos)
                g_cpu = g_feat.cpu()
                global_buf = torch.empty((init_rows, g_cpu.size(1)), dtype=g_cpu.dtype)
                global_buf[write_pos:next_pos].copy_(g_cpu)
                del g_cpu
                if do_stripes:
                    assert Ehat is not None
                    assert rel_vec is not None
                    assert Omega is not None
                    E_cpu = Ehat.cpu()
                    E_mod_abs_cpu = None
                    if use_complex:
                        assert Ehat_mod_abs is not None
                        E_mod_abs_cpu = Ehat_mod_abs.cpu()
                    R_cpu = rel_vec.cpu()
                    O_cpu = Omega.cpu()
                    Orient_cpu = orientation_to_store.detach().cpu() if orientation_to_store is not None else None
                    U_cpu = None
                    Gamma_cpu = None
                    if keep_cells:
                        assert Uhat is not None
                        assert Gamma is not None
                        U_cpu = Uhat.cpu()
                        Gamma_cpu = Gamma.cpu()
                    stripe_E_buf = torch.empty((init_rows, *E_cpu.shape[1:]), dtype=E_cpu.dtype)
                    stripe_R_buf = torch.empty((init_rows, *R_cpu.shape[1:]), dtype=R_cpu.dtype)
                    stripe_O_buf = torch.empty((init_rows, *O_cpu.shape[1:]), dtype=O_cpu.dtype)
                    if Orient_cpu is not None:
                        stripe_orient_buf = torch.empty((init_rows, *Orient_cpu.shape[1:]), dtype=Orient_cpu.dtype)
                    if use_complex:
                        assert E_mod_abs_cpu is not None
                        stripe_mod_abs_buf = torch.empty((init_rows, *E_mod_abs_cpu.shape[1:]), dtype=E_mod_abs_cpu.dtype)
                    stripe_E_buf[write_pos:next_pos].copy_(E_cpu)
                    if use_complex:
                        assert E_mod_abs_cpu is not None
                        assert stripe_mod_abs_buf is not None
                        stripe_mod_abs_buf[write_pos:next_pos].copy_(E_mod_abs_cpu)
                        # Stocker Z si disponible
                        if Z is not None:
                            Z_cpu = Z.cpu()
                            if stripe_Z_buf is None:
                                stripe_Z_buf = torch.empty((init_rows, *Z_cpu.shape[1:]), dtype=Z_cpu.dtype)
                            stripe_Z_buf[write_pos:next_pos].copy_(Z_cpu)
                            del Z_cpu
                    stripe_R_buf[write_pos:next_pos].copy_(R_cpu)
                    stripe_O_buf[write_pos:next_pos].copy_(O_cpu)
                    if Orient_cpu is not None:
                        assert stripe_orient_buf is not None
                        stripe_orient_buf[write_pos:next_pos].copy_(Orient_cpu)
                    if keep_cells:
                        assert U_cpu is not None
                        assert Gamma_cpu is not None
                        stripe_U_buf = torch.empty((init_rows, *U_cpu.shape[1:]), dtype=U_cpu.dtype)
                        stripe_Gamma_buf = torch.empty((init_rows, *Gamma_cpu.shape[1:]), dtype=Gamma_cpu.dtype)
                        stripe_U_buf[write_pos:next_pos].copy_(U_cpu)
                        stripe_Gamma_buf[write_pos:next_pos].copy_(Gamma_cpu)
                    del E_cpu, E_mod_abs_cpu, R_cpu, O_cpu, Orient_cpu, U_cpu, Gamma_cpu
            else:
                global_buf = _ensure_capacity(global_buf, next_pos)
                global_buf[write_pos:next_pos].copy_(g_feat.cpu())
                if do_stripes:
                    assert Ehat is not None
                    assert rel_vec is not None
                    assert Omega is not None
                    stripe_E_buf = _ensure_capacity(stripe_E_buf, next_pos)
                    if use_complex:
                        stripe_mod_abs_buf = _ensure_capacity(stripe_mod_abs_buf, next_pos)
                    stripe_R_buf = _ensure_capacity(stripe_R_buf, next_pos)
                    stripe_O_buf = _ensure_capacity(stripe_O_buf, next_pos)
                    orient_cpu: Optional[torch.Tensor] = None
                    if orientation_to_store is not None:
                        orient_cpu = orientation_to_store.detach().cpu()
                        if stripe_orient_buf is None:
                            assert orient_cpu is not None
                            stripe_orient_buf = torch.empty(
                                (stripe_E_buf.size(0), *orient_cpu.shape[1:]),
                                dtype=orient_cpu.dtype,
                            )
                        else:
                            stripe_orient_buf = _ensure_capacity(stripe_orient_buf, next_pos)
                    stripe_E_buf[write_pos:next_pos].copy_(Ehat.cpu())
                    if use_complex:
                        assert Ehat_mod_abs is not None
                        assert stripe_mod_abs_buf is not None
                        stripe_mod_abs_buf[write_pos:next_pos].copy_(Ehat_mod_abs.cpu())
                        # Stocker Z si disponible
                        if Z is not None:
                            stripe_Z_buf = _ensure_capacity(stripe_Z_buf, next_pos)
                            stripe_Z_buf[write_pos:next_pos].copy_(Z.cpu())
                    stripe_R_buf[write_pos:next_pos].copy_(rel_vec.cpu())
                    stripe_O_buf[write_pos:next_pos].copy_(Omega.cpu())
                    if orient_cpu is not None:
                        assert stripe_orient_buf is not None
                        stripe_orient_buf[write_pos:next_pos].copy_(orient_cpu)
                    if keep_cells:
                        assert Uhat is not None
                        assert Gamma is not None
                        stripe_U_buf = _ensure_capacity(stripe_U_buf, next_pos)
                        stripe_Gamma_buf = _ensure_capacity(stripe_Gamma_buf, next_pos)
                        stripe_U_buf[write_pos:next_pos].copy_(Uhat.cpu())
                        stripe_Gamma_buf[write_pos:next_pos].copy_(Gamma.cpu())

            write_pos = next_pos

            # Stockage métadonnées
            if isinstance(pids, torch.Tensor):
                all_pids.extend(pids.tolist())
            else:
                all_pids.extend(pids)
            
            if isinstance(camids, torch.Tensor):
                all_camids.extend(camids.tolist())
            else:
                all_camids.extend(camids)
            
            # Log périodique
            if self.logger and (batch_idx % 20 == 0 or (batch_idx + 1) == total_batches):
                done_batches = batch_idx + 1
                done_items = write_pos
                elapsed = max(time.time() - stage_start, 1e-6)
                items_per_sec = done_items / elapsed
                if total_items > 0:
                    pct = 100.0 * done_items / total_items
                    remaining_items = max(0, total_items - done_items)
                    eta_sec = remaining_items / max(items_per_sec, 1e-6)
                    prog = (
                        f"items {done_items}/{total_items} ({pct:.2f}%) | "
                        f"batches {done_batches}/{total_batches}"
                    )
                else:
                    eta_sec = 0.0
                    prog = f"batches {done_batches}/{total_batches}"
                self.logger.info(
                    f"  {desc}: {prog} | speed {items_per_sec:.1f} img/s | "
                    f"elapsed {self._fmt_duration(elapsed)} | eta {self._fmt_duration(eta_sec)} | "
                    f"{self._gpu_mem_str()}"
                )

        if global_buf is None:
            # Empty loader fallback
            global_feats = torch.empty((0, 0), dtype=torch.float32)
            stripe_E = stripe_R = stripe_O = None
            stripe_mod_abs = None
            self._last_extracted_orient = None
            self._last_extracted_Z = None
            if return_cells:
                return global_feats, stripe_E, stripe_mod_abs, stripe_R, stripe_O, None, None, all_pids, all_camids
            return global_feats, stripe_E, stripe_R, stripe_O, all_pids, all_camids

        global_feats = global_buf[:write_pos]
        
        if do_stripes:
            assert stripe_E_buf is not None
            assert stripe_R_buf is not None
            assert stripe_O_buf is not None
            stripe_E = stripe_E_buf[:write_pos]
            stripe_R = stripe_R_buf[:write_pos]
            stripe_O = stripe_O_buf[:write_pos]
            stripe_mod_abs = None
            if use_complex:
                assert stripe_mod_abs_buf is not None
                stripe_mod_abs = stripe_mod_abs_buf[:write_pos]
            stripe_U = None
            stripe_Gamma = None
            if keep_cells:
                assert stripe_U_buf is not None
                assert stripe_Gamma_buf is not None
                stripe_U = stripe_U_buf[:write_pos]
                stripe_Gamma = stripe_Gamma_buf[:write_pos]
            self._last_extracted_orient = stripe_orient_buf[:write_pos] if stripe_orient_buf is not None else None
        else:
            stripe_E = stripe_R = stripe_O = None
            stripe_mod_abs = None
            stripe_U = stripe_Gamma = None
            self._last_extracted_orient = None
        
        if return_cells:
            if use_complex:
                # Stocker Z buffer dans self pour utilisation ultérieure dans evaluate_similarity
                self._last_extracted_Z = stripe_Z_buf[:write_pos] if stripe_Z_buf is not None else None
                return global_feats, stripe_E, stripe_mod_abs, stripe_R, stripe_O, stripe_U, stripe_Gamma, all_pids, all_camids
            return global_feats, stripe_E, stripe_R, stripe_O, stripe_U, stripe_Gamma, all_pids, all_camids
        if use_complex:
            # Stocker Z buffer dans self pour utilisation ultérieure dans evaluate_similarity
            self._last_extracted_Z = stripe_Z_buf[:write_pos] if stripe_Z_buf is not None else None
            return global_feats, stripe_E, stripe_mod_abs, stripe_R, stripe_O, all_pids, all_camids
        return global_feats, stripe_E, stripe_R, stripe_O, all_pids, all_camids
    
    def compute_stripe_topk(
        self,
        qE,
        qRel,
        qO,
        gE,
        gRel,
        gO,
        k: int,
        qModAbs=None,
        gModAbs=None,
        qU=None,
        qGamma=None,
        gU=None,
        gGamma=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule top-k distances sans matrice complète.
        """
        from loss_optimized import set_to_set_similarity  # type: ignore
        use_complex = bool(getattr(self.cfg, "use_complex_hermitian_embedding", False))
        use_orientation_ot = bool(getattr(self.cfg, "use_orientation_guided_ot", False))
        
        Nq = qE.size(0)
        Ng = gE.size(0)
        k = min(k, Ng)
        
        chunk_q = 64
        chunk_g = 256
        rel_are_normalized = False
        qRel_src = qRel
        gRel_src = gRel
        if qRel is not None and gRel is not None and qRel.dim() == 2 and gRel.dim() == 2:
            qRel_src = F.normalize(qRel.float(), dim=1)
            gRel_src = F.normalize(gRel.float(), dim=1)
            rel_are_normalized = True
        
        dist_topk = torch.empty((Nq, k), dtype=torch.float32)
        idx_topk = torch.empty((Nq, k), dtype=torch.long)
        
        if self.logger:
            self.logger.info(f"Computing stripe top-{k} distances...")
        
        for i0 in range(0, Nq, chunk_q):
            i1 = min(Nq, i0 + chunk_q)
            
            Eq_chunk = qE[i0:i1].to(self.device)
            Rq_chunk = qRel_src[i0:i1].to(self.device) if qRel_src is not None else None
            Oq_chunk = qO[i0:i1].to(self.device)
            Uq_chunk = qU[i0:i1].to(self.device) if qU is not None else None
            Gq_chunk = qGamma[i0:i1].to(self.device) if qGamma is not None else None
            Orient_q_chunk = None
            if use_orientation_ot and hasattr(self, "_last_extracted_orient_q") and self._last_extracted_orient_q is not None:
                Orient_q_chunk = self._last_extracted_orient_q[i0:i1].to(self.device)
            
            # Heap running pour ce bloc de queries
            best_vals = torch.full((i1 - i0, k), float("inf"), device=self.device)
            best_idxs = torch.zeros((i1 - i0, k), dtype=torch.long, device=self.device)
            
            for j0 in range(0, Ng, chunk_g):
                j1 = min(Ng, j0 + chunk_g)
                
                Eg_chunk = gE[j0:j1].to(self.device)
                Rg_chunk = gRel_src[j0:j1].to(self.device) if gRel_src is not None else None
                Og_chunk = gO[j0:j1].to(self.device)
                Ug_chunk = gU[j0:j1].to(self.device) if gU is not None else None
                Gg_chunk = gGamma[j0:j1].to(self.device) if gGamma is not None else None
                Orient_g_chunk = None
                if use_orientation_ot and hasattr(self, "_last_extracted_orient_g") and self._last_extracted_orient_g is not None:
                    Orient_g_chunk = self._last_extracted_orient_g[j0:j1].to(self.device)
                
                # Extract complex Hermitian embeddings if available
                Zq_chunk = None
                Zg_chunk = None
                if use_complex and self._last_extracted_Z is not None:
                    # Note: qZ and gZ are extracted sequentially from extract_features(query) + extract_features(gallery)
                    # Need to map to indices properly. For now assume query is first Nq and gallery is second Ng
                    if hasattr(self, "_last_extracted_Z_q"):
                        Zq_chunk = self._last_extracted_Z_q[i0:i1].to(self.device) if self._last_extracted_Z_q is not None else None
                    if hasattr(self, "_last_extracted_Z_g"):
                        Zg_chunk = self._last_extracted_Z_g[j0:j1].to(self.device) if self._last_extracted_Z_g is not None else None
                
                use_cell_ot, use_stripe_ot = self._resolve_ot_modes()
                S = set_to_set_similarity(
                    Eq_chunk, Rq_chunk, Oq_chunk,
                    Eg_chunk, Rg_chunk, Og_chunk,
                    sim_mode=self.cfg.sim_mode,
                    alpha=self.cfg.alpha_mix,
                    temp=self.cfg.set_match_temp,
                    use_wp=self.cfg.use_wp_in_agg,
                    use_zscore=self.cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(self.cfg, "zscore_kappa", 2.5),
                    crg_lambda=float(getattr(self.cfg, "crg_lambda", 0.5)),
                    Z_q=Zq_chunk,  # Complex Hermitian embeddings query
                    Z_g=Zg_chunk,  # Complex Hermitian embeddings gallery
                    Orient_q=Orient_q_chunk,
                    Orient_g=Orient_g_chunk,
                    use_orientation_guided_ot=use_orientation_ot,
                    orientation_ot_cost_weight=float(getattr(self.cfg, "orientation_ot_cost_weight", 0.05)),
                    orientation_ot_mass_weight=float(getattr(self.cfg, "orientation_ot_mass_weight", 1.0)),
                    rel_are_normalized=rel_are_normalized,
                    use_cell_ot=use_cell_ot,
                    use_stripe_ot=use_stripe_ot,
                    uniform_ot_marginals=getattr(self.cfg, "use_uniform_ot_marginals", False),
                    ot_epsilon=getattr(self.cfg, "ot_epsilon", 0.1),
                    ot_num_iters=getattr(self.cfg, "ot_num_iters", 100),
                    ot_margi_eps=getattr(self.cfg, "ot_margi_eps", 1e-9),
                    use_cross_view_consistency=bool(getattr(self.cfg, "use_cross_view_consistency", False)),
                    cross_view_alpha=float(getattr(self.cfg, "cross_view_alpha", 0.7)),
                    cross_view_pos_lambda=float(getattr(self.cfg, "cross_view_pos_lambda", 0.75)),
                    cross_view_phi_scale=float(getattr(self.cfg, "cross_view_phi_scale", 8.0)),
                    cross_view_phi_bias=float(getattr(self.cfg, "cross_view_phi_bias", 0.5)),
                    cross_view_norm_transitive=bool(getattr(self.cfg, "cross_view_norm_transitive", True)),
                    Uq_cells=Uq_chunk,
                    Ug_cells=Ug_chunk,
                    Gamma_q=Gq_chunk,
                    Gamma_g=Gg_chunk,
                    use_cross_view_row_consistency=bool(getattr(self.cfg, "use_cross_view_row_consistency", False)),
                    cross_view_row_weight=float(getattr(self.cfg, "cross_view_row_weight", 0.0)),
                    cross_view_row_pos_lambda=float(getattr(self.cfg, "cross_view_row_pos_lambda", 0.75)),
                    use_view_prototype_propagation=bool(getattr(self.cfg, "use_view_prototype_propagation", False)),
                    view_prototype_path=str(getattr(self.cfg, "view_prototype_path", "")),
                    view_propagation_lambda=float(getattr(self.cfg, "view_propagation_lambda", 0.15)),
                    view_prototype_temp=float(getattr(self.cfg, "view_prototype_temp", 10.0)),
                    use_view_prototype_span=bool(getattr(self.cfg, "use_view_prototype_span", False)),
                    view_prototype_span_lambda=float(getattr(self.cfg, "view_prototype_span_lambda", 1e-3)),
                    view_transition_self=float(getattr(self.cfg, "view_transition_self", 1.0)),
                    view_transition_neighbor1=float(getattr(self.cfg, "view_transition_neighbor1", 0.7)),
                    view_transition_neighbor2=float(getattr(self.cfg, "view_transition_neighbor2", 0.2)),
                    use_view_uncertainty_gate=bool(getattr(self.cfg, "use_view_uncertainty_gate", True)),
                )
                
                dist_chunk = 1.0 - S  # (chunk_q, chunk_g)
                
                # Merge avec le heap courant
                all_vals = torch.cat([best_vals, dist_chunk], dim=1)
                all_idxs = torch.cat([
                    best_idxs,
                    torch.arange(j0, j1, device=self.device).expand(dist_chunk.size(0), -1)
                ], dim=1)
                
                # Top-k
                best_vals, pos = torch.topk(all_vals, k, dim=1, largest=False, sorted=True)
                best_idxs = torch.gather(all_idxs, 1, pos)
                
                del Eg_chunk, Rg_chunk, Og_chunk, Ug_chunk, Gg_chunk, Zg_chunk, Orient_g_chunk, S, dist_chunk
            
            dist_topk[i0:i1] = best_vals.cpu()
            idx_topk[i0:i1] = best_idxs.cpu()
            if self.logger and (((i0 // chunk_q) + 1) % 10 == 0 or i1 == Nq):
                self.logger.info(f"  stripe exact: {i1}/{Nq} queries")
            del Uq_chunk, Gq_chunk, Orient_q_chunk

        return dist_topk, idx_topk

    def compute_stripe_topk_candidates(
        self,
        qE,
        qRel,
        qO,
        gE,
        gRel,
        gO,
        candidate_idx: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-k stripe sur une shortlist de candidats par query.
        candidate_idx: (Nq, Kc) indices gallery (CPU ou GPU).
        """
        from loss_optimized import set_to_set_similarity_candidates  # type: ignore

        if bool(getattr(self.cfg, "use_view_prototype_propagation", False)):
            raise NotImplementedError(
                "View-prototype propagation is implemented only for exact stripe OT scoring. "
                "Disable shortlist candidates and use exact/full stripe evaluation."
            )

        Nq = qE.size(0)
        if candidate_idx.dim() != 2 or candidate_idx.size(0) != Nq:
            raise ValueError(f"candidate_idx doit être (Nq, Kc), reçu {candidate_idx.shape}")
        if candidate_idx.size(1) == 0:
            raise ValueError("candidate_idx vide")

        candidate_idx = candidate_idx.long().cpu()
        Kc = candidate_idx.size(1)
        k = min(k, Kc)

        chunk_q = 64
        rel_are_normalized = False
        qRel_src = qRel
        gRel_src = gRel
        if qRel is not None and gRel is not None and qRel.dim() == 2 and gRel.dim() == 2:
            qRel_src = F.normalize(qRel.float(), dim=1)
            gRel_src = F.normalize(gRel.float(), dim=1)
            rel_are_normalized = True

        dist_topk = torch.empty((Nq, k), dtype=torch.float32)
        idx_topk = torch.empty((Nq, k), dtype=torch.long)

        if self.logger:
            self.logger.info(f"Computing stripe top-{k} on candidate pool K={Kc}...")

        C = qE.size(1)
        D = qE.size(2)
        gRel_dim2 = gRel_src is not None and gRel_src.dim() == 2
        rel_last_dim = gRel_src.size(-1) if gRel_src is not None else 0

        for i0 in range(0, Nq, chunk_q):
            i1 = min(Nq, i0 + chunk_q)
            B = i1 - i0

            Eq_chunk = qE[i0:i1].to(self.device)
            Rq_chunk = qRel_src[i0:i1].to(self.device) if qRel_src is not None else None
            Oq_chunk = qO[i0:i1].to(self.device)

            cand = candidate_idx[i0:i1]
            flat = cand.reshape(-1)

            Eg_chunk = gE.index_select(0, flat).view(B, Kc, C, D).to(self.device)
            if gRel_src is None:
                Rg_chunk = None
            elif gRel_dim2:
                Rg_chunk = gRel_src.index_select(0, flat).view(B, Kc, rel_last_dim).to(self.device)
            else:
                C_rel = gRel_src.size(1)
                S_rel = gRel_src.size(2)
                Rg_chunk = gRel_src.index_select(0, flat).view(B, Kc, C_rel, S_rel).to(self.device)
            Og_chunk = gO.index_select(0, flat).view(B, Kc, C).to(self.device)

            S = set_to_set_similarity_candidates(
                Eq_chunk, Rq_chunk, Oq_chunk,
                Eg_chunk, Rg_chunk, Og_chunk,
                self.cfg.sim_mode,
                self.cfg.alpha_mix,
                self.cfg.set_match_temp,
                self.cfg.use_wp_in_agg,
                self.cfg.use_zscore_if_C9,
                getattr(self.cfg, "zscore_kappa", 2.5),
                rel_are_normalized=rel_are_normalized
            )
            dist_chunk = 1.0 - S  # (B, Kc)

            best_vals, pos = torch.topk(dist_chunk, k, dim=1, largest=False, sorted=True)
            cand_dev = cand.to(self.device)
            best_idxs = torch.gather(cand_dev, 1, pos)

            dist_topk[i0:i1] = best_vals.cpu()
            idx_topk[i0:i1] = best_idxs.cpu()

            if self.logger and (((i0 // chunk_q) + 1) % 10 == 0 or i1 == Nq):
                self.logger.info(f"  stripe shortlist: {i1}/{Nq} queries")

            del Eg_chunk, Rg_chunk, Og_chunk, S, dist_chunk, best_vals, pos, cand_dev, best_idxs

        return dist_topk, idx_topk

    def _extract_eval_split(
        self,
        model,
        loader,
        desc: str,
        need_cross_view_cells: bool,
        use_complex: bool,
    ):
        start = time.time()
        if need_cross_view_cells:
            if use_complex:
                feat, stripes, mod_abs, rel, omega, cells, gamma, pids, camids = self.extract_features(
                    model, loader, desc, return_cells=True
                )
            else:
                feat, stripes, rel, omega, cells, gamma, pids, camids = self.extract_features(
                    model, loader, desc, return_cells=True
                )
                mod_abs = None
        else:
            if use_complex:
                feat, stripes, mod_abs, rel, omega, pids, camids = self.extract_features(
                    model, loader, desc
                )
            else:
                feat, stripes, rel, omega, pids, camids = self.extract_features(
                    model, loader, desc
                )
                mod_abs = None
            cells = gamma = None

        complex_z = self._last_extracted_Z if use_complex else None
        orientation = getattr(self, "_last_extracted_orient", None)
        desc_lower = str(desc).lower()
        if "query" in desc_lower:
            self._last_extracted_orient_q = orientation
        elif "gallery" in desc_lower:
            self._last_extracted_orient_g = orientation
        elapsed = time.time() - start
        if self.logger:
            self.logger.info(
                f"{desc.replace('Extracting ', '').capitalize()} extraction done in "
                f"{self._fmt_duration(elapsed)} | "
                f"N={feat.size(0)} D={feat.size(1) if feat.dim() > 1 else 0}"
            )
        return feat, stripes, mod_abs, rel, omega, cells, gamma, pids, camids, complex_z, elapsed

    def _maybe_offload_model_for_distance(self, model) -> bool:
        if self.device.type != "cuda":
            return False
        try:
            model.to("cpu")
            self._cleanup_cuda_cache()
            if self.logger:
                self.logger.info(
                    "Moved model to CPU after feature extraction to free CUDA memory "
                    "for distance computation."
                )
            return True
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"Could not move model to CPU before distance computation: {exc}"
                )
            return False

    def _restore_model_after_distance(self, model, should_restore: bool) -> None:
        if not should_restore:
            return
        try:
            model.to(self.device)
            self._cleanup_cuda_cache()
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"Could not restore model to {self.device} after evaluation: {exc}"
                )
    
    @torch.no_grad()
    def evaluate(self, model, q_loader, g_loader) -> Dict[str, float]:
        """
        Évaluation complète avec métriques globales et stripe.
        
        Returns:
            Dict avec mAP, Rank-k pour global et/ou stripe selon config
        """
        with Timer("Evaluation", self.logger):
            eval_start = time.time()
            skip_global_cfg = bool(getattr(self.cfg, "skip_global_eval", False))
            stripe_only_cfg = bool(getattr(self.cfg, "stripe_only_mode", False))
            skip_global_effective = skip_global_cfg or stripe_only_cfg
            use_cell_ot, use_stripe_ot = self._resolve_ot_modes()
            need_cross_view_cells = self._needs_cross_view_cells()
            if use_cell_ot:
                raise RuntimeError(
                    "Cell OT evaluation is not supported in this evaluator path. "
                    "Use use_stripe_ot_matching=True for full query/gallery evaluation."
                )
            if self.logger:
                self.logger.info(
                    "Eval setup: "
                    f"device={self.device} | "
                    f"skip_global_cfg={skip_global_cfg} | "
                    f"stripe_only_mode={stripe_only_cfg} | "
                    f"skip_global_effective={skip_global_effective} | "
                    f"skip_stripe={bool(getattr(self.cfg, 'skip_stripe_eval', False))} | "
                    f"force_full_dist_stripe={bool(getattr(self.cfg, 'force_full_dist_stripe', False))} | "
                    f"use_cell_ot={use_cell_ot} | "
                    f"use_stripe_ot={use_stripe_ot} | "
                    f"ot_eps={float(getattr(self.cfg, 'ot_epsilon', 0.1))} | "
                    f"ot_iters={int(getattr(self.cfg, 'ot_num_iters', 100))} | "
                    f"cross_view={bool(getattr(self.cfg, 'use_cross_view_consistency', False))} | "
                    f"cv_alpha={float(getattr(self.cfg, 'cross_view_alpha', 0.7)):.3f} | "
                    f"cv_pos_lambda={float(getattr(self.cfg, 'cross_view_pos_lambda', 0.75)):.3f} | "
                    f"cv_phi_scale={float(getattr(self.cfg, 'cross_view_phi_scale', 8.0)):.3f} | "
                    f"cv_phi_bias={float(getattr(self.cfg, 'cross_view_phi_bias', 0.5)):.3f} | "
                    f"cv_row={need_cross_view_cells} | "
                    f"cv_row_weight={float(getattr(self.cfg, 'cross_view_row_weight', 0.0)):.3f} | "
                    f"orientation_ot={bool(getattr(self.cfg, 'use_orientation_guided_ot', False))} | "
                    f"ori_cost_w={float(getattr(self.cfg, 'orientation_ot_cost_weight', 0.05)):.3f} | "
                    f"ori_mass_w={float(getattr(self.cfg, 'orientation_ot_mass_weight', 1.0)):.3f} | "
                    f"crg_lambda={float(getattr(self.cfg, 'crg_lambda', 0.5)):.3f}"
                )

            # Extraction
            use_complex = bool(getattr(self.cfg, "use_complex_hermitian_embedding", False))
            qf, qE, qModAbs, qRel, qO, qU, qGamma, q_pids, q_camids, self._last_extracted_Z_q, q_extract_time = (
                self._extract_eval_split(
                    model, q_loader, "Extracting query", need_cross_view_cells, use_complex
                )
            )
            gf, gE, gModAbs, gRel, gO, gU, gGamma, g_pids, g_camids, self._last_extracted_Z_g, g_extract_time = (
                self._extract_eval_split(
                    model, g_loader, "Extracting gallery", need_cross_view_cells, use_complex
                )
            )

            restore_model_to_device = self._maybe_offload_model_for_distance(model)

            try:
                # Logs de débogage optionnels sur les normes des features
                if getattr(self.cfg, "debug_eval_stats", False) and self.logger:
                    _logger = self.logger
                    def _log_norm(label: str, t: torch.Tensor):
                        if t is None:
                            return
                        flat = t.detach().float().view(t.size(0), -1)
                        norms = torch.linalg.norm(flat, dim=1)
                        _logger.info(
                            f"[DebugEval] {label} norms: "
                            f"min={norms.min():.4f} max={norms.max():.4f} "
                            f"mean={norms.mean():.4f} std={norms.std(unbiased=False):.4f}"
                        )

                    _log_norm("query_global", qf)
                    _log_norm("gallery_global", gf)
                    if qE is not None:
                        _log_norm("query_stripe", qE)
                        _log_norm("gallery_stripe", gE)

                # Conversion pour métriques
                q_pids_arr = np.array(q_pids)
                g_pids_arr = np.array(g_pids)
                q_camids_arr = np.array(q_camids)
                g_camids_arr = np.array(g_camids)

                metrics = {}
                stripe_only_mode = bool(getattr(self.cfg, "stripe_only_mode", False))
                concat_cosine_eval = bool(getattr(self.cfg, "concat_stripe_global_cosine_eval", False)) and qE is not None
                need_stripe_eval = not getattr(self.cfg, "skip_stripe_eval", False) and qE is not None
                stripe_candidate_pool = int(getattr(self.cfg, "stripe_candidate_pool", 0))
                if stripe_only_mode:
                    stripe_candidate_pool = 0
                if need_cross_view_cells and stripe_candidate_pool > 0:
                    if self.logger:
                        self.logger.warning(
                            "Row cross-view consistency requires exact stripe scoring; disabling candidate shortlist."
                        )
                    stripe_candidate_pool = 0
                skip_global_eval_effective = bool(getattr(self.cfg, "skip_global_eval", False)) or stripe_only_mode
                # concat-cosine mode overrides both global-only and stripe-OT eval paths
                if concat_cosine_eval:
                    skip_global_eval_effective = True
                    need_stripe_eval = False
                global_idx_for_stripe = None

                # Fallback auto sur CPU quand la recherche stripe exacte devient trop coûteuse.
                if need_stripe_eval and stripe_candidate_pool <= 0 and self.device.type == "cpu":
                    est_pairs = int(qE.size(0) * gE.size(0))
                    auto_threshold = max(0, int(getattr(self.cfg, "stripe_auto_candidate_threshold", 5_000_000)))
                    auto_pool = max(0, int(getattr(self.cfg, "stripe_auto_candidate_pool", 1000)))
                    if auto_pool > 0 and auto_threshold > 0 and est_pairs > auto_threshold:
                        stripe_candidate_pool = max(1, min(auto_pool, gE.size(0)))
                        if self.logger:
                            self.logger.warning(
                                f"Stripe exact search is too large on CPU (Nq*Ng={est_pairs:,}); "
                                f"auto-enabling shortlist with K={stripe_candidate_pool} candidates/query."
                            )

                # === évaluation globale ===
                if not skip_global_eval_effective:
                    if self.logger:
                        self.logger.info("Stage: global ranking")
                    global_stage_start = time.time()
                    if getattr(self.cfg, "force_full_dist", False):
                        save_path = self._resolve_full_dist_global_path()
                        dist_g_np = self._maybe_reuse_full_dist(
                            save_path, (qf.size(0), gf.size(0)), "global"
                        )
                        if dist_g_np is None:
                            if self.logger:
                                bytes_needed = qf.size(0) * gf.size(0) * np.dtype(np.float32).itemsize
                                self.logger.info(
                                    f"Computing full global distance matrix to disk: {save_path} "
                                    f"({bytes_needed / 1e9:.2f} GB)"
                                )
                            assert save_path is not None
                            dist_g_np = self.compute_global_full_dist_memmap(qf, gf, save_path)
                            if self.logger:
                                self.logger.info(f"Saved full global distance matrix to {save_path}")

                        m_global = compute_cmc_map(
                            dist_g_np, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam,
                            logger=self.logger,
                            progress_label="global-metrics",
                            progress_every=5000,
                        )
                    else:
                        k_global = max(1, min(self.cfg.eval_topk, gf.size(0)))
                        k_global_compute = k_global
                        if need_stripe_eval and stripe_candidate_pool > 0:
                            k_global_compute = max(k_global, min(stripe_candidate_pool, gf.size(0)))
                        dist_g, idx_g = cosine_topk_chunked(
                            qf, gf, self.device, k=k_global_compute
                        )
                        global_idx_for_stripe = idx_g
                        if k_global_compute > k_global:
                            dist_g_eval = dist_g[:, :k_global]
                            idx_g_eval = idx_g[:, :k_global]
                        else:
                            dist_g_eval = dist_g
                            idx_g_eval = idx_g
                        m_global = compute_cmc_map_topk(
                            dist_g_eval, idx_g_eval, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam
                        )
                    if self.logger:
                        self.logger.info(
                            f"Global ranking finished in {self._fmt_duration(time.time() - global_stage_start)}"
                        )

                    metrics.update({
                        "mAP": m_global.get("mAP", 0.0),
                        "Rank-1": m_global.get("Rank-1", 0.0),
                        "Rank-5": m_global.get("Rank-5", 0.0),
                        "Rank-10": m_global.get("Rank-10", 0.0),
                    })
                elif need_stripe_eval and stripe_candidate_pool > 0:
                    # Même si on saute la métrique globale, on peut l'utiliser comme shortlist.
                    k_global_compute = max(1, min(stripe_candidate_pool, gf.size(0)))
                    if self.logger:
                        self.logger.info(
                            f"Computing global shortlist for stripe evaluation (K={k_global_compute})..."
                        )
                    _, global_idx_for_stripe = cosine_topk_chunked(
                        qf, gf, self.device, k=k_global_compute
                    )

                # === évaluation stripe ===
                if need_stripe_eval:
                    if self.logger:
                        self.logger.info("Stage: stripe ranking")
                    stripe_stage_start = time.time()
                    if getattr(self.cfg, "force_full_dist_stripe", False):
                        save_path = self._resolve_full_dist_stripe_path()
                        if save_path is None:
                            save_path = Path(getattr(self.cfg, "out_dir", ".")) / "full_dist_stripe.npy"
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                        dist_s_np = self._maybe_reuse_full_dist(
                            save_path, (qE.size(0), gE.size(0)), "stripe"
                        )
                        if dist_s_np is None:
                            if self.logger:
                                bytes_needed = qE.size(0) * gE.size(0) * np.dtype(np.float32).itemsize
                                self.logger.info(
                                    f"Computing full stripe distance matrix to disk: {save_path} "
                                    f"({bytes_needed / 1e9:.2f} GB)"
                                )
                            dist_s_np = self.compute_stripe_full_dist_memmap(
                                qE, qModAbs, qRel, qO, gE, gModAbs, gRel, gO, save_path,
                                qU=qU, qGamma=qGamma, gU=gU, gGamma=gGamma,
                            )
                            if self.logger:
                                self.logger.info(f"Saved full stripe distance matrix to {save_path}")
                        m_stripe = compute_cmc_map(
                            dist_s_np, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam,
                            logger=self.logger,
                            progress_label="stripe-metrics",
                            progress_every=5000,
                        )
                    else:
                        k_stripe = getattr(self.cfg, "stripe_topk", 0) or self.cfg.eval_topk
                        k_stripe = max(1, min(k_stripe, gE.size(0)))
                        if stripe_candidate_pool > 0:
                            k_cand = max(k_stripe, min(stripe_candidate_pool, gE.size(0)))
                            if global_idx_for_stripe is None or global_idx_for_stripe.size(1) < k_cand:
                                if self.logger:
                                    self.logger.info(
                                        f"Refreshing global shortlist for stripe evaluation (K={k_cand})..."
                                    )
                                _, global_idx_for_stripe = cosine_topk_chunked(
                                    qf, gf, self.device, k=k_cand
                                )
                            cand_idx = global_idx_for_stripe[:, :k_cand]
                            dist_s, idx_s = self.compute_stripe_topk_candidates(
                                qE, qRel, qO, gE, gRel, gO, candidate_idx=cand_idx, k=k_stripe
                            )
                        else:
                            dist_s, idx_s = self.compute_stripe_topk(
                                qE, qRel, qO, gE, gRel, gO, k=k_stripe,
                                qModAbs=qModAbs, gModAbs=gModAbs,
                                qU=qU, qGamma=qGamma, gU=gU, gGamma=gGamma,
                            )
                        m_stripe = compute_cmc_map_topk(
                            dist_s, idx_s, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam
                        )
                    if self.logger:
                        self.logger.info(
                            f"Stripe ranking finished in {self._fmt_duration(time.time() - stripe_stage_start)}"
                        )

                    metrics.update({
                        "mAP_stripe": m_stripe.get("mAP", 0.0),
                        "Rank-1_stripe": m_stripe.get("Rank-1", 0.0),
                        "Rank-5_stripe": m_stripe.get("Rank-5", 0.0),
                        "Rank-10_stripe": m_stripe.get("Rank-10", 0.0),
                    })

                # === évaluation concat-cosinus ===
                if concat_cosine_eval:
                    if self.logger:
                        self.logger.info("Stage: concat-cosine ranking")
                    concat_stage_start = time.time()
                    # Construire le vecteur concatené [global_D | stripe_1_D | ... | stripe_C_D]
                    qf_dev = qf.to(self.device)
                    gf_dev = gf.to(self.device)
                    qE_dev = qE.to(self.device)  # (Nq, C, D)
                    gE_dev = gE.to(self.device)  # (Ng, C, D)
                    # Reshape stripes: (B, C, D) ↁE(B, C*D)
                    qv = F.normalize(torch.cat([qf_dev, qE_dev.reshape(qE_dev.size(0), -1)], dim=1), dim=1)
                    gv = F.normalize(torch.cat([gf_dev, gE_dev.reshape(gE_dev.size(0), -1)], dim=1), dim=1)
                    del qf_dev, gf_dev, qE_dev, gE_dev

                    if getattr(self.cfg, "force_full_dist_stripe", False):
                        save_path = self._resolve_full_dist_stripe_path()
                        if save_path is None:
                            save_path = Path(getattr(self.cfg, "out_dir", ".")) / "full_dist_concat_cosine.npy"
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            # rename to avoid confusion with OT stripe matrix
                            save_path = save_path.parent / "full_dist_concat_cosine.npy"
                        dist_c_np = self._maybe_reuse_full_dist(
                            save_path, (qv.size(0), gv.size(0)), "concat_cosine"
                        )
                        if dist_c_np is None:
                            if self.logger:
                                bytes_needed = qv.size(0) * gv.size(0) * np.dtype(np.float32).itemsize
                                self.logger.info(
                                    f"Computing full concat-cosine distance matrix: {save_path} "
                                    f"({bytes_needed / 1e9:.2f} GB)"
                                )
                            dist_c_np = self.compute_global_full_dist_memmap(qv, gv, save_path)
                            if self.logger:
                                self.logger.info(f"Saved concat-cosine distance matrix to {save_path}")
                        m_concat = compute_cmc_map(
                            dist_c_np, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam,
                            logger=self.logger,
                            progress_label="concat-metrics",
                            progress_every=5000,
                        )
                    else:
                        k_c = max(1, min(self.cfg.eval_topk, gv.size(0)))
                        dist_c, idx_c = cosine_topk_chunked(qv, gv, self.device, k=k_c)
                        m_concat = compute_cmc_map_topk(
                            dist_c, idx_c, q_pids_arr, g_pids_arr,
                            q_camids_arr, g_camids_arr,
                            remove_same_cam=self.cfg.remove_same_cam
                        )
                    del qv, gv
                    if self.logger:
                        self.logger.info(
                            f"Concat-cosine ranking finished in {self._fmt_duration(time.time() - concat_stage_start)}"
                        )
                    metrics.update({
                        "mAP": m_concat.get("mAP", 0.0),
                        "Rank-1": m_concat.get("Rank-1", 0.0),
                        "Rank-5": m_concat.get("Rank-5", 0.0),
                        "Rank-10": m_concat.get("Rank-10", 0.0),
                        "mAP_concat": m_concat.get("mAP", 0.0),
                        "Rank-1_concat": m_concat.get("Rank-1", 0.0),
                    })

                # Log résumé
                if self.logger:
                    msg = "Results:"
                    if "mAP_concat" in metrics:
                        msg += f" Concat-Cosine mAP={metrics['mAP_concat']:.2%} R1={metrics['Rank-1_concat']:.2%}"
                    elif "mAP" in metrics:
                        msg += f" Global mAP={metrics['mAP']:.2%} R1={metrics['Rank-1']:.2%}"
                    if "mAP_stripe" in metrics:
                        msg += f" | Stripe mAP={metrics['mAP_stripe']:.2%} R1={metrics['Rank-1_stripe']:.2%}"
                    self.logger.info(msg)
                    self.logger.info(
                        "Eval summary: "
                        f"total_time={self._fmt_duration(time.time() - eval_start)} | "
                        f"query_extract={self._fmt_duration(q_extract_time)} | "
                        f"gallery_extract={self._fmt_duration(g_extract_time)}"
                    )

                return metrics
            finally:
                self._restore_model_after_distance(model, restore_model_to_device)
    
    @torch.no_grad()
    def evaluate_checkpoint(self, checkpoint_path: str, q_loader, g_loader,
                           num_classes: Optional[int] = None) -> Dict[str, float]:
        """
        ?value directement depuis un checkpoint.
        """
        from utils_optimized import load_checkpoint  # type: ignore
        from model_optimized import DOCModel  # type: ignore
        
        # Chargement
        ckpt = load_checkpoint(checkpoint_path, self.device)
        
        # Reconstruction modèle
        cfg_dict = ckpt.get("config", {})
        if cfg_dict:
            from config_optimized import CFG  # type: ignore
            eval_cfg = CFG.from_dict(cfg_dict)
        else:
            eval_cfg = self.cfg

        # Keep model architecture from checkpoint config, but force runtime eval behavior
        # from the current evaluator cfg (eval_cfg.json / CLI intent).
        eval_override_keys = [
            "out_dir",
            "device",
            "skip_global_eval",
            "skip_stripe_eval",
            "stripe_only_mode",
            "concat_stripe_global_cosine_eval",
            "force_full_dist",
            "force_full_dist_stripe",
            "save_full_dist",
            "save_full_dist_stripe",
            "reuse_full_dist_if_exists",
            "eval_topk",
            "stripe_topk",
            "stripe_candidate_pool",
            "stripe_auto_candidate_pool",
            "stripe_auto_candidate_threshold",
            "remove_same_cam",
            "use_cell_ot_matching",
            "use_stripe_ot_matching",
            "use_uniform_ot_marginals",
            "ot_epsilon",
            "ot_num_iters",
            "ot_margi_eps",
            "sim_mode",
            "alpha_mix",
            "set_match_temp",
            "crg_lambda",
            "use_wp_in_agg",
            "use_zscore_if_C9",
            "zscore_kappa",
            "use_view_prototype_propagation",
            "view_prototype_path",
            "view_propagation_lambda",
            "view_prototype_temp",
            "use_view_prototype_span",
            "view_prototype_span_lambda",
            "view_transition_self",
            "view_transition_neighbor1",
            "view_transition_neighbor2",
            "use_view_uncertainty_gate",
        ]
        override_values = {
            key: getattr(self.cfg, key)
            for key in eval_override_keys
            if hasattr(self.cfg, key)
        }
        if override_values:
            try:
                eval_cfg = replace(eval_cfg, **override_values)
            except Exception:
                # Fallback for non-dataclass mutable config objects.
                for key, value in override_values.items():
                    setattr(eval_cfg, key, value)
        
        pid2label = ckpt.get("pid2label", {})
        n_classes = num_classes or len(pid2label)
        
        model = DOCModel(
            eval_cfg.backbone_name,
            n_classes,
            interstripe_transformer=bool(getattr(eval_cfg, "interstripe_transformer", True)),
            interstripe_num_stripes=int(getattr(eval_cfg, "C_stripes", 5)),
            interstripe_num_heads=int(getattr(eval_cfg, "interstripe_num_heads", 8)),
            interstripe_num_layers=int(getattr(eval_cfg, "interstripe_num_layers", 2)),
            interstripe_dropout=float(getattr(eval_cfg, "interstripe_dropout", 0.1)),
            interstripe_concat_global_local=bool(getattr(eval_cfg, "interstripe_concat_global_local", False)),
            interstripe_concat_dropout=float(getattr(eval_cfg, "interstripe_concat_dropout", 0.0)),
            gnn_use_horizontal=bool(getattr(eval_cfg, "gnn_use_horizontal", False)),
            gnn_residual_init=float(getattr(eval_cfg, "gnn_residual_init", 0.7)),
            use_residual_film_orientation=bool(getattr(eval_cfg, "use_residual_film_orientation", False)),
            film_hidden_dim=int(getattr(eval_cfg, "film_hidden_dim", 128)),
            film_zero_init=bool(getattr(eval_cfg, "film_zero_init", True)),
            film_formula=str(getattr(eval_cfg, "film_formula", "residual")),
            film_gamma_activation=str(getattr(eval_cfg, "film_gamma_activation", "tanh")),
            film_orientation_source=str(getattr(eval_cfg, "film_orientation_source", "view_prototype")),
            film_view_prototype_path=str(getattr(
                eval_cfg,
                "film_view_prototype_path",
                getattr(eval_cfg, "view_prototype_path", ""),
            )),
            film_prototype_temp=float(getattr(eval_cfg, "film_prototype_temp", 10.0)),
            film_use_view_prototype_span=bool(getattr(eval_cfg, "film_use_view_prototype_span", False)),
            film_view_prototype_span_lambda=float(getattr(eval_cfg, "film_view_prototype_span_lambda", 1e-3)),
        ).to(self.device)
        # Autorise les clés manquantes/supplémentaires (ex: stripe_classifier absent en preset sans stripes)
        model_state = DOCModel.adapt_legacy_single_film_state_dict(ckpt["model_state"])
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        # Evaluate with the merged runtime config, then restore original cfg.
        orig_cfg = self.cfg
        self.cfg = eval_cfg
        try:
            metrics = self.evaluate(model, q_loader, g_loader)
        finally:
            self.cfg = orig_cfg
        
        # Cleanup
        model.close()
        del model
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return metrics
