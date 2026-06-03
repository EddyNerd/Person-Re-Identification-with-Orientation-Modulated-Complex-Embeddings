import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from typing import Dict, Optional, Tuple, List, Any
import time
import shutil
from dataclasses import replace

from model_optimized import DOCModel  # type: ignore
from loss_optimized import DOCLoss, ViewPrototypeRepository, resolve_ot_modes  # type: ignore
import utils_optimized as _utils  # type: ignore


save_checkpoint = getattr(_utils, "save_checkpoint")
load_checkpoint = getattr(_utils, "load_checkpoint")
EarlyStopping = getattr(_utils, "EarlyStopping")
Timer = getattr(_utils, "Timer")
ExternalOrientationProvider = getattr(_utils, "ExternalOrientationProvider")
StripeOrientationEstimator = getattr(_utils, "StripeOrientationEstimator")


class Trainer:
    """
    Entraîneur pour le modèle DOC avec toutes les fonctionnalités modernes.
    """
    
    def __init__(self, cfg, device_manager, logger):
        self.cfg = cfg
        self.dm = device_manager
        self.device = device_manager.device
        self.logger = logger
        
        # État
        self.model: Any = None
        self.optimizer: Any = None
        self.scaler: Any = None
        self.scheduler: Any = None
        self.loss_fn: Any = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.orientation_provider: Optional[Any] = None
        self.stripe_orientation_estimator: Optional[Any] = None
        self.view_prototype_repository = ViewPrototypeRepository()
        self.runtime_cfg = cfg
        self._manual_phase_lrs = False
        self._active_phase_name = "single"
        
    def setup(self, num_classes: int):
        """Initialise le modèle, optimiseur et scheduler."""
        self.logger.info(f"Setting up model with {num_classes} classes")
        
        # Modèle
        self.model = DOCModel(
            self.cfg.backbone_name,
            num_classes,
            pretrained=self.cfg.pretrained,
            hook_layer=self.cfg.hook_layer,
            interstripe_transformer=bool(getattr(self.cfg, "interstripe_transformer", True)),
            interstripe_num_stripes=int(getattr(self.cfg, "C_stripes", 5)),
            interstripe_num_heads=int(getattr(self.cfg, "interstripe_num_heads", 8)),
            interstripe_num_layers=int(getattr(self.cfg, "interstripe_num_layers", 2)),
            interstripe_dropout=float(getattr(self.cfg, "interstripe_dropout", 0.1)),
            interstripe_concat_global_local=bool(getattr(self.cfg, "interstripe_concat_global_local", False)),
            interstripe_concat_dropout=float(getattr(self.cfg, "interstripe_concat_dropout", 0.0)),
            gnn_use_horizontal=bool(getattr(self.cfg, "gnn_use_horizontal", False)),
            gnn_residual_init=float(getattr(self.cfg, "gnn_residual_init", 0.7)),
            use_residual_film_orientation=bool(getattr(self.cfg, "use_residual_film_orientation", False)),
            film_hidden_dim=int(getattr(self.cfg, "film_hidden_dim", 128)),
            film_zero_init=bool(getattr(self.cfg, "film_zero_init", True)),
            film_formula=str(getattr(self.cfg, "film_formula", "residual")),
            film_gamma_activation=str(getattr(self.cfg, "film_gamma_activation", "tanh")),
            film_orientation_source=str(getattr(self.cfg, "film_orientation_source", "view_prototype")),
            film_view_prototype_path=str(getattr(
                self.cfg,
                "film_view_prototype_path",
                getattr(self.cfg, "view_prototype_path", ""),
            )),
            film_prototype_temp=float(getattr(self.cfg, "film_prototype_temp", 10.0)),
            film_use_view_prototype_span=bool(getattr(self.cfg, "film_use_view_prototype_span", False)),
            film_view_prototype_span_lambda=float(getattr(self.cfg, "film_view_prototype_span_lambda", 1e-3)),
        ).to(self.device)

        # Dimension des features pour center loss (si dispo)
        center_feat_dim = None
        classifier = getattr(self.model.backbone, "classifier", None)
        if classifier is not None and hasattr(classifier, "in_features"):
            center_feat_dim = classifier.in_features
        center_g_dim = None
        # global_feat dimension = last conv? use output from forward; fallback to classifier in_features if exists
        if hasattr(self.model.backbone, "classifier") and hasattr(self.model.backbone.classifier, "in_features"):
            center_g_dim = self.model.backbone.classifier.in_features
        
        # Fonction de perte
        self.loss_fn = DOCLoss(
            self.cfg,
            num_classes,
            center_feat_dim,
            center_g_dim,
            view_prototype_repository=self.view_prototype_repository,
        ).to(self.device)

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
            self.logger.info(
                f"Loaded stripe orientation estimator from {_se.checkpoint_path}"
            )
        
        # Optimiseur : backbone + stripe_classifier + center losses éventuelles
        # Paramètres backbone (sans le head stripe pour éviter la double inclusion)
        base_params = [p for n, p in self.model.named_parameters() if not n.startswith("stripe_classifier")]
        param_groups = [
            {"params": base_params, "lr": self.cfg.base_lr, "weight_decay": self.cfg.weight_decay},
        ]
        # Head de stripes (dédié)  Es'il existe
        if getattr(self.model, "stripe_classifier", None) is not None:
            param_groups.append({
                "params": self.model.stripe_classifier.parameters(),
                "lr": self.cfg.base_lr * getattr(self.cfg, "stripe_lr_mult", 1.0),
                "weight_decay": self.cfg.weight_decay,
            })
        # Center losses éventuelles
        if self.loss_fn.center_loss is not None:
            param_groups.append({"params": self.loss_fn.center_loss.parameters(), "lr": self.cfg.base_lr, "weight_decay": 0.0})
        if self.loss_fn.center_loss_g is not None:
            param_groups.append({"params": self.loss_fn.center_loss_g.parameters(), "lr": self.cfg.base_lr, "weight_decay": 0.0})

        self.optimizer = torch.optim.AdamW(param_groups)
        
        # Scheduler
        if self.cfg.scheduler == "multistep":
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.cfg.milestones,
                gamma=self.cfg.gamma
            )
        elif self.cfg.scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.cfg.epochs // 3,
                T_mult=2
            )
        
        # AMP Scaler
        pin_memory, use_amp = self.dm.get_io_flags(
            self.cfg.pin_memory, self.cfg.use_amp
        )
        self.scaler = GradScaler(enabled=use_amp)
        
        # Early stopping
        if self.cfg.early_stop_patience > 0:
            self.early_stopper = EarlyStopping(
                patience=self.cfg.early_stop_patience,
                min_delta=self.cfg.early_stop_min_delta,
                mode="max"
            )
        else:
            self.early_stopper = None

        self.runtime_cfg = self.cfg
        
        return self

    def _set_optimizer_lr(self, base_lr: float):
        stripe_mult = float(getattr(self.runtime_cfg, "stripe_lr_mult", 1.0))
        for idx, group in enumerate(self.optimizer.param_groups):
            if idx == 0:
                group["lr"] = float(base_lr)
            elif idx == 1:
                group["lr"] = float(base_lr) * stripe_mult
            else:
                group["lr"] = float(base_lr)

    def _set_trainable_for_phase(self, phase_name: str):
        if phase_name != "mod_rel_support" or not bool(getattr(self.runtime_cfg, "phase2_train_film_only", True)):
            for p in self.model.parameters():
                p.requires_grad = True
            return

        for name, p in self.model.named_parameters():
            trainable = (
                ("residual_film_orientation" in name)
                or ("stripe_classifier" in name)
            )
            p.requires_grad = trainable

    def _build_phase_schedule(self) -> List[Tuple[int, int, str, float]]:
        total_epochs = int(getattr(self.cfg, "epochs", 1))
        p1 = int(getattr(self.cfg, "phase1_epochs", 0))
        p2 = int(getattr(self.cfg, "phase2_epochs", 0))
        p3 = int(getattr(self.cfg, "phase3_epochs", 0))

        if p1 <= 0 and p2 <= 0 and p3 <= 0:
            p1 = max(1, int(round(0.6 * total_epochs)))
            p2 = max(1, int(round(0.25 * total_epochs)))
            p3 = max(1, total_epochs - p1 - p2)

        if p1 + p2 + p3 != total_epochs:
            p3 = max(1, total_epochs - p1 - p2)

        plan = []
        start = 1
        if p1 > 0:
            plan.append((start, start + p1 - 1, "mod_abs_anchor", float(getattr(self.cfg, "phase1_lr_scale", 1.0))))
            start += p1
        if p2 > 0 and start <= total_epochs:
            end = min(total_epochs, start + p2 - 1)
            plan.append((start, end, "mod_rel_support", float(getattr(self.cfg, "phase2_lr_scale", 0.5))))
            start = end + 1
        if start <= total_epochs:
            plan.append((start, total_epochs, "joint_calibration", float(getattr(self.cfg, "phase3_lr_scale", 0.1))))
        return plan

    def _phase_cfg(self, phase_name: str):
        if phase_name == "mod_abs_anchor":
            return replace(
                self.cfg,
                use_residual_film_orientation=False,
                use_L_ID_g=False,
                use_L_tri_g=False,
                use_L_ID_mod_abs=True,
                use_L_ID_mod_rel=False,
                use_L_tri_mod_abs=True,
                use_L_tri_mod_rel=False,
                use_L_tri_complex=False,
                use_L_ID_s=False,
                use_L_tri_s_ot=False,
                w_ID_mod_abs=0.6,
                w_tri_mod_abs=0.4,
            )
        if phase_name == "mod_rel_support":
            return replace(
                self.cfg,
                use_residual_film_orientation=True,
                use_L_ID_g=False,
                use_L_tri_g=False,
                use_L_ID_mod_abs=False,
                use_L_ID_mod_rel=True,
                use_L_tri_mod_abs=False,
                use_L_tri_mod_rel=True,
                use_L_tri_complex=False,
                w_ID_mod_rel=0.6,
                w_tri_mod_rel=0.4,
                use_L_ID_s=False,
                use_L_tri_s_ot=False,
            )
        return replace(
            self.cfg,
            use_residual_film_orientation=True,
            use_L_ID_g=False,
            use_L_tri_g=False,
            use_L_ID_mod_abs=True,
            use_L_ID_mod_rel=True,
            use_L_tri_mod_abs=False,
            use_L_tri_mod_rel=False,
            use_L_tri_complex=True,
            w_ID_mod_abs=0.3,
            w_ID_mod_rel=0.3,
            w_tri_complex=0.4,
            use_L_ID_s=False,
            use_L_tri_s_ot=False,
        )

    def _resolve_phase(self, epoch: int, schedule: List[Tuple[int, int, str, float]]) -> Tuple[str, float]:
        for start, end, phase_name, lr_scale in schedule:
            if start <= epoch <= end:
                return phase_name, lr_scale
        return "joint_calibration", 0.1

    def _apply_phase(self, phase_name: str, lr_scale: float):
        self._active_phase_name = phase_name
        self.runtime_cfg = self._phase_cfg(phase_name)
        self.loss_fn.cfg = self.runtime_cfg
        self._set_trainable_for_phase(phase_name)
        self._set_optimizer_lr(float(getattr(self.cfg, "base_lr", 3.5e-4)) * float(lr_scale))
        if phase_name == "mod_abs_anchor":
            phase_desc = "mod_abs anchor (FiLM OFF): 0.6 * L_ID_mod_abs + 0.4 * L_tri_mod_abs"
        elif phase_name == "mod_rel_support":
            phase_desc = "mod_rel support (FiLM abs+rel active): 0.6 * L_ID_mod_rel + 0.4 * L_tri_mod_rel"
        else:
            phase_desc = "joint calibration (low LR): 0.3 * L_ID_mod_abs + 0.3 * L_ID_mod_rel + 0.4 * L_tri_complex"
        self.logger.info(
            f"[Phase] {phase_name} | lr_scale={lr_scale:.3f} | base_lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        self.logger.info(f"[Phase] details: {phase_desc}")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _gpu_mem_str(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "gpu_mem=n/a"
        alloc = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9
        return f"gpu_mem alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"
    
    def train_epoch(self, train_loader) -> Tuple[float, Dict]:
        """
        Entraîne une époque complète.
        
        Returns:
            avg_loss: Perte moyenne
            metrics: Dict avec les moyennes des métriques
        """
        self.model.train()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        epoch_loss = 0.0
        all_losses = []
        
        pin_memory, use_amp = self.dm.get_io_flags(
            self.cfg.pin_memory, self.cfg.use_amp
        )
        
        pbar_prefix = f"Epoch {self.current_epoch:3d}"
        total_batches = len(train_loader)
        epoch_start = time.time()
        seen_items = 0
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            
            # Dépacking batch
            if self.cfg.use_L_aug and len(batch) == 5:
                imgs, imgs2, pids, _, paths = batch
                imgs = imgs.to(self.device, non_blocking=pin_memory)
                imgs2 = imgs2.to(self.device, non_blocking=pin_memory)
                has_aug = True
            else:
                imgs, pids, _, paths = batch
                imgs = imgs.to(self.device, non_blocking=pin_memory)
                imgs2 = None
                has_aug = False
            
            pids = pids.to(self.device, non_blocking=pin_memory)
            
            # Forward
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                outputs = self._forward_batch(imgs, pids, imgs2 if has_aug else None, paths)
                loss, loss_dict = self.loss_fn(outputs, pids)

            if not torch.isfinite(loss):
                self.logger.error(
                    f"Non-finite loss detected at epoch={self.current_epoch}, batch={batch_idx}: {loss.item()}"
                )
                raise RuntimeError("Training aborted due to non-finite loss")
            
            # Backward
            if use_amp:
                self.scaler.scale(loss).backward()
                
                if self.cfg.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.clip_grad
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.clip_grad
                    )
                
                self.optimizer.step()
            
            # Logging
            loss_val = loss.item()
            epoch_loss += loss_val
            all_losses.append(loss_val)
            self.global_step += 1
            seen_items += int(imgs.size(0))
            
            step_time = time.time() - step_start
            
            # Log périodique
            if batch_idx % self.cfg.log_batch_freq == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_train_step(
                    self.current_epoch, batch_idx, loss_val, lr, step_time
                )

                done_batches = batch_idx + 1
                progress_pct = 100.0 * done_batches / max(1, total_batches)
                elapsed = max(time.time() - epoch_start, 1e-6)
                batches_per_sec = done_batches / elapsed
                imgs_per_sec = seen_items / elapsed
                eta_sec = (total_batches - done_batches) / max(batches_per_sec, 1e-6)
                
                # Log détaillé des pertes
                loss_details = " | ".join([
                    f"{k}={v:.4f}" for k, v in loss_dict.items()
                    if not k.endswith("_w") and k != "total"
                ])  # Affiche toutes les composantes (inclut L_center si présente)
                self.logger.info(
                    f"  {pbar_prefix} Batch {done_batches:4d}/{total_batches} ({progress_pct:.2f}%) | "
                    f"Loss={loss_val:.4f} | LR={lr:.2e} | speed={imgs_per_sec:.1f} img/s | "
                    f"elapsed={self._fmt_duration(elapsed)} | eta={self._fmt_duration(eta_sec)} | "
                    f"{self._gpu_mem_str()}"
                )
                self.logger.info(f"  {pbar_prefix} Loss details | {loss_details}")
        
        # Stats époque
        avg_loss = epoch_loss / len(train_loader)
        
        # Mise à jour scheduler
        if (not self._manual_phase_lrs) and self.scheduler is not None:
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(self.current_epoch)
            else:
                self.scheduler.step()
        
        metrics = {
            "avg_loss": avg_loss,
            "min_loss": min(all_losses),
            "max_loss": max(all_losses),
            "epoch_time": time.time() - epoch_start,
            "samples_seen": seen_items,
        }
        
        return avg_loss, metrics
    
    def _forward_batch(self, imgs: torch.Tensor, pids: torch.Tensor,
                          imgs2: Optional[torch.Tensor] = None,
                          paths: Optional[List[str]] = None) -> Dict:
        """
        Forward pass pour un batch avec extraction de tous les composants.
        """
        # Forward principal
        logits, global_feat, feat_map = self.model.forward_global(imgs)
        
        outputs = {
            "logits": logits,
            "global_feat": global_feat,
            "global_feat_backbone": global_feat,
            "feat_map": feat_map,
            # Expose backbone classifier so L_ID_s can run
            "classifier": getattr(self.model.backbone, "classifier", None),
            "stripe_classifier": getattr(self.model, "stripe_classifier", None),
        }
        
        # Extraction stripes si nécessaire
        cfg = self.runtime_cfg

        need_rel_vec = bool(getattr(cfg, "use_rel_vec", False))
        if cfg.use_L_relNCE or cfg.use_L_setNCE or cfg.use_L_aug:
            need_rel_vec = True
        if bool(getattr(cfg, "use_relvec_global_fusion", False)):
            need_rel_vec = True

        use_cell_ot, _use_stripe_ot = resolve_ot_modes(
            bool(getattr(cfg, "use_cell_ot_matching", False)),
            getattr(cfg, "use_stripe_ot_matching", None),
        )

        orientation_vec = None
        if self.stripe_orientation_estimator is not None:
            orientation_vec = self.stripe_orientation_estimator.predict_batch(
                imgs,
                num_stripes=int(getattr(cfg, "C_stripes", 5)),
                output_dtype=feat_map.dtype,
            )
        elif self.orientation_provider is not None and paths is not None:
            orientation_vec = self.orientation_provider.get_batch(
                list(paths),
                device=imgs.device,
                dtype=feat_map.dtype,
            )

        need_cell_ot = use_cell_ot and (
            cfg.use_L_tri_s_ot or cfg.use_L_setNCE or cfg.use_L_aug
        )

        need_cells_for_omega = str(getattr(cfg, "omega_mode", "beta_rel")).lower() in {"rel", "beta_rel"}
        need_row_cross_view = (
            _use_stripe_ot
            and bool(getattr(cfg, "use_cross_view_consistency", False))
            and bool(getattr(cfg, "use_cross_view_row_consistency", False))
            and float(getattr(cfg, "cross_view_row_weight", 0.0)) > 0.0
        )

        use_complex_losses = bool(
            getattr(cfg, "use_L_ID_mod_abs", False)
            or getattr(cfg, "use_L_ID_mod_rel", False)
            or getattr(cfg, "use_L_tri_mod_abs", False)
            or getattr(cfg, "use_L_tri_mod_rel", False)
            or getattr(cfg, "use_L_tri_complex", False)
        )
        return_complex = use_complex_losses or bool(getattr(cfg, "use_complex_hermitian_embedding", False))

        use_stripes = (
            cfg.use_L_ID_s or cfg.use_L_tri_s_ot or cfg.use_L_setNCE or
            cfg.use_L_attach or cfg.use_L_div or
            cfg.use_L_aug or cfg.use_L_relNCE or need_rel_vec or use_complex_losses
        )
        
        if use_stripes:
            Uhat = Gamma = None
            Ehat_abs = Ehat_rel = None
            if need_rel_vec or need_cells_for_omega or need_cell_ot or need_row_cross_view:
                if return_complex:
                    Ehat, Beta, Uhat, Gamma, Ehat_abs, Ehat_rel = self.model.extract_stripes_adaptive(
                        feat_map,
                        cfg.C_stripes,
                        cfg.R_rows,
                        cfg,
                        orientation_vec=orientation_vec,
                        return_cells=True,
                        return_complex=True,
                    )
                else:
                    Ehat, Beta, Uhat, Gamma = self.model.extract_stripes_adaptive(
                        feat_map,
                        cfg.C_stripes,
                        cfg.R_rows,
                        cfg,
                        orientation_vec=orientation_vec,
                        return_cells=True,
                    )
            else:
                if return_complex:
                    Ehat, Beta, Ehat_abs, Ehat_rel = self.model.extract_stripes_adaptive(
                        feat_map,
                        cfg.C_stripes,
                        cfg.R_rows,
                        cfg,
                        orientation_vec=orientation_vec,
                        return_complex=True,
                    )
                else:
                    Ehat, Beta = self.model.extract_stripes_adaptive(
                        feat_map,
                        cfg.C_stripes,
                        cfg.R_rows,
                        cfg,
                        orientation_vec=orientation_vec,
                    )
            Omega = self.model.compute_omega(
                Ehat, Beta, cfg.omega_mode, Uhat=Uhat, Gamma=Gamma, cfg=cfg
            )
            
            outputs.update({
                "Ehat": Ehat,
                "Beta": Beta,
                "Omega": Omega,
                "Beta_ng": Beta.detach(),
                "Omega_ng": Omega.detach(),
            })
            if orientation_vec is not None:
                outputs["Orient"] = orientation_vec
            if (need_cell_ot or need_row_cross_view) and Uhat is not None and Gamma is not None:
                outputs.update({
                    "Uhat": Uhat,
                    "Gamma": Gamma,
                })
            if return_complex:
                if Ehat_abs is None or Ehat_rel is None:
                    raise RuntimeError("Complex stripe outputs are missing for the first view")
                outputs.update({
                    "Ehat_abs": F.normalize(Ehat_abs.float(), dim=-1),
                    "Ehat_rel": F.normalize(Ehat_rel.float(), dim=-1),
                    "Ehat_mod_abs": F.normalize(Ehat_abs.float(), dim=-1),
                    "Ehat_mod_rel": F.normalize(Ehat_rel.float(), dim=-1),
                })
            if need_rel_vec:
                rel_vec = self.model.compute_hierarchical_relation_vector(Ehat, Beta, Uhat, Gamma, cfg)
                outputs.update({
                    "Uhat": Uhat,
                    "Gamma": Gamma,
                    "rel_vec": rel_vec,
                })
                if bool(getattr(cfg, "use_relvec_global_fusion", False)):
                    alpha = float(getattr(cfg, "relvec_global_alpha", 0.5))
                    fused = self.model.fuse_global_with_relvec(
                        outputs["global_feat_backbone"].float(), rel_vec.float(), alpha=alpha, normalize_out=True
                    )
                    outputs["global_feat"] = fused
                    outputs["global_feat_fused"] = fused
        
        # Deuxième vue pour augmentation
        if imgs2 is not None:
            logits2, global_feat2, feat_map2 = self.model.forward_global(imgs2)
            orientation_vec2 = orientation_vec
            if self.stripe_orientation_estimator is not None:
                orientation_vec2 = self.stripe_orientation_estimator.predict_batch(
                    imgs2,
                    num_stripes=int(getattr(cfg, "C_stripes", 5)),
                    output_dtype=feat_map2.dtype,
                )
            outputs.update({
                "logits2": logits2,
                "global_feat2": global_feat2,
                "global_feat2_backbone": global_feat2,
            })
            
            if use_stripes:
                Uhat2 = Gamma2 = None
                Ehat2_abs = Ehat2_rel = None
                if need_rel_vec or need_cells_for_omega or need_cell_ot or need_row_cross_view:
                    if return_complex:
                        Ehat2, Beta2, Uhat2, Gamma2, Ehat2_abs, Ehat2_rel = self.model.extract_stripes_adaptive(
                            feat_map2,
                            cfg.C_stripes,
                            cfg.R_rows,
                            cfg,
                            orientation_vec=orientation_vec2,
                            return_cells=True,
                            return_complex=True,
                        )
                    else:
                        Ehat2, Beta2, Uhat2, Gamma2 = self.model.extract_stripes_adaptive(
                            feat_map2,
                            cfg.C_stripes,
                            cfg.R_rows,
                            cfg,
                            orientation_vec=orientation_vec2,
                            return_cells=True,
                        )
                else:
                    if return_complex:
                        Ehat2, Beta2, Ehat2_abs, Ehat2_rel = self.model.extract_stripes_adaptive(
                            feat_map2,
                            cfg.C_stripes,
                            cfg.R_rows,
                            cfg,
                            orientation_vec=orientation_vec2,
                            return_complex=True,
                        )
                    else:
                        Ehat2, Beta2 = self.model.extract_stripes_adaptive(
                            feat_map2,
                            cfg.C_stripes,
                            cfg.R_rows,
                            cfg,
                            orientation_vec=orientation_vec2,
                        )
                Omega2 = self.model.compute_omega(
                    Ehat2, Beta2, cfg.omega_mode, Uhat=Uhat2, Gamma=Gamma2, cfg=cfg
                )
                
                outputs.update({
                    "Ehat2": Ehat2,
                    "Beta2": Beta2,
                    "Omega2": Omega2,
                    "Omega2_ng": Omega2.detach(),
                })
                if orientation_vec2 is not None:
                    outputs["Orient2"] = orientation_vec2
                if (need_cell_ot or need_row_cross_view) and Uhat2 is not None and Gamma2 is not None:
                    outputs.update({
                        "Uhat2": Uhat2,
                        "Gamma2": Gamma2,
                    })
                if return_complex:
                    if Ehat2_abs is None or Ehat2_rel is None:
                        raise RuntimeError("Complex stripe outputs are missing for the second view")
                    outputs.update({
                        "Ehat_abs2": F.normalize(Ehat2_abs.float(), dim=-1),
                        "Ehat_rel2": F.normalize(Ehat2_rel.float(), dim=-1),
                        "Ehat_mod_abs2": F.normalize(Ehat2_abs.float(), dim=-1),
                        "Ehat_mod_rel2": F.normalize(Ehat2_rel.float(), dim=-1),
                    })
                if need_rel_vec:
                    rel_vec2 = self.model.compute_hierarchical_relation_vector(Ehat2, Beta2, Uhat2, Gamma2, cfg)
                    outputs.update({
                        "Uhat2": Uhat2,
                        "Gamma2": Gamma2,
                        "rel_vec2": rel_vec2,
                    })
                    if bool(getattr(cfg, "use_relvec_global_fusion", False)):
                        alpha = float(getattr(cfg, "relvec_global_alpha", 0.5))
                        fused2 = self.model.fuse_global_with_relvec(
                            outputs["global_feat2_backbone"].float(), rel_vec2.float(), alpha=alpha, normalize_out=True
                        )
                        outputs["global_feat2"] = fused2
                        outputs["global_feat2_fused"] = fused2
        
        return outputs
    
    def save(self, path: str, is_best: bool = False, metrics: Optional[Dict] = None):
        """Sauvegarde le checkpoint."""
        payload = dict(
            path=path,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.scaler.state_dict() if self.scaler else None,
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            epoch=self.current_epoch,
            metrics=metrics,
            cfg_dict=self.cfg.to_dict(),
            pid2label=getattr(self, "pid2label", {}) or {},
            best_metric=self.best_metric,
        )
        save_checkpoint(**payload)

        # Sauvegarde miroir dans runs_docloss pour compatibilité / fallback
        from pathlib import Path
        p = Path(path)
        if "runs_docloss" not in p.parts:
            mirror_dir = Path("runs_docloss")
            mirror_dir.mkdir(parents=True, exist_ok=True)
            mirror_path = mirror_dir / p.name
            payload["path"] = str(mirror_path)
            save_checkpoint(**payload)
        
        if is_best:
            self.logger.info(f"  Saved BEST model to {path}")
        else:
            self.logger.info(f"  Saved checkpoint to {path}")
    
    def load(self, path: str) -> Dict:
        """Charge depuis un checkpoint."""
        ckpt = load_checkpoint(path, self.device)
        
        model_state = DOCModel.adapt_legacy_single_film_state_dict(ckpt["model_state"])
        self.model.load_state_dict(model_state, strict=False)

        if "optimizer_state" in ckpt and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if "scaler_state" in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state"])

        if "scheduler_state" in ckpt and self.scheduler:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                msg = f"Could not load scheduler state: {e}"
                if self.logger and hasattr(self.logger, "warning"):
                    self.logger.warning(msg)
                else:
                    print(msg)

        self.current_epoch = ckpt.get("epoch", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
        
        return ckpt
    
    def close(self):
        """Nettoyage."""
        if self.model:
            self.model.close()
            del self.model
            self.model = None
        
        if self.optimizer:
            # Libère les états (ex: Adam moments) pour éviter fuite GPU/CPU
            self.optimizer.state.clear()
        
        if self.device.type == "cuda":
            # S'assurer que toutes les ops sont terminées avant de libérer le cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def train(self, train_loader, val_loader=None, evaluator=None) -> Dict:
        """
        Boucle d'entraînement complète.
        
        Returns:
            history: Dict avec les métriques d'entraînement
        """
        self.logger.info(f"Starting training for {self.cfg.epochs} epochs")
        self.logger.info(f"Config: {self.cfg}")
        use_cell_ot, use_stripe_ot = resolve_ot_modes(
            bool(getattr(self.cfg, "use_cell_ot_matching", False)),
            getattr(self.cfg, "use_stripe_ot_matching", None),
        )
        uniform_ot = bool(getattr(self.cfg, "use_uniform_ot_marginals", False))
        if use_cell_ot:
            ot_marginal_source = "cell_gamma"
        elif use_stripe_ot and uniform_ot:
            ot_marginal_source = "uniform"
        elif use_stripe_ot and bool(getattr(self.cfg, "use_gamma_weights_for_matching", False)):
            ot_marginal_source = "gnn_gamma_mean"
        elif use_stripe_ot:
            ot_marginal_source = "beta"
        else:
            ot_marginal_source = "n/a"
        self.logger.info(
            "Train setup: "
            f"device={self.device} | use_amp={bool(getattr(self.cfg, 'use_amp', False))} | "
            f"batch(PxK)={getattr(self.cfg, 'P', '?')}x{getattr(self.cfg, 'K', '?')} | "
            f"steps/epoch={getattr(self.cfg, 'steps_per_epoch', '?')} | "
            f"scheduler={getattr(self.cfg, 'scheduler', 'n/a')} | base_lr={getattr(self.cfg, 'base_lr', 0.0)}"
        )
        self.logger.info(
            "Matching setup: "
            f"sim_mode={getattr(self.cfg, 'sim_mode', 'mix')} | "
            f"alpha_mix={float(getattr(self.cfg, 'alpha_mix', 0.5)):.3f} | "
            f"use_cell_ot={use_cell_ot} | use_stripe_ot={use_stripe_ot} | "
            f"ot_eps={float(getattr(self.cfg, 'ot_epsilon', 0.1)):.4f} | "
            f"ot_iters={int(getattr(self.cfg, 'ot_num_iters', 100))} | "
            f"uniform_ot_marginals={uniform_ot} | "
            f"ot_marginal_source={ot_marginal_source}"
        )
        self.logger.info(
            "Cross-view setup: "
            f"enabled={bool(getattr(self.cfg, 'use_cross_view_consistency', False))} | "
            f"alpha={float(getattr(self.cfg, 'cross_view_alpha', 0.7)):.3f} | "
            f"pos_lambda={float(getattr(self.cfg, 'cross_view_pos_lambda', 0.75)):.3f} | "
            f"phi_scale={float(getattr(self.cfg, 'cross_view_phi_scale', 8.0)):.3f} | "
            f"phi_bias={float(getattr(self.cfg, 'cross_view_phi_bias', 0.5)):.3f} | "
            f"norm_transitive={bool(getattr(self.cfg, 'cross_view_norm_transitive', True))} | "
            f"row={bool(getattr(self.cfg, 'use_cross_view_row_consistency', False))} | "
            f"row_weight={float(getattr(self.cfg, 'cross_view_row_weight', 0.0)):.3f} | "
            f"row_pos_lambda={float(getattr(self.cfg, 'cross_view_row_pos_lambda', 0.75)):.3f}"
        )
        
        history = {
            "train_loss": [],
            "val_metrics": [],
            "best_epoch": 0,
            "best_metric": 0.0,
            "phase_schedule": [],
        }

        phase_training = bool(getattr(self.cfg, "enable_three_phase_film_training", False))
        phase_schedule = self._build_phase_schedule() if phase_training else []
        if phase_training:
            self._manual_phase_lrs = True
            history["phase_schedule"] = [
                {
                    "start_epoch": start,
                    "end_epoch": end,
                    "phase": phase_name,
                    "lr_scale": lr_scale,
                }
                for (start, end, phase_name, lr_scale) in phase_schedule
            ]
            schedule_txt = ", ".join(
                [f"{name}[{start}-{end}]x{scale:.2f}" for (start, end, name, scale) in phase_schedule]
            )
            self.logger.info(f"Three-phase training enabled: {schedule_txt}")
        else:
            self._manual_phase_lrs = False
        
        best_path = None
        eval_ran = False
        eval_count = 0
        val_metrics = None
        
        try:
            for epoch in range(1, self.cfg.epochs + 1):
                self.current_epoch = epoch
                if phase_training:
                    phase_name, lr_scale = self._resolve_phase(epoch, phase_schedule)
                    if phase_name != self._active_phase_name or epoch == 1:
                        self._apply_phase(phase_name, lr_scale)
                else:
                    self.runtime_cfg = self.cfg
                    self.loss_fn.cfg = self.cfg
                epoch_start = time.time()
                
                # Entraînement
                avg_loss, train_metrics = self.train_epoch(train_loader)
                epoch_time = time.time() - epoch_start
                
                # Log époque
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_epoch(epoch, avg_loss, lr, epoch_time, None)
                self.logger.info(
                    f"Epoch {epoch:3d} summary | loss(avg/min/max)={avg_loss:.4f}/{train_metrics['min_loss']:.4f}/{train_metrics['max_loss']:.4f} | "
                    f"samples={train_metrics['samples_seen']} | epoch_time={self._fmt_duration(train_metrics['epoch_time'])}"
                )
                history["train_loss"].append(avg_loss)
                
                # Évaluation
                val_metrics = None
                if val_loader is not None and evaluator is not None:
                    should_eval = (
                        epoch == self.cfg.epochs or
                        (self.cfg.eval_during_train and self.cfg.eval_every > 0 and epoch % self.cfg.eval_every == 0)
                    )
                    
                    if should_eval:
                        eval_start = time.time()
                        val_metrics = evaluator.evaluate(self.model, val_loader[0], val_loader[1])
                        eval_time = time.time() - eval_start
                        
                        self.logger.log_eval(epoch, val_metrics, eval_time)
                        self.logger.info(
                            f"Validation summary | eval_time={self._fmt_duration(eval_time)} | "
                            f"mAP={val_metrics.get('mAP', 0.0):.2%} | "
                            f"mAP_stripe={val_metrics.get('mAP_stripe', 0.0):.2%}"
                        )
                        history["val_metrics"].append(val_metrics)
                        eval_count += 1
                        
                        # Early stopping check
                        current_metric = val_metrics.get("mAP", val_metrics.get("mAP_stripe", 0))
                        
                        if self.early_stopper:
                            is_best = self.early_stopper(current_metric)
                            if self.early_stopper.early_stop:
                                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                                break
                        else:
                            is_best = current_metric > self.best_metric
                        
                        if is_best:
                            self.best_metric = current_metric
                            history["best_epoch"] = epoch
                            history["best_metric"] = current_metric
                            
                            best_path = f"{self.cfg.out_dir}/best_model.pth"
                            self.save(best_path, is_best=True, metrics=val_metrics)
                        
                        eval_ran = True
                
                # Sauvegarde périodique
                if self.cfg.checkpoint_freq > 0 and epoch % self.cfg.checkpoint_freq == 0:
                    periodic_path = f"{self.cfg.out_dir}/checkpoint_epoch_{epoch}.pth"
                    self.save(periodic_path, metrics=val_metrics)
            
            # Sauvegarde finale
            last_path = f"{self.cfg.out_dir}/last_model.pth"
            self.save(last_path, metrics=val_metrics)

            # Si aucun meilleur modèle n'a été identifié (pas d'éval), créer un best par défaut
            if best_path is None:
                best_path = f"{self.cfg.out_dir}/best_model.pth"
                self.save(best_path, is_best=True, metrics=val_metrics)

            # Évaluation finale si aucune évaluation n'a été réalisée pendant l'entraînement
            if not eval_ran and val_loader is not None and evaluator is not None:
                self.logger.info("Final evaluation after training (no eval ran during epochs)")
                val_metrics = evaluator.evaluate(self.model, val_loader[0], val_loader[1])
                history["val_metrics"].append(val_metrics)
                current_metric = val_metrics.get("mAP", val_metrics.get("mAP_stripe", 0))
                self.best_metric = current_metric
                history["best_epoch"] = self.current_epoch
                history["best_metric"] = current_metric
                best_path = f"{self.cfg.out_dir}/best_model.pth"
                self.save(best_path, is_best=True, metrics=val_metrics)
                eval_count += 1

            # Si aucune évaluation ou une seule (ex: uniquement finale), aligner best sur last
            if eval_count <= 1 and best_path:
                try:
                    shutil.copyfile(last_path, best_path)
                    self.logger.info("Best checkpoint aligned to last (<=1 eval run).")
                except Exception as e:
                    self.logger.warning(f"Could not align best to last: {e}")
            
            self.logger.info(f"Training completed. Best epoch: {history['best_epoch']} "
                           f"with mAP={history['best_metric']:.2%}")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Sauvegarde d'urgence
            interrupt_path = f"{self.cfg.out_dir}/interrupted_epoch_{self.current_epoch}.pth"
            self.save(interrupt_path)
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            self.close()
        
        return history
