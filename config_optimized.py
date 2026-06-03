
import os
from dataclasses import dataclass, replace, asdict, field
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path


@dataclass(frozen=True)
class CFG:
    """
    Configuration immutable pour l'entraînement Re-ID.
    
    Toute modification doit utiliser replace(cfg, champ=nouvelle_valeur)
    """
    # --- Paths ---
    data_root: str = os.getenv("DATA_ROOT", "")
    train_list: str = "info/train_name.txt"
    test_list: str = "info/test_name.txt"
    train_tracks_mat: str = "info/tracks_train_info.mat"
    test_tracks_mat: str = "info/tracks_test_info.mat"
    query_idx_path: str = "info/query_IDX.mat"
    val_query_list: str = ""
    val_gallery_list: str = ""
    train_only: bool = False
    out_dir: str = "runs_docloss"
    fig_dirname: str = "figures"

    # --- Hardware ---
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True

    # --- Model ---
    backbone_name: str = "osnet_x1_0"
    pretrained: bool = True
    hook_layer: str = ""

    # --- Input ---
    height: int = 256
    width: int = 128
    P: int = 16
    K: int = 4
    batch_size: int = 64  # Sera auto-ajusté à P*K
    steps_per_epoch: int = 0

    # --- Optimization ---
    epochs: int = 200
    base_lr: float = 3.5e-4
    warmup_lr: float = 3.5e-5
    warmup_epochs: int = 10
    milestones: Tuple[int, ...] = (40, 70)
    gamma: float = 0.1
    weight_decay: float = 5e-4
    clip_grad: float = 1.0
    scheduler: str = "multistep"  # "multistep", "cosine", "none"
    stripe_lr_mult: float = 1.0

    # --- Early Stopping ---
    early_stop_patience: int = 0  # 0 = désactivé
    early_stop_min_delta: float = 0.001

    # --- DOC Settings ---
    C_stripes: int = 4
    R_rows: int = 4
    beta_energy: str = "l2"  # "l2" ou "amax"
    beta_thresh: float = 0.45
    beta_slope: float = 0.125
    beta_temp: float = 1.0
    beta_offset: float = 0.0
    omega_mode: str = "beta_rel"  # "ones", "beta", "rel", "beta_rel"
    omega_rel_mode: str = "mean"
    omega_rel_temp: float = 1.0
    use_zscore_if_C9: bool = False
    force_a_one: bool = False  # debug/eval: force a(r,c)=1 during stripe extraction
    force_extract_stripe_gnn: bool = True  # True = use GNN refinement, False = legacy extraction
    force_gnn_gamma_sigmoid: bool = True  # True = use sigmoid(W_gamma u') weights in GNN, False = use cell beta weights
    gnn_use_horizontal: bool = False  # False = vertical intra-stripe GNN; True = add optional left/right cross-stripe links
    gnn_residual_init: float = 0.7  # initial learned mix: alpha*u_updated + (1-alpha)*u_center
    interstripe_transformer: bool = False  # True = apply Transformer coordination across stripes after GNN aggregation
    interstripe_num_heads: int = 8
    interstripe_num_layers: int = 2
    interstripe_dropout: float = 0.1
    interstripe_concat_global_local: bool = False  # True = concat stripe descriptors with global descriptor after inter-stripe Transformer
    interstripe_concat_dropout: float = 0.0
    use_residual_film_orientation: bool = True  # True = orientation-conditioned residual FiLM on stripe descriptors before OT
    film_hidden_dim: int = 128
    film_zero_init: bool = True
    film_formula: str = "residual"  # "residual" or "beta_full"
    film_gamma_activation: str = "tanh"  # "tanh"/"sigmoid2" bound gamma to [-1,1]; "none" keeps legacy
    # "stripe_estimator" is true stripe-level conditioning: orientation_vec has shape (B, C, 2).
    # "external" is image-level by default: orientation_vec has shape (B, 2) and is broadcast.
    film_orientation_source: str = "stripe_estimator"  # "view_prototype", "external", or "stripe_estimator"
    film_view_prototype_path: str = "runs_manual_view_prototypes/mars_manual_view_prototypes_3000_identity.pt"
    film_prototype_temp: float = 10.0
    film_use_view_prototype_span: bool = False
    film_view_prototype_span_lambda: float = 1e-3
    use_complex_hermitian_embedding: bool = True  # True = use complex Hermitian matching at eval/inference (AUTO-ENABLED)
    use_complex_hermitian_Z: bool = True  # True = use explicit complex Z embeddings; False = use q_mod_abs/q_mod_rel path
    crg_lambda: float = 0.5  # Cross-Residual Gated Hermitian penalty strength
    film_external_orientation_csv: str = ""
    film_external_path_field: str = "image_path"
    film_external_angle_field: str = "pred_angle_deg"
    film_external_class_field: str = "pred_class_logits"
    film_external_confidence_field: str = "confidence"
    film_external_min_confidence: float = 0.0
    film_stripe_estimator_checkpoint: str = "third_party/mebow_official/models/model_hboe.pth"
    film_stripe_estimator_num_classes: int = 72
    film_stripe_estimator_height: int = 256
    film_stripe_estimator_width: int = 192
    film_stripe_estimator_sector_mode: str = "semantic10"  # "semantic10" or "semantic8_merge_sides"
    film_stripe_estimator_output_frame: str = "relative"  # "relative" or "absolute_visible"
    # --- Hierarchical relation vector (dimension D) ---
    use_rel_vec: bool = False
    relvec_lambda_inter_had: float = 1.0
    relvec_lambda_inter_diff: float = 0.5
    relvec_mu_intra_had: float = 1.0
    relvec_mu_intra_diff: float = 0.5
    relvec_eta_inter: float = 0.5
    relvec_eta_intra: float = 0.5
    relvec_eps: float = 1e-6
    relvec_normalize: bool = True
    relvec_detach_weights: bool = True
    use_relvec_global_fusion: bool = False
    relvec_global_alpha: float = 0.5

    # --- Matching ---
    sim_mode: str = "mix"  # "app", "rel", "mix"
    alpha_mix: float = 0.5
    set_match_temp: float = 0.45
    use_wp_in_agg: bool = True
    zscore_kappa: float = 2.5
    project_stripes: bool = True
    use_gamma_weights_for_matching: bool = False  # False keeps dynamic stripe OT marginals on Beta; True forces Gamma/GNN weights
    # OT flags:
    # - use_cell_ot_matching=True => OT groupe de cellules vers groupe de cellules
    # - use_stripe_ot_matching=True => OT groupe de stripes vers groupe de stripes
    # Legacy behavior compatibility: if use_stripe_ot_matching is None,
    # use_cell_ot_matching is interpreted as legacy stripe OT switch.
    use_cell_ot_matching: bool = False
    use_stripe_ot_matching: Optional[bool] = None
    ot_epsilon: float = 0.1   # Entropic regularization for Sinkhorn (0.1 adapte au stripe-to-stripe C=5)
    ot_num_iters: int = 100  # Number of Sinkhorn iterations
    ot_margi_eps: float = 1e-9  # Stability epsilon for marginal normalization
    skip_omega_rel_when_ot: bool = True  # Skip expensive Omega relation branch when OT drives matching
    use_uniform_ot_marginals: bool = False  # Use uniform marginals (1/C) instead of Omega for OT
    use_orientation_guided_ot: bool = True  # Add local stripe orientation prior to stripe OT
    orientation_ot_cost_weight: float = 0.05  # Weak additive cost: C_visual + w*C_ori
    orientation_ot_mass_weight: float = 1.0  # Stronger dynamic marginal gating: exp(-w*C_ori)
    # Cross-view consistency for stripe OT matching (Exp 4.1)
    use_cross_view_consistency: bool = False
    cross_view_alpha: float = 0.7            # blend between direct app similarity and transitive consistency
    cross_view_pos_lambda: float = 0.75      # positional prior decay over stripe index distance
    cross_view_phi_scale: float = 8.0        # sigmoid slope for feature-conditioned transition
    cross_view_phi_bias: float = 0.5         # sigmoid center in [0,1] similarity domain
    cross_view_norm_transitive: bool = True  # normalize second-order transitive matrix to [0,1]
    use_cross_view_row_consistency: bool = False  # also mix high-low row consistency into stripe OT cost
    cross_view_row_weight: float = 0.0       # final mix weight for row-level high-low consistency
    cross_view_row_pos_lambda: float = 0.75  # positional prior decay over row index distance
    use_view_prototype_propagation: bool = False  # inject prototype-conditioned view context after the transformer and before stripe OT
    view_prototype_path: str = "runs_manual_view_prototypes/mars_manual_view_prototypes_3000_identity.pt"
    view_propagation_lambda: float = 0.15
    view_prototype_temp: float = 10.0
    use_view_prototype_span: bool = False  # residualize each prototype against the span of all other view prototypes
    view_prototype_span_lambda: float = 1e-3  # ridge strength for span residualization; 0.0 is exact projection
    view_transition_self: float = 1.0
    view_transition_neighbor1: float = 0.7
    view_transition_neighbor2: float = 0.2
    use_view_uncertainty_gate: bool = False
    stripe_only_mode: bool = False  # True = disable global branch loss/eval and rank on stripe OT only
    concat_stripe_global_cosine_eval: bool = False  # True = eval by concat([global | stripe_1 | ... | stripe_C]) + cosine distance

    # --- Loss Toggles ---
    use_L_ID_g: bool = False
    use_L_tri_g: bool = False
    use_L_ID_s: bool = False
    use_L_tri_s_ot: bool = False
    use_L_attach: bool = False
    use_L_setNCE: bool = False
    use_L_div: bool = False
    use_L_local_match: bool = False
    use_L_aug: bool = False
    use_L_center: bool = False
    use_L_center_g: bool = False
    use_L_relNCE: bool = False
    use_L_ID_mod_abs: bool = True
    use_L_ID_mod_rel: bool = True
    use_L_tri_mod_abs: bool = True
    use_L_tri_mod_rel: bool = True
    use_L_tri_complex: bool = True

    # --- Weights ---
    w_ID_g: float = 1.0
    w_tri_g: float = 1.0
    w_ID_s: float = 0.5
    w_tri_s_ot: float = 1.0
    w_attach: float = 0.5
    w_setNCE: float = 1.0
    w_div: float = 0.1
    w_local_match: float = 0.1  # conservé mais non utilisé (local fusionné)
    w_aug: float = 0.2
    w_center: float = 0.002
    w_center_g: float = 0.005
    w_relNCE: float = 0.5
    w_ID_mod_abs: float = 0.3
    w_ID_mod_rel: float = 0.3
    w_tri_mod_abs: float = 0.4
    w_tri_mod_rel: float = 0.2
    w_tri_complex: float = 0.4

    # --- Three-phase FiLM training (optional) ---
    enable_three_phase_film_training: bool = False
    phase1_epochs: int = 0
    phase2_epochs: int = 0
    phase3_epochs: int = 0
    phase1_lr_scale: float = 1.0
    phase2_lr_scale: float = 0.5
    phase3_lr_scale: float = 0.1
    phase2_train_film_only: bool = True
    # --- Outputs ---
    save_full_dist: str = ""  # disabled: no global memmap (viz_suite will recompute on demand)
    save_full_dist_stripe: str = ""  # chemin .npy pour matrice complète stripe (optionnel)
    cache_dir: str = ""  # racine cache locale; defaut: <out_dir>/reid_matrices

    # --- Loss Parameters ---
    label_smoothing: float = 0.1
    triplet_margin: float = 0.3
    set_nce_temp: float = 0.15
    setnce_use_unified: bool = False
    attach_temp: float = 0.07
    aug_match_temp: float = 0.4
    local_match_margin: float = 0.15
    local_match_center_only: bool = False

    # --- Evaluation ---
    eval_every: int = 0
    eval_topk: int = 1000  # top-k obligatoire; 0 n'est plus autorisé
    eval_during_train: bool = False
    stripe_topk: int = 0
    stripe_candidate_pool: int = 0  # 0 => recherche stripe exacte sur toute la gallery
    stripe_auto_candidate_pool: int = 1000  # fallback auto CPU si exact trop lent
    stripe_auto_candidate_threshold: int = 5_000_000  # active le fallback si Nq*Ng > seuil
    remove_same_cam: bool = True
    skip_stripe_eval: bool = False
    skip_global_eval: bool = False
    keep_distractors: bool = False
    use_all_as_query: bool = False  # si True: toutes les images test deviennent queries+gallery
    use_query_track_frames: bool = True  # si True et tracks dispo: les 1 980 tracklets query deviennent toutes leurs frames
    strict_paths: bool = False
    allow_pid_fallback: bool = True
    debug_eval_stats: bool = False  # logs de stats de features pendant l'évaluation
    force_full_dist: bool = False  # autorise le top-k; full distance uniquement si demandé
    force_full_dist_stripe: bool = False  # allow full stripe distance matrix (otherwise top-k)
    reuse_full_dist_if_exists: bool = True  # reuse existing full .npy matrices instead of recomputing
    full_dist_global_run_override: str = ""  # si rempli, lit/écrit la matrice globale dans <cache_dir>/<override>/
    full_dist_stripe_run_override: str = ""  # si rempli, lit/écrit la matrice stripe dans <cache_dir>/<override>/
    always_full_distance_train: bool = True  # Allow forced full stripe distance eval when validation is enabled

    # --- System ---
    checkpoint_freq: int = 0  # Sauvegarde tous les N epochs (0 = seulement best/last)
    log_batch_freq: int = 50  # Log tous les N batches

    def __post_init__(self):
        # Validation des chemins
        if not self.data_root:
            object.__setattr__(self, 'data_root', os.getcwd())
        
        # Auto-adjust batch_size
        expected_bs = self.P * self.K
        if self.batch_size != expected_bs:
            object.__setattr__(self, 'batch_size', expected_bs)
        
        # Validate hyperparameters
        assert self.C_stripes >= 1, "C_stripes must be >= 1"
        assert self.R_rows >= 1, "R_rows must be >= 1"
        assert 0 <= self.alpha_mix <= 1, "alpha_mix must be in [0,1]"
        assert 0 <= self.relvec_global_alpha <= 1, "relvec_global_alpha must be in [0,1]"
        assert 0 <= self.cross_view_alpha <= 1, "cross_view_alpha must be in [0,1]"
        assert 0 <= self.cross_view_row_weight <= 1, "cross_view_row_weight must be in [0,1]"
        assert 0 <= self.view_propagation_lambda <= 1, "view_propagation_lambda must be in [0,1]"
        assert self.orientation_ot_cost_weight >= 0, "orientation_ot_cost_weight must be >= 0"
        assert self.orientation_ot_mass_weight >= 0, "orientation_ot_mass_weight must be >= 0"
        assert self.w_ID_mod_abs >= 0 and self.w_ID_mod_rel >= 0, "w_ID_mod_abs/w_ID_mod_rel must be >= 0"
        assert self.w_tri_mod_abs >= 0 and self.w_tri_mod_rel >= 0 and self.w_tri_complex >= 0, "triplet weights must be >= 0"
        assert self.phase1_epochs >= 0 and self.phase2_epochs >= 0 and self.phase3_epochs >= 0, "phase epochs must be >= 0"
        assert self.phase1_lr_scale > 0 and self.phase2_lr_scale > 0 and self.phase3_lr_scale > 0, "phase lr_scale must be > 0"
        assert self.stripe_lr_mult > 0, "stripe_lr_mult must be > 0"
        assert self.view_prototype_temp > 0, "view_prototype_temp must be > 0"
        assert self.view_prototype_span_lambda >= 0, "view_prototype_span_lambda must be >= 0"
        assert self.film_hidden_dim >= 1, "film_hidden_dim must be >= 1"
        assert self.film_formula in {"residual", "beta_full"}, (
            "film_formula must be 'residual' or 'beta_full'"
        )
        assert 0 < self.gnn_residual_init < 1, "gnn_residual_init must be in ]0,1["
        assert self.film_gamma_activation in {"none", "identity", "tanh", "sigmoid2"}, (
            "film_gamma_activation must be 'none', 'identity', 'tanh', or 'sigmoid2'"
        )
        assert self.film_prototype_temp > 0, "film_prototype_temp must be > 0"
        assert self.film_view_prototype_span_lambda >= 0, "film_view_prototype_span_lambda must be >= 0"
        assert isinstance(self.use_complex_hermitian_embedding, bool), "use_complex_hermitian_embedding must be a bool"
        assert self.crg_lambda >= 0, "crg_lambda must be >= 0"
        assert self.film_orientation_source in {"view_prototype", "external", "stripe_estimator"}, (
            "film_orientation_source must be 'view_prototype', 'external', or 'stripe_estimator'"
        )
        assert self.film_stripe_estimator_sector_mode in {"semantic10", "semantic8_merge_sides", "semantic8", "merged_sides"}, (
            "film_stripe_estimator_sector_mode must be 'semantic10' or 'semantic8_merge_sides'"
        )
        assert self.film_stripe_estimator_output_frame in {"relative", "absolute_visible"}, (
            "film_stripe_estimator_output_frame must be 'relative' or 'absolute_visible'"
        )
        assert self.view_transition_self >= 0, "view_transition_self must be >= 0"
        assert self.view_transition_neighbor1 >= 0, "view_transition_neighbor1 must be >= 0"
        assert self.view_transition_neighbor2 >= 0, "view_transition_neighbor2 must be >= 0"
        if self.use_stripe_ot_matching is not None:
            assert not (self.use_cell_ot_matching and self.use_stripe_ot_matching), (
                "OT mode ambiguous: enable either cell OT or stripe OT, not both"
            )
        
        # Windows keeps num_workers as configured. Entry points in
        # this folder are protected by if __name__ == "__main__" for spawn.

        # Evaluation safety guards: forbid full distance matrix by default
        if self.eval_topk <= 0:
            object.__setattr__(self, 'eval_topk', 1000)
        # stripe_topk hérite du global si non défini ou invalide
        if self.stripe_topk <= 0:
            object.__setattr__(self, 'stripe_topk', self.eval_topk)
        if self.stripe_candidate_pool < 0:
            object.__setattr__(self, 'stripe_candidate_pool', 0)
        if self.stripe_auto_candidate_pool < 0:
            object.__setattr__(self, 'stripe_auto_candidate_pool', 0)
        if self.stripe_auto_candidate_threshold < 0:
            object.__setattr__(self, 'stripe_auto_candidate_threshold', 0)
        # Default: keep top-k mode for global eval (avoids huge matrices).
        if not bool(self.always_full_distance_train):
            object.__setattr__(self, 'save_full_dist', "")
            object.__setattr__(self, 'force_full_dist', False)

        # Experimental plan: always evaluate stripes with full distance matrix during training.
        if bool(self.always_full_distance_train):
            object.__setattr__(self, 'eval_during_train', True)
            # Ne pas forcer eval_every=1 : l'eval finale (epoch == epochs) est
            # garantie par le trainer. eval_every=0 signifie "seulement en fin".
            object.__setattr__(self, 'force_full_dist', True)
            object.__setattr__(self, 'force_full_dist_stripe', True)
            object.__setattr__(self, 'stripe_candidate_pool', 0)
            object.__setattr__(self, 'stripe_auto_candidate_pool', 0)
            object.__setattr__(self, 'skip_stripe_eval', False)

    def to_dict(self) -> dict:
        """Convert to dict for JSON-safe serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "CFG":
        """Reconstruct a CFG from a dict."""
        rename_map = {
            "use_L_ID_raw": "use_L_ID_mod_abs",
            "use_L_ID_mod": "use_L_ID_mod_rel",
            "use_L_tri_raw": "use_L_tri_mod_abs",
            "use_L_tri_mod": "use_L_tri_mod_rel",
            "w_ID_raw": "w_ID_mod_abs",
            "w_ID_mod": "w_ID_mod_rel",
            "w_tri_raw": "w_tri_mod_abs",
            "w_tri_mod": "w_tri_mod_rel",
        }
        migrated = dict(d)
        for old_key, new_key in rename_map.items():
            if old_key in migrated and new_key not in migrated:
                migrated[new_key] = migrated[old_key]

        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in migrated.items() if k in valid_fields}
        return cls(**filtered)


def apply_preset(cfg: CFG, preset: str) -> CFG:
    """
    Apply a predefined training recipe.
    Returns a NEW config (immutable dataclass replace).
    """
    # Normalize and handle aliases ("R0/B0" -> "R0")
    p = (preset or "").upper().strip()
    if "/" in p:
        p = p.split("/")[0]
    
    # Base: reset tous les flags (A0 = ID_g + tri_g)
    base_overrides = {
        'use_L_ID_g': True,
        'use_L_tri_g': True,
        'use_L_ID_s': False,
        'use_L_tri_s_ot': False,
        'use_L_attach': False,
        'use_L_setNCE': False,
        'use_L_div': False,
        'use_L_local_match': False,
        'use_L_aug': False,
        'use_L_center': False,
        'use_L_center_g': False,
        'use_L_relNCE': False,
        'use_L_ID_mod_abs': False,
        'use_L_ID_mod_rel': False,
        'use_L_tri_mod_abs': False,
        'use_L_tri_mod_rel': False,
        'use_L_tri_complex': False,
        'use_rel_vec': False,
        'use_relvec_global_fusion': False,
    }
    
    presets = {
        # A0: ID_g + tri_g (baseline)
        # Pas de stripes nécessaires -> skip stripe eval pour éviter le top-k
        "A0": {**base_overrides, 'skip_stripe_eval': True, 'train_only': False, 'out_dir': 'runs_A0'},
        # A1: stripe classification baseline (no attach/setNCE/aug/rel)
        "A1": {**base_overrides,
             'use_L_attach': False,
               'use_L_ID_s': True,      # supervision locale des stripes
             'use_L_tri_s_ot': True,  # triplet structurel OT sur stripes
               'w_ID_s': 0.3,           # poids modéré
             'w_tri_s_ot': 0.5,
               'use_L_setNCE': False,
               'use_L_local_match': False,
               'use_L_aug': False,
               'use_L_relNCE': False,
               'skip_stripe_eval': False,
               'skip_global_eval': False,
               'train_only': False,
               'out_dir': 'runs_A1'},
        # A1.1: A1 sans L_attach (stripes supervisées mais pas forcées vers le global)
        "A1.1": {**base_overrides,
                 'use_L_attach': False,
                 'use_L_ID_s': True,
                 'use_L_tri_s_ot': True,
                 'w_ID_s': 0.3,
                 'w_tri_s_ot': 0.5,
                 'use_L_setNCE': False,
                 'use_L_local_match': False,
                 'use_L_aug': False,
                 'use_L_relNCE': False,
                 'skip_stripe_eval': False,
                 'skip_global_eval': False,
                 'train_only': False,
                 'out_dir': 'runs_A1_1'},
        # A2: Stripes autonomes (setNCE + attach), global éteint
        "A2": {**base_overrides,
               'use_L_ID_g': False,
               'use_L_tri_g': False,
               'use_L_attach': True,
               'use_L_setNCE': True,
               'setnce_use_unified': True,
               'use_L_local_match': False,
               'use_L_aug': False,
               'use_L_relNCE': False,
               'skip_global_eval': False,
               'skip_stripe_eval': False,
               'train_only': False,
               'out_dir': 'runs_A2'},
        # A2.1: Variante A2 avec attachement seul (sans setNCE, sans local_match)
        "A2.1": {**base_overrides,
                 'use_L_ID_g': False,
                 'use_L_tri_g': False,
                 'use_L_attach': True,
                 'use_L_setNCE': False,
                 'use_L_local_match': False,
                 'use_L_aug': False,
                 'use_L_relNCE': False,
                 'skip_global_eval': False,
                 'skip_stripe_eval': False,
                 'train_only': False,
                 'out_dir': 'runs_A2_1'},
        # A2.2: Local row/col + attach (meme base que A2.1, avec poids local explicite)
        "A2.2": {**base_overrides,
                 'use_L_ID_g': False,
                 'use_L_tri_g': False,
                 'use_L_attach': True,
                 'use_L_setNCE': False,
                 'use_L_local_match': True,
                 'local_match_center_only': False,
                 'w_local_match': 1.0,
                 'use_L_aug': False,
                 'use_L_relNCE': False,
                 'skip_global_eval': False,
                 'skip_stripe_eval': False,
                 'train_only': False,
                 'out_dir': 'runs_A2_2'},
        # A2.3: Set + attach, set contrastive pur (sans fusion locale)
        "A2.3": {**base_overrides,
                 'use_L_ID_g': False,
                 'use_L_tri_g': False,
                 'use_L_attach': True,
                 'use_L_setNCE': True,
                 'setnce_use_unified': False,
                 'use_L_local_match': False,
                 'use_L_aug': False,
                 'use_L_relNCE': False,
                 'skip_global_eval': False,
                 'skip_stripe_eval': False,
                 'train_only': False,
                 'out_dir': 'runs_A2_3'},
        # B0: Robustesse multi-vues (setNCE + aug, attach léger)
        "B0": {**base_overrides,
                'use_L_ID_g': False,
                'use_L_tri_g': False,
                'use_L_attach': True,
                'use_L_setNCE': True,
                'use_L_local_match': False,
                'use_L_aug': True,
                'use_L_relNCE': False,
                'w_attach': 0.3,
                'w_setNCE': 1.0,
                'w_aug': 0.5,
                'skip_global_eval': False,
                'skip_stripe_eval': False,
                'train_only': False,
                'out_dir': 'runs_B0'},
        # EXP5_FiLM_Hermitian: GNN + inter-stripe Transformer + FiLM + Hermitian complex losses.
        "EXP5_FILM_HERMITIAN": {**base_overrides,
                'use_L_ID_g': False,
                'use_L_tri_g': False,
                'interstripe_transformer': True,
                'use_L_ID_mod_abs': True,
                'use_L_ID_mod_rel': True,
                'use_L_tri_mod_abs': True,
                'use_L_tri_mod_rel': True,
                'use_L_tri_complex': True,
                'use_L_aug': False,
                'use_L_attach': False,
                'use_L_setNCE': False,
                'use_L_local_match': False,
                'use_L_relNCE': False,
                'w_ID_mod_abs': 0.3,
                'w_ID_mod_rel': 0.3,
                'w_tri_mod_abs': 0.4,
                'w_tri_mod_rel': 0.2,
                'w_tri_complex': 0.4,
                'enable_three_phase_film_training': True,
                'phase1_epochs': 100,
                'phase2_epochs': 50,
                'phase3_epochs': 50,
                'phase1_lr_scale': 1.0,
                'phase2_lr_scale': 0.5,
                'phase3_lr_scale': 0.1,
                'use_cell_ot_matching': False,
                'use_stripe_ot_matching': True,
                'use_uniform_ot_marginals': False,
                'use_gamma_weights_for_matching': False,
                'skip_omega_rel_when_ot': True,
                'use_orientation_guided_ot': False,
                'orientation_ot_cost_weight': 0.0,
                'orientation_ot_mass_weight': 0.0,
                'skip_global_eval': False,
                'skip_stripe_eval': False,
                'train_only': False,
                'out_dir': 'runs_EXP5_FiLM_Hermitian_C4'},
        # Backward-compatible alias for older commands.
        "B2": {**base_overrides,
                'use_L_ID_g': False,
                'use_L_tri_g': False,
                'interstripe_transformer': True,
                'use_L_ID_mod_abs': True,
                'use_L_ID_mod_rel': True,
                'use_L_tri_mod_abs': True,
                'use_L_tri_mod_rel': True,
                'use_L_tri_complex': True,
                'use_L_aug': False,
                'use_L_attach': False,
                'use_L_setNCE': False,
                'use_L_local_match': False,
                'use_L_relNCE': False,
                'w_ID_mod_abs': 0.3,
                'w_ID_mod_rel': 0.3,
                'w_tri_mod_abs': 0.4,
                'w_tri_mod_rel': 0.2,
                'w_tri_complex': 0.4,
                'enable_three_phase_film_training': True,
                'phase1_epochs': 100,
                'phase2_epochs': 50,
                'phase3_epochs': 50,
                'phase1_lr_scale': 1.0,
                'phase2_lr_scale': 0.5,
                'phase3_lr_scale': 0.1,
                'use_cell_ot_matching': False,
                'use_stripe_ot_matching': True,
                'use_uniform_ot_marginals': False,
                'use_gamma_weights_for_matching': False,
                'skip_omega_rel_when_ot': True,
                'use_orientation_guided_ot': False,
                'orientation_ot_cost_weight': 0.0,
                'orientation_ot_mass_weight': 0.0,
                'skip_global_eval': False,
                'skip_stripe_eval': False,
                'train_only': False,
                'out_dir': 'runs_EXP5_FiLM_Hermitian_C4'},
        # R0: Relations pures (relNCE + aug, apparence coupée)
        "R0": {**base_overrides,
                'use_L_ID_g': False,
                'use_L_tri_g': False,
                'use_L_attach': False,
                'use_L_setNCE': False,
                'use_L_local_match': False,
                'use_L_aug': True,
                'use_L_relNCE': True,
                'use_rel_vec': True,
                'w_relNCE': 1.0,
                'w_aug': 0.3,
                'skip_global_eval': False,
                'skip_stripe_eval': False,
                'out_dir': 'runs_R0'},
    }
    
    if p not in presets:
        raise ValueError(f"Unknown preset: '{preset}'. Available: {list(presets.keys())}")
    
    return replace(cfg, **presets[p])


def apply_matching_code(cfg: CFG, code: str, base_C: Optional[int] = None) -> CFG:
    """
    Modifie la config pour l'inférence (ablation study).
    """
    n = (code or "").upper().strip()
    base_C = base_C or cfg.C_stripes
    
    # Defaults
    overrides = {
        'skip_stripe_eval': False,
        'skip_global_eval': False,
        'sim_mode': "mix",
        'omega_mode': "beta_rel",
        'beta_energy': "l2",
        'force_a_one': False,
        'use_wp_in_agg': True,
        'use_zscore_if_C9': False,
        'C_stripes': base_C,
    }
    
    codes = {
        # Evaluation modes (nouvelle nomenclature)
        "G0": {'skip_stripe_eval': True, 'skip_global_eval': False},
        "O0": {'skip_stripe_eval': False, 'skip_global_eval': True,
               'omega_mode': "ones", 'force_a_one': True},
        "O1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones"},
        "O2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'sim_mode': "app"},
        "O3": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'sim_mode': "rel"},
        "O1.1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'alpha_mix': 0.8},
        "P1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta"},
        "P2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "rel"},
        "P3": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel"},
        "P3.1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel",
                 'use_wp_in_agg': False},
        "P3.2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel",
                 'beta_energy': "amax"},
         # P4 : évaluation relations pures (sim_mode=rel)
        "P4": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
                 'omega_mode': "ones", 'alpha_mix': 0.0},
        # P5 : rel_vec inter-only (r_inter)
        "P5": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
               'use_rel_vec': True, 'omega_mode': "beta_rel", 'alpha_mix': 0.0,
               'relvec_eta_inter': 1.0, 'relvec_eta_intra': 0.0,
               'use_relvec_global_fusion': False},
        # P6 : rel_vec intra-only (r_intra)
        "P6": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
               'use_rel_vec': True, 'omega_mode': "beta_rel", 'alpha_mix': 0.0,
               'relvec_eta_inter': 0.0, 'relvec_eta_intra': 1.0,
               'use_relvec_global_fusion': False},
    }
    
    if n not in codes:
        raise ValueError(f"Unknown matching code: '{code}'")
    
    overrides.update(codes[n])
    
    # Basic consistency guards
    if overrides.get('use_zscore_if_C9', False) and overrides.get('C_stripes', base_C) != 9:
        overrides['C_stripes'] = 9  # z-score requires exactly 9 stripes
    
    return replace(cfg, **overrides)


PRESET_TO_RECOMMENDED_EVAL_CODES: Dict[str, List[str]] = {
    "A0": ["G0"],
    "A1": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "A1.1": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "A2": ["O2", "O3", "O1.1", "P1", "P2", "P3"],
    "A2.1": ["O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "A2.2": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3"],
    "A2.3": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3"],
    "B0": ["O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P3.1", "P3.2", "P4", "P5", "P6"],
    "EXP5_FILM_HERMITIAN": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "EXP5_FiLM_Hermitian": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "B2": ["G0", "O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P5", "P6"],
    "R0": ["O0", "O1", "O2", "O3", "O1.1", "P1", "P2", "P3", "P3.1", "P3.2", "P4", "P5", "P6"],
}


# Codes d'évaluation (nomenclature)
EVAL_CODES: Dict[str, Dict] = {
    # Evaluation modes (nouvelle nomenclature)
    "G0": {'skip_stripe_eval': True, 'skip_global_eval': False},
    "O0": {'skip_stripe_eval': False, 'skip_global_eval': True,
           'omega_mode': "ones", 'force_a_one': True},
    "O1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones"},
    "O2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'sim_mode': "app"},
    "O3": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'sim_mode': "rel"},
    "O1.1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "ones", 'alpha_mix': 0.8},
    "P1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta"},
    "P2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "rel"},
    "P3": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel"},
    "P3.1": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel",
             'use_wp_in_agg': False},
    "P3.2": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel",
             'beta_energy': "amax"},
    "Z9": {'skip_stripe_eval': False, 'skip_global_eval': True, 'omega_mode': "beta_rel",
           'C_stripes': 9, 'use_zscore_if_C9': True},
    # P4 : évaluation relations pures (sim_mode=rel)
    "P4": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
           'omega_mode': "ones", 'alpha_mix': 0.0},
    # P5 : rel_vec inter-only (r_inter)
    "P5": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
           'use_rel_vec': True, 'omega_mode': "beta_rel", 'alpha_mix': 0.0,
           'relvec_eta_inter': 1.0, 'relvec_eta_intra': 0.0,
           'use_relvec_global_fusion': False},
    # P6 : rel_vec intra-only (r_intra)
    "P6": {'skip_stripe_eval': False, 'skip_global_eval': True, 'sim_mode': "rel",
           'use_rel_vec': True, 'omega_mode': "beta_rel", 'alpha_mix': 0.0,
           'relvec_eta_inter': 0.0, 'relvec_eta_intra': 1.0,
           'use_relvec_global_fusion': False},
}
