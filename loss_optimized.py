import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING


def resolve_ot_modes(use_cell_ot_matching: bool = False,
                     use_stripe_ot_matching: Optional[bool] = None) -> Tuple[bool, bool]:
    """
    Resolve OT flags with backward compatibility.

    Returns:
        (use_cell_ot, use_stripe_ot)
    """
    use_cell = bool(use_cell_ot_matching)
    if use_stripe_ot_matching is None:
        # Legacy behavior: old `use_cell_ot_matching` toggled stripe OT.
        return False, use_cell
    return use_cell, bool(use_stripe_ot_matching)


VIEW_PROTOTYPE_ORDER = [
    "front",
    "front-left",
    "left side front",
    "left side back",
    "back-left",
    "back",
    "back-right",
    "right side back",
    "right side front",
    "front-right",
]


class ViewPrototypeRepository:
    """Load and cache view prototype bundles outside the matching math."""

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def resolve_path(prototype_path: str) -> Path:
        path = Path(prototype_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def load(
        self,
        prototype_path: str,
        device: torch.device,
        dtype: torch.dtype,
        use_span: bool = False,
        span_lambda: float = 1e-3,
    ) -> torch.Tensor:
        resolved = self.resolve_path(prototype_path)
        span_lam = max(0.0, float(span_lambda))
        cache_key = f"{resolved}|span={int(bool(use_span))}|lambda={span_lam:.8g}"
        if cache_key not in self._cache:
            if not resolved.exists():
                raise FileNotFoundError(f"View prototype bundle not found: {resolved}")
            bundle = torch.load(str(resolved), map_location="cpu")
            if not isinstance(bundle, dict):
                raise ValueError(f"Unexpected prototype bundle format in {resolved}")
            if "prototypes" not in bundle:
                raise ValueError(f"Prototype bundle missing required key 'prototypes' in {resolved}")
            prototypes = bundle["prototypes"]
            view_names = bundle.get("view_names", bundle.get("class_names"))
            if not isinstance(view_names, list):
                raise ValueError(f"Invalid view_names/class_names in prototype bundle: {resolved}")
            if not torch.is_tensor(prototypes):
                prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
            prototypes = prototypes.float()
            if prototypes.dim() != 2:
                raise ValueError(f"Prototype tensor must be 2D, got {tuple(prototypes.shape)}")
            if len(view_names) != prototypes.size(0):
                raise ValueError(
                    f"Prototype name count ({len(view_names)}) does not match tensor rows "
                    f"({prototypes.size(0)}) in {resolved}"
                )
            if len(view_names) == len(VIEW_PROTOTYPE_ORDER):
                canonical_aliases = {
                    "face": "front",
                    "front": "front",
                }
                name_to_idx = {}
                for idx, raw_name in enumerate(view_names):
                    view_name = str(raw_name)
                    canonical_name = canonical_aliases.get(view_name, view_name)
                    if canonical_name not in name_to_idx:
                        name_to_idx[canonical_name] = idx
                missing = [name for name in VIEW_PROTOTYPE_ORDER if name not in name_to_idx]
                if missing:
                    raise ValueError(
                        f"Prototype bundle {resolved} is missing required view names: {missing}"
                    )
                ordered = torch.stack([prototypes[name_to_idx[name]] for name in VIEW_PROTOTYPE_ORDER], dim=0)
            else:
                # Multi-view bundles are consumed in their declared angular order.
                ordered = prototypes
            ordered = F.normalize(ordered, dim=1)
            if use_span:
                ordered = _span_residualize_view_prototypes(ordered, span_lam)
            self._cache[cache_key] = ordered.cpu().contiguous()
        return self._cache[cache_key].to(device=device, dtype=dtype)

    def clear(self) -> None:
        self._cache.clear()


_DEFAULT_VIEW_PROTOTYPE_REPOSITORY = ViewPrototypeRepository()


def set_view_prototype_repository(repository: Optional[ViewPrototypeRepository]) -> None:
    """Install a repository for tests or distributed launchers that preload prototypes."""
    global _DEFAULT_VIEW_PROTOTYPE_REPOSITORY
    _DEFAULT_VIEW_PROTOTYPE_REPOSITORY = repository or ViewPrototypeRepository()


def _resolve_view_prototype_path(prototype_path: str) -> Path:
    return _DEFAULT_VIEW_PROTOTYPE_REPOSITORY.resolve_path(prototype_path)


def _load_view_prototypes(
    prototype_path: str,
    device: torch.device,
    dtype: torch.dtype,
    use_span: bool = False,
    span_lambda: float = 1e-3,
) -> torch.Tensor:
    return _DEFAULT_VIEW_PROTOTYPE_REPOSITORY.load(
        prototype_path,
        device=device,
        dtype=dtype,
        use_span=use_span,
        span_lambda=span_lambda,
    )


def _span_residualize_view_prototypes(
    prototypes: torch.Tensor,
    span_lambda: float,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Replace each prototype by its residual after projection on the span of all
    other prototypes. This is a symmetric one-vs-rest alternative to ordered
    Gram-Schmidt, with optional ridge regularization.
    """
    P = F.normalize(prototypes.float(), dim=1)
    num_views, _ = P.shape
    residuals = []

    for idx in range(num_views):
        others = torch.cat([P[:idx], P[idx + 1:]], dim=0)
        if others.numel() == 0:
            residuals.append(P[idx])
            continue

        gram = torch.matmul(others, others.t())
        if span_lambda > 0.0:
            eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            gram = gram + float(span_lambda) * eye
            rhs = torch.matmul(others, P[idx])
            weights = torch.linalg.solve(gram, rhs)
        else:
            rhs = torch.matmul(others, P[idx])
            weights = torch.matmul(torch.linalg.pinv(gram), rhs)
        projection = torch.matmul(weights, others)
        residual = P[idx] - projection

        residual_norm = torch.linalg.vector_norm(residual)
        if torch.isfinite(residual_norm) and residual_norm > eps:
            residuals.append(residual / residual_norm)
        else:
            residuals.append(P[idx])

    return torch.stack(residuals, dim=0)


def _build_view_transition_matrix(
    num_views: int,
    self_weight: float,
    neighbor1_weight: float,
    neighbor2_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    transition = torch.zeros(num_views, num_views, device=device, dtype=dtype)
    for i in range(num_views):
        transition[i, i] = float(self_weight)
        if i - 1 >= 0:
            transition[i, i - 1] = float(neighbor1_weight)
        if i + 1 < num_views:
            transition[i, i + 1] = float(neighbor1_weight)
        if i - 2 >= 0:
            transition[i, i - 2] = float(neighbor2_weight)
        if i + 2 < num_views:
            transition[i, i + 2] = float(neighbor2_weight)
    return transition / transition.sum(dim=1, keepdim=True).clamp_min(1e-9)


def _compute_view_prototype_context(
    E_q_norm: torch.Tensor,
    Omega_q: torch.Tensor,
    E_g_norm: torch.Tensor,
    Omega_g: torch.Tensor,
    *,
    prototype_path: str,
    prototype_temp: float,
    prototype_use_span: bool,
    prototype_span_lambda: float,
    transition_self: float,
    transition_neighbor1: float,
    transition_neighbor2: float,
    use_uncertainty_gate: bool,
    eps_margin: float,
    view_prototype_repository: Optional[ViewPrototypeRepository] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repository = view_prototype_repository or _DEFAULT_VIEW_PROTOTYPE_REPOSITORY
    prototypes = repository.load(
        prototype_path,
        device=E_q_norm.device,
        dtype=E_q_norm.dtype,
        use_span=prototype_use_span,
        span_lambda=prototype_span_lambda,
    )
    transition = _build_view_transition_matrix(
        prototypes.size(0),
        transition_self,
        transition_neighbor1,
        transition_neighbor2,
        device=E_q_norm.device,
        dtype=E_q_norm.dtype,
    )

    q_logits = float(prototype_temp) * torch.einsum("qcd,md->qcm", E_q_norm, prototypes)
    g_logits = float(prototype_temp) * torch.einsum("gcd,md->gcm", E_g_norm, prototypes)
    q_view_probs = torch.softmax(q_logits, dim=-1)
    g_view_probs = torch.softmax(g_logits, dim=-1)

    q_weights = Omega_q.float() / (Omega_q.float().sum(dim=1, keepdim=True) + eps_margin)
    g_weights = Omega_g.float() / (Omega_g.float().sum(dim=1, keepdim=True) + eps_margin)
    pi_q = torch.einsum("qc,qcm->qm", q_weights, q_view_probs)
    pi_g = torch.einsum("gc,gcm->gm", g_weights, g_view_probs)

    q_bridge = torch.matmul(pi_q, transition)
    g_bridge = torch.matmul(pi_g, transition)
    bridge = q_bridge[:, None, :] * g_bridge[None, :, :]
    bridge = bridge / bridge.sum(dim=-1, keepdim=True).clamp_min(eps_margin)
    context = torch.einsum("qgm,md->qgd", bridge, prototypes)

    if use_uncertainty_gate:
        gate_q = 1.0 - q_view_probs.amax(dim=-1)
        gate_g = 1.0 - g_view_probs.amax(dim=-1)
    else:
        gate_q = torch.ones_like(q_view_probs[..., 0])
        gate_g = torch.ones_like(g_view_probs[..., 0])

    return context, gate_q.clamp(0.0, 1.0), gate_g.clamp(0.0, 1.0)


def _propagate_similarity_with_view_context(
    sim_app: torch.Tensor,
    E_q_norm: torch.Tensor,
    E_g_norm: torch.Tensor,
    context: torch.Tensor,
    gate_q: torch.Tensor,
    gate_g: torch.Tensor,
    propagation_lambda: float,
) -> torch.Tensor:
    lam = float(propagation_lambda)
    if lam <= 0.0:
        return sim_app

    a = lam * gate_q.float().clamp_min(0.0)
    b = lam * gate_g.float().clamp_min(0.0)

    q_ctx = torch.einsum("qkd,qgd->qgk", E_q_norm, context)
    g_ctx = torch.einsum("gld,qgd->qgl", E_g_norm, context)
    ctx_sq = torch.einsum("qgd,qgd->qg", context, context).clamp_min(1e-9)

    a_exp = a[:, None, :]
    b_exp = b[None, :, :]
    num = (
        sim_app
        + a_exp[:, :, :, None] * g_ctx[:, :, None, :]
        + b_exp[:, :, None, :] * q_ctx[:, :, :, None]
        + (a_exp[:, :, :, None] * b_exp[:, :, None, :]) * ctx_sq[:, :, None, None]
    )

    q_norm = torch.sqrt(
        (1.0 + 2.0 * a_exp * q_ctx + (a_exp * a_exp) * ctx_sq[:, :, None]).clamp_min(1e-6)
    )
    g_norm = torch.sqrt(
        (1.0 + 2.0 * b_exp * g_ctx + (b_exp * b_exp) * ctx_sq[:, :, None]).clamp_min(1e-6)
    )
    den = (q_norm[:, :, :, None] * g_norm[:, :, None, :]).clamp_min(1e-6)
    return (num / den).clamp(-1.0, 1.0)


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy avec label smoothing pour régularisation.
    """
    
    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Label smoothing
        targets_one_hot = F.one_hot(targets, n_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes
        
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CircleLoss(nn.Module):
    """
    Circle Loss pour l'apprentissage métrique.
    Alternative plus flexible à la triplet loss.
    """
    
    def __init__(self, m: float = 0.25, gamma: float = 256.0):
        super().__init__()
        self.m = m
        self.gamma = gamma
    
    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=1)
        sim = feats @ feats.t()  # (B, B)
        
        B = sim.size(0)
        eye = torch.eye(B, device=sim.device, dtype=torch.bool)
        labels = labels.view(-1, 1)
        
        # Masques positifs et négatifs
        pos_mask = (labels == labels.t()) & (~eye)
        neg_mask = (labels != labels.t())
        
        # Extraction des similarités
        sp = sim[pos_mask]
        sn = sim[neg_mask]
        
        if sp.numel() == 0 or sn.numel() == 0:
            return torch.tensor(0.0, device=sim.device)
        
        # Paramètres adaptatifs
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        
        dp = 1 - self.m
        dn = self.m
        
        # Logits pondérés
        logit_p = -ap * (sp - dp) * self.gamma
        logit_n = an * (sn - dn) * self.gamma
        
        # Circle loss
        loss = F.softplus(
            torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)
        )
        
        return loss


class TripletLoss(nn.Module):
    """
    Implémentation locale d'une triplet loss margin-ranking pour éviter la dépendance torchreid.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # inputs: (B, D) embeddings
        n = inputs.size(0)
        # Distance euclidienne
        dist = torch.cdist(inputs, inputs, p=2)

        # Masques
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos

        # Hardest positive/negative pour chaque anchor
        dist_ap, _ = (dist * mask_pos.float()).max(dim=1)
        # Pour les négatifs, on met +inf sur la diag et sur pos pour ignorer
        dist_neg = dist + 1e5 * mask_pos.float()
        dist_an, _ = dist_neg.min(dim=1)

        y = dist_an.new_ones(dist_an.size())
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    """
    Center loss pour compacter les features intra-classe.
    """
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        denom = features.size(0)
        if denom == 0:
            # Conserve un tenseur lié au graph pour éviter des pertes sans grad.
            return features.sum() * 0.0
        centers_batch = self.centers[labels]
        loss = 0.5 * ((features - centers_batch) ** 2).sum() / denom
        return loss


def set_to_set_similarity(Eq: torch.Tensor, Rq: Optional[torch.Tensor], Oq: torch.Tensor,
                          Eg: torch.Tensor, Rg: Optional[torch.Tensor], Og: torch.Tensor,
                          *,
                          Z_q: Optional[torch.Tensor] = None, Z_g: Optional[torch.Tensor] = None,
                          sim_mode: str = "mix", alpha: float = 0.5,
                          temp: float = 0.15, use_wp: bool = True,
                          use_zscore: bool = False, zscore_kappa: float = 2.5,
                          crg_lambda: float = 0.5,
                          rel_are_normalized: bool = False, weights_source: str = "beta",
                          use_ot: bool = False, Uq_cells: Optional[torch.Tensor] = None,
                          Ug_cells: Optional[torch.Tensor] = None,
                          Gamma_q: Optional[torch.Tensor] = None, Gamma_g: Optional[torch.Tensor] = None,
                          ot_epsilon: float = 0.01, ot_num_iters: int = 100,
                          uniform_ot_marginals: bool = False,
                          use_cell_ot: bool = False,
                          use_stripe_ot: bool = False,
                          ot_margi_eps: float = 1e-9,
                          use_cross_view_consistency: bool = False,
                          cross_view_alpha: float = 0.7,
                          cross_view_pos_lambda: float = 0.75,
                          cross_view_phi_scale: float = 8.0,
                          cross_view_phi_bias: float = 0.5,
                          cross_view_norm_transitive: bool = True,
                          use_cross_view_row_consistency: bool = False,
                          cross_view_row_weight: float = 0.0,
                          cross_view_row_pos_lambda: float = 0.75,
                          Orient_q: Optional[torch.Tensor] = None,
                          Orient_g: Optional[torch.Tensor] = None,
                          use_orientation_guided_ot: bool = False,
                          orientation_ot_cost_weight: float = 0.05,
                          orientation_ot_mass_weight: float = 1.0,
                          use_view_prototype_propagation: bool = False,
                          view_prototype_path: str = "",
                          view_propagation_lambda: float = 0.15,
                          view_prototype_temp: float = 10.0,
                          use_view_prototype_span: bool = False,
                          view_prototype_span_lambda: float = 1e-3,
                          view_transition_self: float = 1.0,
                          view_transition_neighbor1: float = 0.7,
                          view_transition_neighbor2: float = 0.2,
                          use_view_uncertainty_gate: bool = True,
                          view_prototype_repository: Optional[ViewPrototypeRepository] = None) -> torch.Tensor:
    """
    Calcule la similarité ensemble-à-ensemble entre query et gallery.
    
    Args:
        Eq, Eg: Embeddings (Bq, C, D), (Bg, C, D)
        Rq, Rg: Signatures relationnelles (Bq, D)/(Bq, C, C), (Bg, D)/(Bg, C, C)
        Oq, Og: Poids Omega (Bq, C), (Bg, C)
        sim_mode: "app", "rel", ou "mix"
        alpha: Poids pour le mélange app/rel
        temp: Température pour softmax
        use_wp: Utiliser les poids dans l'agrégation
        use_zscore: Normalisation z-score pour C >= 9
        zscore_kappa: Paramètre de mise à l'échelle pour sigmoid
        
    Returns:
        S: Matrice de similarité (Bq, Bg)
    """
    # Sanity checks pour éviter des broadcasts silencieux
    assert Eq.shape[1] == Eg.shape[1], f"C mismatch Eq {Eq.shape} vs Eg {Eg.shape}"
    assert Oq.shape[1] == Og.shape[1] == Eq.shape[1], "Omega doit avoir taille C"

    # Compatibilité legacy: use_ot=True active OT stripe si aucun mode explicite n'est fourni.
    if use_ot and (not use_cell_ot) and (not use_stripe_ot):
        use_stripe_ot = True

    if use_cell_ot and use_stripe_ot:
        raise ValueError("OT config ambiguity: both cell OT and stripe OT are enabled")

    # Branche OT cellule-à-cellule
    if use_cell_ot:
        if Uq_cells is None or Ug_cells is None or Gamma_q is None or Gamma_g is None:
            raise ValueError(
                "Cell OT requires Uq_cells, Ug_cells, Gamma_q and Gamma_g"
            )
        return cell_level_ot_matching(
            Uq_cells.float(), Gamma_q.float(),
            Ug_cells.float(), Gamma_g.float(),
            epsilon=ot_epsilon,
            num_iterations=ot_num_iters,
            eps_margin=ot_margi_eps,
        )

    # Branche OT stripe-à-stripe
    if use_stripe_ot:
        Oq_ot = Oq.float()
        Og_ot = Og.float()
        if (
            weights_source == "gamma"
            and not uniform_ot_marginals
            and Gamma_q is not None
            and Gamma_g is not None
        ):
            # Optional Gamma experiment: use cell/GNN confidence as stripe OT marginals.
            # Reference dynamic margins keep Oq/Og, which are Beta-derived.
            Oq_ot = Gamma_q.float().mean(dim=2)
            Og_ot = Gamma_g.float().mean(dim=2)
        return stripe_level_ot_matching(
            Eq.float(), Oq_ot,
            Eg.float(), Og_ot,
            epsilon=ot_epsilon,
            num_iterations=ot_num_iters,
            eps_margin=ot_margi_eps,
            uniform_marginals=uniform_ot_marginals,
            use_cross_view_consistency=use_cross_view_consistency,
            cross_view_alpha=cross_view_alpha,
            cross_view_pos_lambda=cross_view_pos_lambda,
            cross_view_phi_scale=cross_view_phi_scale,
            cross_view_phi_bias=cross_view_phi_bias,
            cross_view_norm_transitive=cross_view_norm_transitive,
            Uhat_q=Uq_cells,
            Uhat_g=Ug_cells,
            Z_q=Z_q,
            Z_g=Z_g,
            Gamma_q=Gamma_q,
            Gamma_g=Gamma_g,
            use_cross_view_row_consistency=use_cross_view_row_consistency,
            cross_view_row_weight=cross_view_row_weight,
            cross_view_row_pos_lambda=cross_view_row_pos_lambda,
            Orient_q=Orient_q,
            Orient_g=Orient_g,
            use_orientation_guided_ot=use_orientation_guided_ot,
            orientation_ot_cost_weight=orientation_ot_cost_weight,
            orientation_ot_mass_weight=orientation_ot_mass_weight,
            use_view_prototype_propagation=use_view_prototype_propagation,
            view_prototype_path=view_prototype_path,
            view_propagation_lambda=view_propagation_lambda,
            view_prototype_temp=view_prototype_temp,
            use_view_prototype_span=use_view_prototype_span,
            view_prototype_span_lambda=view_prototype_span_lambda,
            view_transition_self=view_transition_self,
            view_transition_neighbor1=view_transition_neighbor1,
            view_transition_neighbor2=view_transition_neighbor2,
            use_view_uncertainty_gate=use_view_uncertainty_gate,
            view_prototype_repository=view_prototype_repository,
            crg_lambda=crg_lambda,
        )
    
    Bq = Eq.size(0)
    Bg = Eg.size(0)
    C = Eq.size(1)
    use_z = use_zscore and C >= 9
    has_rel_inputs = Rq is not None and Rg is not None
    need_rel = sim_mode != "app"
    # Si mode rel sans relation disponible, fallback implicite vers app.
    need_app = (sim_mode != "rel") or use_z or (not has_rel_inputs)

    sim_app = None
    if need_app:
        # sim_app: (Bq, Bg, C, C) - similarité apparence entre toutes les paires de stripes
        if Z_q is not None and Z_g is not None and Z_q.is_complex() and Z_g.is_complex():
            sim_app = hermitian_stripe_similarity(Z_q, Z_g, normalize=True, crg_lambda=crg_lambda)
        else:
            sim_app = torch.einsum("qcd,gkd->qgck", Eq, Eg)
    
    sim_rel = None
    if need_rel and has_rel_inputs:
        assert Rq is not None and Rg is not None
        if Rq.dim() == 2 and Rg.dim() == 2:
            # rel_vec path: similarité relationnelle globale (Bq, Bg), diffusée sur (C, C)
            if rel_are_normalized:
                rq = Rq.float()
                rg = Rg.float()
            else:
                rq = F.normalize(Rq.float(), dim=1)
                rg = F.normalize(Rg.float(), dim=1)
            sim_rel_pair = torch.matmul(rq, rg.t())  # (Bq, Bg)
            sim_rel = sim_rel_pair.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, C)
        elif Rq.dim() == 3 and Rg.dim() == 3:
            # compatibilité descendante
            sim_rel = torch.einsum("qcs,gks->qgck", Rq, Rg)
        else:
            raise ValueError(f"Rq/Rg dimensions invalides: {Rq.shape}, {Rg.shape}")
    
    if use_z:
        assert sim_app is not None
        # Calcul des statistiques intra-image
        eps = 1e-9
        
        # Stats pour queries
        eye_q = torch.eye(C, device=Eq.device).unsqueeze(0)
        sim_intra_q = torch.einsum("qcd,qkd->qck", Eq, Eq)
        vals_q = sim_intra_q.masked_select(~eye_q.bool()).view(Bq, -1)
        mu_q = vals_q.mean(dim=1, keepdim=True).view(Bq, 1, 1, 1)
        std_q = vals_q.std(dim=1, unbiased=False, keepdim=True).view(Bq, 1, 1, 1)
        
        # Stats pour gallery
        eye_g = torch.eye(C, device=Eg.device).unsqueeze(0)
        sim_intra_g = torch.einsum("gcd,gkd->gck", Eg, Eg)
        vals_g = sim_intra_g.masked_select(~eye_g.bool()).view(Bg, -1)
        mu_g = vals_g.mean(dim=1, keepdim=True).view(1, Bg, 1, 1)
        std_g = vals_g.std(dim=1, unbiased=False, keepdim=True).view(1, Bg, 1, 1)
        
        # Z-scores symétriques
        z_x = (sim_app - mu_q) / (std_q + eps)
        z_y = (sim_app - mu_g) / (std_g + eps)
        z_sym = 0.5 * (z_x + z_y)
        z_prob = torch.sigmoid(z_sym / zscore_kappa)
        
        s_rel = (sim_rel + 1.0) * 0.5 if sim_rel is not None else z_prob
        sim = alpha * z_prob + (1 - alpha) * s_rel
    else:
        # Mode standard
        assert sim_app is not None or sim_rel is not None
        if sim_mode == "app":
            sim = sim_app
        elif sim_mode == "rel" and sim_rel is not None:
            sim = sim_rel
        else:  # mix
            assert sim_app is not None
            sim_app_t = sim_app
            if sim_rel is not None:
                sim = alpha * sim_app_t + (1 - alpha) * sim_rel
            else:
                sim = sim_app_t

    # Filet de sécurité : borne les logits avant softmax pour éviter inf/NaN
    assert sim is not None
    sim = sim.clamp(-50, 50)

    # Poids pour l'attention (produit extérieur des Omegas)
    wp = (Oq.unsqueeze(1).unsqueeze(-1) * Og.unsqueeze(0).unsqueeze(2)).clamp_min(1e-6)
    wp_agg = wp if use_wp else torch.ones_like(wp)
    
    # Soft matching bidirectionnel
    # Direction 1: query stripes -> gallery stripes
    log_wp = torch.log(wp + 1e-12)  # éviter -inf quand wp==0 après clamp
    att_k = torch.softmax(sim / temp + log_wp, dim=3)  # (Bq, Bg, C, C)
    score_1 = (att_k * sim * wp_agg).sum(dim=3) / (att_k * wp_agg).sum(dim=3).clamp_min(1e-6)
    
    # Direction 2: gallery stripes -> query stripes
    att_c = torch.softmax(sim / temp + log_wp, dim=2)
    score_2 = (att_c * sim * wp_agg).sum(dim=2) / (att_c * wp_agg).sum(dim=2).clamp_min(1e-6)
    
    # Moyenne des deux directions et sur les stripes
    S = 0.5 * (score_1.mean(dim=2) + score_2.mean(dim=2))  # (Bq, Bg)
    
    return S


def sinkhorn_algorithm(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.01,
    num_iterations: int = 100,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Résout le problème de transport optimal régularisé (Sinkhorn-Knopp).
    
    Args:
        C: Matrice de coût (Bq*Bg, M, M) où M = C*R (nombre de cellules)
        a: Marginals source (Bq*Bg, M) - poids de visibilité query
        b: Marginals destination (Bq*Bg, M) - poids de visibilité gallery
        epsilon: Paramètre de régularisation entropique
        num_iterations: Nombre d'itérations Sinkhorn
        eps: Epsilon de stabilité
        
    Returns:
        P: Plan de transport optimal (Bq*Bg, M, M)
    """
    device = C.device
    dtype = C.dtype
    
    # Initialiser le kernel: K = exp(-C / epsilon)
    K = torch.exp(-C / epsilon).to(dtype)
    
    # Itérations Sinkhorn : P^{(t+1)} = diag(u) @ K @ diag(v)
    batch_shape = C.shape[:-2]
    M = C.shape[-1]
    
    # Initialiser u et v
    u = torch.ones(*batch_shape, M, device=device, dtype=dtype)
    v = torch.ones(*batch_shape, M, device=device, dtype=dtype)
    
    for _ in range(num_iterations):
        # u -> a / (K @ v)
        Kv = torch.einsum("...ij,...j->...i", K, v)
        u = a / (Kv + eps)
        
        # v -> b / (K.T @ u)
        KTu = torch.einsum("...ij,...i->...j", K, u)
        v = b / (KTu + eps)
    
    # Plan de transport final: P = diag(u) @ K @ diag(v)
    P = torch.einsum("...i,...ij,...j->...ij", u, K, v)
    
    return P


def cell_level_ot_matching(
    Uhat_q: torch.Tensor,
    Gamma_q: torch.Tensor,
    Uhat_g: torch.Tensor,
    Gamma_g: torch.Tensor,
    epsilon: float = 0.01,
    num_iterations: int = 100,
    eps_margin: float = 1e-9,
) -> torch.Tensor:
    """
    Matching au niveau des cellules via Optimal Transport avec marginals visibility-aware.
    
    Args:
        Uhat_q: Descripteurs raffinés query (Bq, C, R, D)
        Gamma_q: Poids de visibilité query (Bq, C, R) - confiance GNN
        Uhat_g: Descripteurs raffinés gallery (Bg, C, R, D)
        Gamma_g: Poids de visibilité gallery (Bg, C, R)
        epsilon: Régularisation entropique de Sinkhorn
        num_iterations: Itérations Sinkhorn
        eps_margin: Stabilité pour les marginals
        
    Returns:
        S_ot: Matrice de similarité (Bq, Bg) basée sur OT
    """
    Bq, C, R, D = Uhat_q.shape
    Bg = Uhat_g.shape[0]
    device = Uhat_q.device
    dtype = Uhat_q.dtype
    
    # Déplier cellules: (Bq, C, R, D) -> (Bq, M, D) où M = C*R
    M = C * R
    U_q_flat = Uhat_q.view(Bq, M, D)  # (Bq, M, D)
    U_g_flat = Uhat_g.view(Bg, M, D)  # (Bg, M, D)
    
    # Normaliser
    U_q_norm = F.normalize(U_q_flat, dim=-1)  # (Bq, M, D)
    U_g_norm = F.normalize(U_g_flat, dim=-1)  # (Bg, M, D)
    
    # Matrice de coût: distance cosinus entre toutes paires de cellules
    # C[b_i, b_j, i, j] = 1 - cosine_sim(U_q_norm[b_i, i], U_g_norm[b_j, j])
    C = 1.0 - torch.einsum("qid,gjd->qgij", U_q_norm, U_g_norm)  # (Bq, Bg, M, M)
    C = C.clamp(0.0, 2.0)  # Clamp à [0, 2]
    
    # Marginals visibility-aware
    Gamma_q_flat = Gamma_q.view(Bq, M)  # (Bq, M)
    Gamma_g_flat = Gamma_g.view(Bg, M)  # (Bg, M)
    
    # Normaliser pour obtenir des distributions de probabilité
    # a[q, i] = gamma_q[q, i] / sum(gamma_q[q])
    a = Gamma_q_flat / (Gamma_q_flat.sum(dim=1, keepdim=True) + eps_margin)  # (Bq, M)
    b = Gamma_g_flat / (Gamma_g_flat.sum(dim=1, keepdim=True) + eps_margin)  # (Bg, M)
    
    # Résoudre OT pour chaque paire (q, g)
    # Reshaper C en (Bq*Bg, M, M)
    C_reshaped = C.reshape(Bq * Bg, M, M)
    a_expanded = a.repeat_interleave(Bg, dim=0)  # (Bq*Bg, M)
    b_expanded = b.repeat(Bq, 1)  # (Bq*Bg, M)
    
    # Sinkhorn
    P = sinkhorn_algorithm(
        C_reshaped, a_expanded, b_expanded,
        epsilon=epsilon,
        num_iterations=num_iterations,
        eps=eps_margin,
    )  # (Bq*Bg, M, M)
    
    # Distance de Sinkhorn: <P, C>_F = sum(P * C)
    S_ot_vec = (P * C_reshaped).sum(dim=(1, 2))  # (Bq*Bg,)
    S_ot = S_ot_vec.reshape(Bq, Bg)  # (Bq, Bg)
    
    # Transformer en similarité: S = 1 - D (plus grand = plus similaire)
    S_ot = 1.0 - S_ot.clamp(0.0, 2.0)
    
    return S_ot


def _cross_view_transitive_from_similarity(
    sim01: torch.Tensor,
    pos_lambda: float,
    phi_scale: float,
    phi_bias: float,
    norm_transitive: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Second-order transitive similarity over the last two axes of sim01."""
    n = sim01.size(-1)
    idx = torch.arange(n, device=sim01.device, dtype=sim01.dtype)
    prior = torch.exp(-float(pos_lambda) * (idx.view(n, 1) - idx.view(1, n)).abs())
    prior = prior / prior.max().clamp_min(1e-9)

    view_shape = (1,) * (sim01.dim() - 2) + (n, n)
    phi = torch.sigmoid(float(phi_scale) * (sim01 - float(phi_bias)))
    m_tilde = prior.view(view_shape) * phi

    s_trans = torch.matmul(m_tilde, m_tilde)
    if norm_transitive:
        s_trans = s_trans / s_trans.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
    return s_trans.clamp(0.0, 1.0), prior


def hermitian_stripe_similarity(
    Z_q: torch.Tensor,
    Z_g: torch.Tensor,
    normalize: bool = True,
    crg_lambda: float = 0.5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute Cross-Residual Gated Hermitian stripe similarity.

    For every query/gallery sample pair and every stripe pair (i, j):
      H_ij = <Z_i^q, Z_j^g>_C = conj(Z_i^q)^T Z_j^g
      S_ij = |Re(H_ij)| * exp(-lambda * |Im(H_ij)| / (|Re(H_ij)| + eps)).

    Args:
        Z_q: (Bq, Cq, D) complex query stripes
        Z_g: (Bg, Cg, D) complex gallery stripes
        normalize: if True, divide H by complex vector norms to keep S in [0,1]
        crg_lambda: strength of the imaginary residual penalty
        eps: numerical stability epsilon

    Returns:
        S: (Bq, Bg, Cq, Cg) similarity matrix.
    """
    if not Z_q.is_complex() or not Z_g.is_complex():
        raise TypeError(f"Expected complex tensors, got Z_q.dtype={Z_q.dtype}, Z_g.dtype={Z_g.dtype}")

    if Z_q.dim() != 3 or Z_g.dim() != 3:
        raise ValueError(f"Expected (B,C,D) tensors, got Z_q={Z_q.shape}, Z_g={Z_g.shape}")
    if Z_q.size(-1) != Z_g.size(-1):
        raise ValueError(f"Feature dim mismatch: Z_q={Z_q.shape}, Z_g={Z_g.shape}")

    # Hermitian stripe-to-stripe interaction:
    # H[q,g,i,j] = sum_d conj(Z_q[q,i,d]) * Z_g[g,j,d]
    hermitian_prods = torch.einsum("qid,gjd->qgij", torch.conj(Z_q), Z_g)

    if normalize:
        q_norm = torch.sqrt((torch.abs(Z_q) ** 2).sum(dim=-1)).clamp_min(eps)  # (Bq, Cq)
        g_norm = torch.sqrt((torch.abs(Z_g) ** 2).sum(dim=-1)).clamp_min(eps)  # (Bg, Cg)
        denom = q_norm[:, None, :, None] * g_norm[None, :, None, :]
        hermitian_prods = hermitian_prods / denom.clamp_min(eps)

    real_abs = hermitian_prods.real.abs()
    imag_abs = hermitian_prods.imag.abs()
    gate = torch.exp(-float(crg_lambda) * imag_abs / (real_abs + float(eps)))
    sim = real_abs * gate

    return sim.clamp(0.0, 1.0)


def _orientation_cost_matrix(
    Orient_q: Optional[torch.Tensor],
    Orient_g: Optional[torch.Tensor],
    Bq: int,
    Bg: int,
    C: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Return C_ori[q,g,i,j] = 1 - cos(theta_qi - theta_gj).

    Supported orientation formats:
    - (B, C, 2): single orientation vector per stripe.
    - (B, C, 4): dual orientation (abs + rel). In this case, OT uses
        the absolute component (first 2 dims) by default.
    """
    if Orient_q is None or Orient_g is None:
        return None
    if Orient_q.dim() == 2:
        Orient_q = Orient_q.unsqueeze(1).expand(-1, C, -1)
    if Orient_g.dim() == 2:
        Orient_g = Orient_g.unsqueeze(1).expand(-1, C, -1)
    if Orient_q.dim() != 3 or Orient_g.dim() != 3:
        raise ValueError(
            f"Orientation-guided OT expects (B,C,2), got {Orient_q.shape} and {Orient_g.shape}"
        )
    if Orient_q.shape[:2] != (Bq, C) or Orient_g.shape[:2] != (Bg, C):
        raise ValueError(
            f"Orientation shape mismatch: Orient_q={Orient_q.shape}, Orient_g={Orient_g.shape}, "
            f"expected ({Bq},{C},2) and ({Bg},{C},2)"
        )

    # Dual-FiLM orientation support: if abs+rel are concatenated, use abs for OT guidance.
    if Orient_q.size(-1) == 4 and Orient_g.size(-1) == 4:
        Orient_q = Orient_q[..., :2]
        Orient_g = Orient_g[..., :2]
    elif Orient_q.size(-1) != 2 or Orient_g.size(-1) != 2:
        raise ValueError(
            "Orientation vectors must have last dim=2 (or 4 for abs+rel), "
            f"got {Orient_q.shape} and {Orient_g.shape}"
        )

    oq = F.normalize(Orient_q.to(device=device, dtype=torch.float32), dim=-1, eps=1e-6)
    og = F.normalize(Orient_g.to(device=device, dtype=torch.float32), dim=-1, eps=1e-6)
    dot = torch.einsum("qid,gjd->qgij", oq, og).clamp(-1.0, 1.0)
    return (1.0 - dot).to(dtype=dtype).clamp(0.0, 2.0)


def stripe_level_ot_matching(
    Ehat_q: torch.Tensor,
    Omega_q: torch.Tensor,
    Ehat_g: torch.Tensor,
    Omega_g: torch.Tensor,
    epsilon: float = 0.01,
    num_iterations: int = 100,
    eps_margin: float = 1e-9,
    uniform_marginals: bool = False,
    use_cross_view_consistency: bool = False,
    cross_view_alpha: float = 0.7,
    cross_view_pos_lambda: float = 0.75,
    cross_view_phi_scale: float = 8.0,
    cross_view_phi_bias: float = 0.5,
    cross_view_norm_transitive: bool = True,
    Uhat_q: Optional[torch.Tensor] = None,
    Uhat_g: Optional[torch.Tensor] = None,
    Gamma_q: Optional[torch.Tensor] = None,
    Gamma_g: Optional[torch.Tensor] = None,
    use_cross_view_row_consistency: bool = False,
    cross_view_row_weight: float = 0.0,
    cross_view_row_pos_lambda: float = 0.75,
    use_view_prototype_propagation: bool = False,
    view_prototype_path: str = "",
    view_propagation_lambda: float = 0.15,
    view_prototype_temp: float = 10.0,
    use_view_prototype_span: bool = False,
    view_prototype_span_lambda: float = 1e-3,
    view_transition_self: float = 1.0,
    view_transition_neighbor1: float = 0.7,
    view_transition_neighbor2: float = 0.2,
    use_view_uncertainty_gate: bool = True,
    view_prototype_repository: Optional[ViewPrototypeRepository] = None,
    Z_q: Optional[torch.Tensor] = None,
    Z_g: Optional[torch.Tensor] = None,
    Orient_q: Optional[torch.Tensor] = None,
    Orient_g: Optional[torch.Tensor] = None,
    use_orientation_guided_ot: bool = False,
    orientation_ot_cost_weight: float = 0.05,
    orientation_ot_mass_weight: float = 1.0,
    crg_lambda: float = 0.5,
) -> torch.Tensor:
    """
    Matching stripe-à-stripe via Optimal Transport avec marginals Omega.

    Args:
        Ehat_q: Stripes query (Bq, C, D)
        Omega_q: Poids de visibilité query (Bq, C)
        Ehat_g: Stripes gallery (Bg, C, D)
        Omega_g: Poids de visibilité gallery (Bg, C)
        Z_q: Complex Hermitian embeddings query (Bq, C, D) with dtype complex128/complex64
        Z_g: Complex Hermitian embeddings gallery (Bg, C, D) with dtype complex128/complex64
        uniform_marginals: Si True, utilise marges uniformes (1/C) au lieu d'Omega

    Returns:
        S_ot: Matrice de similarité (Bq, Bg)
    """
    Bq, C, D = Ehat_q.shape
    Bg = Ehat_g.shape[0]

    E_q_norm = F.normalize(Ehat_q, dim=-1)  # (Bq, C, D)
    E_g_norm = F.normalize(Ehat_g, dim=-1)  # (Bg, C, D)

    # Use complex Hermitian similarity if available
    use_hermitian = False
    if Z_q is not None and Z_g is not None and Z_q.is_complex() and Z_g.is_complex():
        # Cross-Residual Gated Hermitian similarity:
        # S[q,g,i,j] = |Re(H)| * exp(-lambda * |Im(H)| / (|Re(H)| + eps)).
        sim_app = hermitian_stripe_similarity(
            Z_q,
            Z_g,
            normalize=True,
            crg_lambda=crg_lambda,
            eps=eps_margin,
        )  # (Bq, Bg, C, C)
        use_hermitian = True
    else:
        sim_app = torch.einsum("qkd,gld->qgkl", E_q_norm, E_g_norm).clamp(-1.0, 1.0)  # (Bq, Bg, C, C)
    if use_view_prototype_propagation:
        if not view_prototype_path:
            raise ValueError("View prototype propagation requires view_prototype_path")
        context, gate_q, gate_g = _compute_view_prototype_context(
            E_q_norm,
            Omega_q,
            E_g_norm,
            Omega_g,
            prototype_path=view_prototype_path,
            prototype_temp=view_prototype_temp,
            prototype_use_span=use_view_prototype_span,
            prototype_span_lambda=view_prototype_span_lambda,
            transition_self=view_transition_self,
            transition_neighbor1=view_transition_neighbor1,
            transition_neighbor2=view_transition_neighbor2,
            use_uncertainty_gate=use_view_uncertainty_gate,
            eps_margin=eps_margin,
            view_prototype_repository=view_prototype_repository,
        )
        sim_app = _propagate_similarity_with_view_context(
            sim_app,
            E_q_norm,
            E_g_norm,
            context,
            gate_q,
            gate_g,
            propagation_lambda=view_propagation_lambda,
        )

    if use_cross_view_consistency:
        # Similarité en [0,1] pour la modulation sigmoïde.
        # - cosine branch: sim_app in [-1,1] -> map to [0,1]
        # - Hermitian branch: sim_app already in [0,1]
        sim01 = sim_app if use_hermitian else (0.5 * (sim_app + 1.0))

        s_trans, _ = _cross_view_transitive_from_similarity(
            sim01,
            cross_view_pos_lambda,
            cross_view_phi_scale,
            cross_view_phi_bias,
            cross_view_norm_transitive,
        )

        s_final01 = cross_view_alpha * sim01 + (1.0 - cross_view_alpha) * s_trans

        row_weight = float(max(0.0, min(1.0, cross_view_row_weight)))
        if use_cross_view_row_consistency and row_weight > 0.0:
            if Uhat_q is None or Uhat_g is None:
                raise ValueError("Row cross-view consistency requires Uhat_q and Uhat_g")
            if Uhat_q.dim() != 4 or Uhat_g.dim() != 4:
                raise ValueError(f"Uhat tensors must be (B,C,R,D), got {Uhat_q.shape} and {Uhat_g.shape}")
            if Uhat_q.size(1) != C or Uhat_g.size(1) != C or Uhat_q.size(2) != Uhat_g.size(2):
                raise ValueError(f"Uhat shape mismatch for row cross-view: {Uhat_q.shape} vs {Uhat_g.shape}")

            Uq_norm = F.normalize(Uhat_q.float(), dim=-1)
            Ug_norm = F.normalize(Uhat_g.float(), dim=-1)
            # Per stripe pair (k,l), compare row sequences r,s: (Bq,Bg,C,C,R,R).
            sim_row = torch.einsum("qkrd,glsd->qgklrs", Uq_norm, Ug_norm).clamp(-1.0, 1.0)
            sim_row01 = 0.5 * (sim_row + 1.0)
            row_trans, row_prior = _cross_view_transitive_from_similarity(
                sim_row01,
                cross_view_row_pos_lambda,
                cross_view_phi_scale,
                cross_view_phi_bias,
                cross_view_norm_transitive,
            )

            R = Uhat_q.size(2)
            row_w = row_prior.view(1, 1, 1, 1, R, R)
            if Gamma_q is not None and Gamma_g is not None:
                if Gamma_q.shape[:3] != Uhat_q.shape[:3] or Gamma_g.shape[:3] != Uhat_g.shape[:3]:
                    raise ValueError(f"Gamma shape mismatch for row cross-view: {Gamma_q.shape} vs {Gamma_g.shape}")
                gamma_pair = (
                    Gamma_q.float().clamp_min(0.0)[:, None, :, None, :, None]
                    * Gamma_g.float().clamp_min(0.0)[None, :, None, :, None, :]
                )
                row_w = row_w * gamma_pair
            row_den = row_w.sum(dim=(-2, -1)).clamp_min(1e-9)
            row_direct = (sim_row01 * row_w).sum(dim=(-2, -1)) / row_den
            row_consistency = (row_trans * row_w).sum(dim=(-2, -1)) / row_den
            row_final01 = (
                cross_view_alpha * row_direct
                + (1.0 - cross_view_alpha) * row_consistency
            ).clamp(0.0, 1.0)
            s_final01 = (1.0 - row_weight) * s_final01 + row_weight * row_final01

        s_final = 2.0 * s_final01 - 1.0  # Retour en domaine cosinus [-1,1]
        Cost = (1.0 - s_final).clamp(0.0, 2.0)
    else:
        # Matrice de coût stripe-à-stripe standard: 1 - cos(E_q[k], E_g[l])
        Cost = (1.0 - sim_app).clamp(0.0, 2.0)

    C_ori = None
    if use_orientation_guided_ot:
        C_ori = _orientation_cost_matrix(
            Orient_q,
            Orient_g,
            Bq,
            Bg,
            C,
            device=Cost.device,
            dtype=Cost.dtype,
        )
        if C_ori is not None and orientation_ot_cost_weight > 0.0:
            Cost = (Cost + float(orientation_ot_cost_weight) * C_ori).clamp_min(0.0)

    # Marginals: distributions de probabilité sur les stripes
    if uniform_marginals:
        # Mode uniforme: 1/C pour chaque stripe
        a = torch.ones(Bq, C, device=Ehat_q.device, dtype=Ehat_q.dtype) / C
        b = torch.ones(Bg, C, device=Ehat_g.device, dtype=Ehat_g.dtype) / C
        a_expanded = a.repeat_interleave(Bg, dim=0)  # (Bq*Bg, C)
        b_expanded = b.repeat(Bq, 1)                 # (Bq*Bg, C)
    else:
        # Mode dynamique: Omega normalisé (visibilité-aware)
        a_base = Omega_q.float().clamp_min(0.0).to(device=Cost.device)
        b_base = Omega_g.float().clamp_min(0.0).to(device=Cost.device)
        if C_ori is not None and orientation_ot_mass_weight > 0.0:
            orient_aff = torch.exp(-float(orientation_ot_mass_weight) * C_ori.float()).to(dtype=Cost.dtype)
            a_pair = a_base[:, None, :] * orient_aff.mean(dim=3)  # (Bq, Bg, C_query)
            b_pair = b_base[None, :, :] * orient_aff.mean(dim=2)  # (Bq, Bg, C_gallery)
            a_expanded = a_pair.reshape(Bq * Bg, C)
            b_expanded = b_pair.reshape(Bq * Bg, C)
            a_expanded = a_expanded / a_expanded.sum(dim=1, keepdim=True).clamp_min(eps_margin)
            b_expanded = b_expanded / b_expanded.sum(dim=1, keepdim=True).clamp_min(eps_margin)
        else:
            a = a_base / (a_base.sum(dim=1, keepdim=True) + eps_margin)  # (Bq, C)
            b = b_base / (b_base.sum(dim=1, keepdim=True) + eps_margin)  # (Bg, C)
            a_expanded = a.repeat_interleave(Bg, dim=0)  # (Bq*Bg, C)
            b_expanded = b.repeat(Bq, 1)                 # (Bq*Bg, C)

    Cost_reshaped = Cost.reshape(Bq * Bg, C, C)

    P = sinkhorn_algorithm(
        Cost_reshaped, a_expanded, b_expanded,
        epsilon=epsilon,
        num_iterations=num_iterations,
        eps=eps_margin,
    )  # (Bq*Bg, C, C)

    D_ot = (P * Cost_reshaped).sum(dim=(1, 2)).reshape(Bq, Bg)  # (Bq, Bg)
    S_ot = 1.0 - D_ot.clamp(0.0, 2.0)
    return S_ot


def cross_view_similarity_stats(
    Ehat_q: torch.Tensor,
    Ehat_g: torch.Tensor,
    cross_view_alpha: float = 0.7,
    cross_view_pos_lambda: float = 0.75,
    cross_view_phi_scale: float = 8.0,
    cross_view_phi_bias: float = 0.5,
    cross_view_norm_transitive: bool = True,
) -> dict:
    """
    Compute lightweight descriptive stats for cross-view consistency terms.

    Returns means/min/max over all pairwise stripe similarities:
      - S_app in [0,1]
      - S_trans in [0,1]
      - S_final in [0,1]
    """
    with torch.no_grad():
        E_q_norm = F.normalize(Ehat_q.float(), dim=-1)
        E_g_norm = F.normalize(Ehat_g.float(), dim=-1)
        sim_app = torch.einsum("qkd,gld->qgkl", E_q_norm, E_g_norm).clamp(-1.0, 1.0)
        s_app01 = 0.5 * (sim_app + 1.0)

        C = Ehat_q.shape[1]
        idx = torch.arange(C, device=Ehat_q.device, dtype=Ehat_q.dtype)
        prior = torch.exp(-cross_view_pos_lambda * (idx.view(C, 1) - idx.view(1, C)).abs())
        prior = prior / prior.max().clamp_min(1e-9)

        phi = torch.sigmoid(cross_view_phi_scale * (s_app01 - cross_view_phi_bias))
        m_tilde = prior.view(1, 1, C, C) * phi
        s_trans = torch.matmul(m_tilde, m_tilde)
        if cross_view_norm_transitive:
            s_trans = s_trans / s_trans.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
        s_trans = s_trans.clamp(0.0, 1.0)

        s_final = (cross_view_alpha * s_app01 + (1.0 - cross_view_alpha) * s_trans).clamp(0.0, 1.0)

        return {
            "cv_s_app_mean": float(s_app01.mean().item()),
            "cv_s_app_min": float(s_app01.min().item()),
            "cv_s_app_max": float(s_app01.max().item()),
            "cv_s_trans_mean": float(s_trans.mean().item()),
            "cv_s_trans_min": float(s_trans.min().item()),
            "cv_s_trans_max": float(s_trans.max().item()),
            "cv_s_final_mean": float(s_final.mean().item()),
            "cv_s_final_min": float(s_final.min().item()),
            "cv_s_final_max": float(s_final.max().item()),
        }


def set_to_set_similarity_candidates(
    Eq: torch.Tensor,
    Rq: Optional[torch.Tensor],
    Oq: torch.Tensor,
    Eg: torch.Tensor,
    Rg: Optional[torch.Tensor],
    Og: torch.Tensor,
    sim_mode: str = "mix",
    alpha: float = 0.5,
    temp: float = 0.15,
    use_wp: bool = True,
    use_zscore: bool = False,
    zscore_kappa: float = 2.5,
    rel_are_normalized: bool = False,
) -> torch.Tensor:
    """
    Variante pour candidats par query.

    Args:
        Eq: (B, C, D)
        Eg: (B, K, C, D)
        Oq: (B, C)
        Og: (B, K, C)
        Rq: (B, D) ou (B, C, S) ou None
        Rg: (B, K, D) ou (B, K, C, S) ou None

    Returns:
        Similarité (B, K)
    """
    assert Eq.dim() == 3 and Eg.dim() == 4, f"Shapes invalides Eq {Eq.shape}, Eg {Eg.shape}"
    assert Eq.size(0) == Eg.size(0), "Batch query/candidats incohérent"
    assert Eq.size(1) == Eg.size(2), f"C mismatch Eq {Eq.shape} vs Eg {Eg.shape}"
    assert Oq.shape == Eq.shape[:2], f"Oq shape invalide: {Oq.shape} vs {Eq.shape[:2]}"
    assert Og.shape == Eg.shape[:3], f"Og shape invalide: {Og.shape} vs {Eg.shape[:3]}"

    B, C, _ = Eq.shape
    K = Eg.size(1)
    use_z = use_zscore and C >= 9
    has_rel_inputs = Rq is not None and Rg is not None
    need_rel = sim_mode != "app"
    need_app = (sim_mode != "rel") or use_z or (not has_rel_inputs)

    sim_app = None
    if need_app:
        # Similarité d'apparence entre chaque query et ses K candidats
        sim_app = torch.einsum("bcd,bkfd->bkcf", Eq, Eg)  # (B, K, C, C)

    sim_rel = None
    if need_rel and has_rel_inputs:
        assert Rq is not None and Rg is not None
        if Rq.dim() == 2 and Rg.dim() == 3:
            if rel_are_normalized:
                rq = Rq.float()
                rg = Rg.float()
            else:
                rq = F.normalize(Rq.float(), dim=1)
                rg = F.normalize(Rg.float(), dim=2)
            sim_rel_pair = torch.einsum("bd,bkd->bk", rq, rg)  # (B, K)
            sim_rel = sim_rel_pair.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, C)
        elif Rq.dim() == 3 and Rg.dim() == 4:
            sim_rel = torch.einsum("bcs,bkfs->bkcf", Rq, Rg)
        else:
            raise ValueError(f"Rq/Rg dimensions invalides: {Rq.shape}, {Rg.shape}")
    if use_z:
        assert sim_app is not None
        eps = 1e-9
        eye = torch.eye(C, device=Eq.device, dtype=torch.bool)

        sim_intra_q = torch.einsum("bcd,bfd->bcf", Eq, Eq)  # (B, C, C)
        vals_q = sim_intra_q.masked_select(~eye.unsqueeze(0)).view(B, -1)
        mu_q = vals_q.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        std_q = vals_q.std(dim=1, unbiased=False, keepdim=True).view(B, 1, 1, 1)

        sim_intra_g = torch.einsum("bkcd,bkfd->bkcf", Eg, Eg)  # (B, K, C, C)
        vals_g = sim_intra_g.masked_select(~eye.view(1, 1, C, C)).view(B, K, -1)
        mu_g = vals_g.mean(dim=2, keepdim=True).view(B, K, 1, 1)
        std_g = vals_g.std(dim=2, unbiased=False, keepdim=True).view(B, K, 1, 1)

        z_x = (sim_app - mu_q) / (std_q + eps)
        z_y = (sim_app - mu_g) / (std_g + eps)
        z_sym = 0.5 * (z_x + z_y)
        z_prob = torch.sigmoid(z_sym / zscore_kappa)

        s_rel = (sim_rel + 1.0) * 0.5 if sim_rel is not None else z_prob
        sim = alpha * z_prob + (1 - alpha) * s_rel
    else:
        assert sim_app is not None or sim_rel is not None
        if sim_mode == "app":
            sim = sim_app
        elif sim_mode == "rel" and sim_rel is not None:
            sim = sim_rel
        else:
            assert sim_app is not None
            sim_app_t = sim_app
            if sim_rel is not None:
                sim = alpha * sim_app_t + (1 - alpha) * sim_rel
            else:
                sim = sim_app_t

    assert sim is not None
    sim = sim.clamp(-50, 50)

    wp = (Oq.unsqueeze(1).unsqueeze(-1) * Og.unsqueeze(2)).clamp_min(1e-6)  # (B, K, C, C)
    wp_agg = wp if use_wp else torch.ones_like(wp)

    log_wp = torch.log(wp + 1e-12)
    att_k = torch.softmax(sim / temp + log_wp, dim=3)
    score_1 = (att_k * sim * wp_agg).sum(dim=3) / (att_k * wp_agg).sum(dim=3).clamp_min(1e-6)

    att_c = torch.softmax(sim / temp + log_wp, dim=2)
    score_2 = (att_c * sim * wp_agg).sum(dim=2) / (att_c * wp_agg).sum(dim=2).clamp_min(1e-6)

    S = 0.5 * (score_1.mean(dim=2) + score_2.mean(dim=2))  # (B, K)
    return S


def set_contrastive_loss(S: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> Tuple[torch.Tensor, float]:
    """
    Perte contrastive set-level (sans softmax/logits).
    Positif: (1 - S)^2 ; Négatif: relu(S - margin)^2
    """
    B = S.size(0)
    device = S.device

    mask_self = torch.eye(B, device=device, dtype=torch.bool)
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()) & (~mask_self)
    mask_neg = (labels != labels.t())

    pos_vals = S[mask_pos]
    neg_vals = S[mask_neg]

    # Utilise une constante liée au graph pour éviter les pertes sans grad
    zero = S.sum() * 0.0
    pos_loss = ((1 - pos_vals) ** 2).mean() if pos_vals.numel() > 0 else zero
    neg_loss = (F.relu(neg_vals - margin) ** 2).mean() if neg_vals.numel() > 0 else zero

    loss = pos_loss + neg_loss
    valid_frac = mask_pos.sum().item() / (B * (B - 1))
    return loss, valid_frac


def set_contrastive_loss_hard(
    S: torch.Tensor,
    labels: torch.Tensor,
    temp: float = 0.15,
    hard_ratio: float = 0.3,
    center_only: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    InfoNCE stable avec hard negative mining.
    - FP32
    - Masque diagonal
    - Clamp des logits (via centrage)
    - Skip si pas de positifs
    """
    with torch.cuda.amp.autocast(enabled=False):
        B = S.size(0)
        device = S.device
        if B <= 1:
            zero = S.sum() * 0.0
            return zero, 0.0

        # Masques
        eye = torch.eye(B, device=device, dtype=torch.bool)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()) & (~eye)
        mask_neg = (labels != labels.t())

        pos_per_row = mask_pos.sum(dim=1)
        neg_per_row = mask_neg.sum(dim=1)
        valid_rows = (pos_per_row > 0) & (neg_per_row > 0)
        if valid_rows.sum() == 0:
            zero = S.sum() * 0.0
            return zero, 0.0

        # Centre les logits pour stabilité
        S_scaled = S / temp
        S_scaled = S_scaled - S_scaled.max(dim=1, keepdim=True)[0]

        losses = []
        for i in range(B):
            if not valid_rows[i]:
                continue

            pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
            neg_idx = mask_neg[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_sim = S_scaled[i][pos_idx]
            neg_sim = S_scaled[i][neg_idx]

            # Hard negatives : top-K (ratio)
            K = max(1, int(len(neg_idx) * hard_ratio))
            hard_neg_values = neg_sim.topk(K, largest=True).values

            sum_pos = torch.exp(pos_sim).sum()
            sum_hard_neg = torch.exp(hard_neg_values).sum()

            denom = sum_pos + sum_hard_neg + 1e-9
            loss_i = -torch.log(sum_pos / denom + 1e-9)
            losses.append(loss_i)

        if len(losses) == 0:
            zero = S.sum() * 0.0
            return zero, 0.0

        loss = torch.stack(losses).mean()
        valid_frac = mask_pos.sum().item() / (B * (B - 1))
        return loss, valid_frac


def unified_set_loss(
    Ehat: torch.Tensor,
    Omega: torch.Tensor,
    rel_vec: torch.Tensor,
    labels: torch.Tensor,
    temp: float = 0.1,
    hard_ratio: float = 0.5,
    lambda_local: float = 0.6,
    sim_mode: str = "mix",
    alpha: float = 0.5,
    use_wp: bool = True,
    use_zscore_if_C9: bool = False,
    zscore_kappa: float = 2.5,
    weights_source: str = "beta",
    use_ot: bool = False,
    use_cell_ot: bool = False,
    use_stripe_ot: bool = False,
    Uhat: Optional[torch.Tensor] = None,
    Gamma: Optional[torch.Tensor] = None,
    ot_epsilon: float = 0.01,
    ot_num_iters: int = 100,
    ot_margi_eps: float = 1e-9,
    use_cross_view_consistency: bool = False,
    cross_view_alpha: float = 0.7,
    cross_view_pos_lambda: float = 0.75,
    cross_view_phi_scale: float = 8.0,
    cross_view_phi_bias: float = 0.5,
    cross_view_norm_transitive: bool = True,
    use_cross_view_row_consistency: bool = False,
    cross_view_row_weight: float = 0.0,
    cross_view_row_pos_lambda: float = 0.75,
    Orient: Optional[torch.Tensor] = None,
    use_orientation_guided_ot: bool = False,
    orientation_ot_cost_weight: float = 0.05,
    orientation_ot_mass_weight: float = 1.0,
    use_view_prototype_propagation: bool = False,
    view_prototype_path: str = "",
    view_propagation_lambda: float = 0.15,
    view_prototype_temp: float = 10.0,
    use_view_prototype_span: bool = False,
    view_prototype_span_lambda: float = 1e-3,
    view_transition_self: float = 1.0,
    view_transition_neighbor1: float = 0.7,
    view_transition_neighbor2: float = 0.2,
    use_view_uncertainty_gate: bool = True,
    view_prototype_repository: Optional[ViewPrototypeRepository] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Loss unifiée combinant setNCE (soft matching) et local row/col max (permutation-robuste).
    S = λ * S_local + (1-λ) * S_set, puis InfoNCE hard.
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Similarité set-to-set avec composante relationnelle rel_vec.
        S_set = set_to_set_similarity(
            Ehat.float(), rel_vec.float(), Omega.float(),
            Ehat.float(), rel_vec.float(), Omega.float(),
            sim_mode=sim_mode, alpha=alpha, temp=temp,
                use_wp=use_wp, use_zscore=use_zscore_if_C9, zscore_kappa=zscore_kappa, weights_source=weights_source,
                use_ot=use_ot,
                use_cell_ot=use_cell_ot,
                use_stripe_ot=use_stripe_ot,
                Uq_cells=Uhat,
                Ug_cells=Uhat,
                Gamma_q=Gamma,
                Gamma_g=Gamma,
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                ot_margi_eps=ot_margi_eps,
                use_cross_view_consistency=use_cross_view_consistency,
                cross_view_alpha=cross_view_alpha,
                cross_view_pos_lambda=cross_view_pos_lambda,
                cross_view_phi_scale=cross_view_phi_scale,
                cross_view_phi_bias=cross_view_phi_bias,
                cross_view_norm_transitive=cross_view_norm_transitive,
                use_cross_view_row_consistency=use_cross_view_row_consistency,
                cross_view_row_weight=cross_view_row_weight,
                cross_view_row_pos_lambda=cross_view_row_pos_lambda,
                Orient_q=Orient,
                Orient_g=Orient,
                use_orientation_guided_ot=use_orientation_guided_ot,
                orientation_ot_cost_weight=orientation_ot_cost_weight,
                orientation_ot_mass_weight=orientation_ot_mass_weight,
                use_view_prototype_propagation=use_view_prototype_propagation,
                view_prototype_path=view_prototype_path,
                view_propagation_lambda=view_propagation_lambda,
                view_prototype_temp=view_prototype_temp,
                use_view_prototype_span=use_view_prototype_span,
                view_prototype_span_lambda=view_prototype_span_lambda,
                view_transition_self=view_transition_self,
                view_transition_neighbor1=view_transition_neighbor1,
                view_transition_neighbor2=view_transition_neighbor2,
                use_view_uncertainty_gate=use_view_uncertainty_gate,
                view_prototype_repository=view_prototype_repository,
        )
        S_set = torch.nan_to_num(S_set, nan=0.0, posinf=50.0, neginf=-50.0)

        # Similarité locale (row/col max) sur Ehat normalisé
        Eh = F.normalize(Ehat.float(), dim=-1)
        sim = torch.einsum("acd,bkd->abck", Eh, Eh).clamp(-1, 1)
        row_max = sim.max(dim=3).values.mean(dim=2)  # (B,B)
        col_max = sim.max(dim=2).values.mean(dim=2)  # (B,B)
        S_local = 0.5 * (row_max + col_max)

        # Fusion
        S = lambda_local * S_local + (1 - lambda_local) * S_set

        # InfoNCE hard mining
        loss, valid = set_contrastive_loss_hard(S, labels, temp=temp, hard_ratio=hard_ratio)
        return loss, valid


def local_rowcol_max_loss(Ehat: torch.Tensor, labels: torch.Tensor,
                          margin: float = 0.25, center_only: bool = False) -> Tuple[torch.Tensor, float]:
    """
    Contraste set-level par agrégation row/col max sur les stripes (robuste aux permutations).
    Pour deux images A,B : sim_stripes = E_A @ E_B^T (C x C), on prend max par ligne/colonne,
    on moyenne et on applique une loss contrastive marge.
    """
    if center_only:
        C = Ehat.shape[1]
        c_idx = C // 2  # si pair, prend l'indice inférieur
        center = Ehat[:, c_idx, :]  # (B,D)
        S_pair = torch.matmul(center, center.t()).clamp(-1, 1)
        return set_contrastive_loss(S_pair, labels, margin=margin)
    else:
        # Ehat est supposé normalisé
        # sim: (B, B, C, C)
        sim = torch.einsum("acd,bkd->abck", Ehat, Ehat).clamp(-1, 1)

        row_max = sim.max(dim=3).values  # (B,B,C)
        col_max = sim.max(dim=2).values  # (B,B,C)
        S_pair = 0.5 * (row_max.mean(dim=2) + col_max.mean(dim=2))  # (B,B)

        return set_contrastive_loss(S_pair, labels, margin=margin)


def structural_triplet_ot_loss(
    Ehat: torch.Tensor,
    Omega: torch.Tensor,
    labels: torch.Tensor,
    rel_vec: Optional[torch.Tensor] = None,
    margin: float = 0.3,
    sim_mode: str = "mix",
    alpha: float = 0.5,
    temp: float = 0.45,
    use_wp: bool = True,
    use_zscore_if_C9: bool = False,
    zscore_kappa: float = 2.5,
    weights_source: str = "beta",
    use_ot: bool = False,
    use_cell_ot: bool = False,
    use_stripe_ot: bool = False,
    Uhat: Optional[torch.Tensor] = None,
    Gamma: Optional[torch.Tensor] = None,
    ot_epsilon: float = 0.01,
    ot_num_iters: int = 100,
    uniform_ot_marginals: bool = False,
    ot_margi_eps: float = 1e-9,
    use_cross_view_consistency: bool = False,
    cross_view_alpha: float = 0.7,
    cross_view_pos_lambda: float = 0.75,
    cross_view_phi_scale: float = 8.0,
    cross_view_phi_bias: float = 0.5,
    cross_view_norm_transitive: bool = True,
    use_cross_view_row_consistency: bool = False,
    cross_view_row_weight: float = 0.0,
    cross_view_row_pos_lambda: float = 0.75,
    Orient: Optional[torch.Tensor] = None,
    use_orientation_guided_ot: bool = False,
    orientation_ot_cost_weight: float = 0.05,
    orientation_ot_mass_weight: float = 1.0,
    use_view_prototype_propagation: bool = False,
    view_prototype_path: str = "",
    view_propagation_lambda: float = 0.15,
    view_prototype_temp: float = 10.0,
    use_view_prototype_span: bool = False,
    view_prototype_span_lambda: float = 1e-3,
    view_transition_self: float = 1.0,
    view_transition_neighbor1: float = 0.7,
    view_transition_neighbor2: float = 0.2,
    use_view_uncertainty_gate: bool = True,
    view_prototype_repository: Optional[ViewPrototypeRepository] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Triplet loss structurelle avec distance OT implicite via set-to-set similarity.
    D(a,b)=1-S(a,b), puis hard mining: pos le plus distant, neg le plus proche.
    """
    with torch.cuda.amp.autocast(enabled=False):
        E = Ehat.float()
        O = Omega.float()
        R = rel_vec.float() if rel_vec is not None else None

        S = set_to_set_similarity(
            E, R, O,
            E, R, O,
            sim_mode=sim_mode,
            alpha=alpha,
            temp=temp,
            use_wp=use_wp,
            use_zscore=use_zscore_if_C9,
            zscore_kappa=zscore_kappa,
            rel_are_normalized=False,
            weights_source=weights_source,
            use_ot=use_ot,
            use_cell_ot=use_cell_ot,
            use_stripe_ot=use_stripe_ot,
            Uq_cells=Uhat,
            Ug_cells=Uhat,
            Gamma_q=Gamma,
            Gamma_g=Gamma,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            uniform_ot_marginals=uniform_ot_marginals,
            ot_margi_eps=ot_margi_eps,
            use_cross_view_consistency=use_cross_view_consistency,
            cross_view_alpha=cross_view_alpha,
            cross_view_pos_lambda=cross_view_pos_lambda,
            cross_view_phi_scale=cross_view_phi_scale,
            cross_view_phi_bias=cross_view_phi_bias,
            cross_view_norm_transitive=cross_view_norm_transitive,
            use_cross_view_row_consistency=use_cross_view_row_consistency,
            cross_view_row_weight=cross_view_row_weight,
            cross_view_row_pos_lambda=cross_view_row_pos_lambda,
            use_view_prototype_propagation=use_view_prototype_propagation,
            view_prototype_path=view_prototype_path,
            view_propagation_lambda=view_propagation_lambda,
            view_prototype_temp=view_prototype_temp,
            use_view_prototype_span=use_view_prototype_span,
            view_prototype_span_lambda=view_prototype_span_lambda,
            view_transition_self=view_transition_self,
            view_transition_neighbor1=view_transition_neighbor1,
            view_transition_neighbor2=view_transition_neighbor2,
            use_view_uncertainty_gate=use_view_uncertainty_gate,
            view_prototype_repository=view_prototype_repository,
        )
        S = torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        D = 1.0 - S  # plus petit = plus similaire

        B = D.size(0)
        device = D.device
        if B <= 1:
            zero = D.sum() * 0.0
            return zero, 0.0

        eye = torch.eye(B, device=device, dtype=torch.bool)
        y = labels.view(-1, 1)
        mask_pos = (y == y.t()) & (~eye)
        mask_neg = (y != y.t())

        losses = []
        valid = 0
        for i in range(B):
            pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
            neg_idx = mask_neg[i].nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            # Hard positive: plus grande distance parmi les positifs.
            d_pos = D[i, pos_idx].max()
            # Hard negative: plus petite distance parmi les négatifs.
            d_neg = D[i, neg_idx].min()
            losses.append(F.relu(d_pos - d_neg + margin))
            valid += 1

        if len(losses) == 0:
            zero = D.sum() * 0.0
            return zero, 0.0

        loss = torch.stack(losses).mean()
        return loss, float(valid) / float(B)


def structural_triplet_complex_ot_loss(
    Ehat_mod_abs: torch.Tensor,
    Ehat_mod_rel: torch.Tensor,
    Omega: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    use_stripe_ot: bool = True,
    use_wp: bool = True,
    temp: float = 0.15,
    ot_epsilon: float = 0.1,
    ot_num_iters: int = 100,
    uniform_ot_marginals: bool = False,
    ot_margi_eps: float = 1e-9,
    crg_lambda: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """
    Triplet structurelle basée sur la similarité complexe hermitienne.
    D = 1 - S_complex, puis hard positive / hard negative par anchor.
    """
    with torch.cuda.amp.autocast(enabled=False):
        from complex_embedding.hermitian import complex_set_to_set_similarity  # type: ignore

        Er = F.normalize(Ehat_mod_abs.float(), dim=-1)
        Em = F.normalize(Ehat_mod_rel.float(), dim=-1)
        O = Omega.float()

        S = complex_set_to_set_similarity(
            Er,
            Em,
            O,
            Er,
            Em,
            O,
            temp=temp,
            use_wp=use_wp,
            use_stripe_ot=use_stripe_ot,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            uniform_ot_marginals=uniform_ot_marginals,
            ot_margi_eps=ot_margi_eps,
            crg_lambda=crg_lambda,
        )
        S = torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        D = 1.0 - S

        B = D.size(0)
        device = D.device
        if B <= 1:
            zero = D.sum() * 0.0
            return zero, 0.0

        eye = torch.eye(B, device=device, dtype=torch.bool)
        y = labels.view(-1, 1)
        mask_pos = (y == y.t()) & (~eye)
        mask_neg = (y != y.t())

        losses = []
        valid = 0
        for i in range(B):
            pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
            neg_idx = mask_neg[i].nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            d_pos = D[i, pos_idx].max()
            d_neg = D[i, neg_idx].min()
            losses.append(F.relu(d_pos - d_neg + margin))
            valid += 1

        if len(losses) == 0:
            zero = D.sum() * 0.0
            return zero, 0.0

        loss = torch.stack(losses).mean()
        return loss, float(valid) / float(B)


def rel_nce_loss(R: torch.Tensor, labels: torch.Tensor, temp: float = 0.15) -> Tuple[torch.Tensor, float]:
    """
    InfoNCE (stable) sur les signatures relationnelles flattenées.
    - FP32 pour la stabilité
    - Masquage diagonal
    - Soustraction du max par ligne
    - ?psilon pour éviter log(0)
    """
    with torch.cuda.amp.autocast(enabled=False):
        B = R.size(0)
        device = R.device
        feat = R.reshape(B, -1).float()
        feat = F.normalize(feat, dim=1)
        S = torch.matmul(feat, feat.t()) / temp  # échelle par température

        mask_pos = (labels.view(-1, 1) == labels.view(1, -1)).float()
        eye = torch.eye(B, device=device)
        mask_pos = mask_pos * (1 - eye)

        num_pos = mask_pos.sum()
        if num_pos < 1:
            zero = S.sum() * 0.0
            return zero, 0.0

        logits = S - S.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits).float()

        numerator = (exp_logits * mask_pos).sum(dim=1)
        denominator = exp_logits.sum(dim=1) - exp_logits.diag()

        eps = 1e-9
        loss = -torch.log(numerator / (denominator + eps) + eps)

        valid_mask = (numerator > 0)
        if valid_mask.sum() < 1:
            zero = S.sum() * 0.0
            return zero, 0.0

        loss = loss[valid_mask].mean()
        valid_frac = num_pos.item() / (B * (B - 1))
        return loss, valid_frac


def attach_loss(Ehat: torch.Tensor, global_feat: torch.Tensor,
                beta: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    Perte d'attachement: force les stripes à être cohérentes avec le global.
    
    Args:
        Ehat: (B, C, D) - embeddings des stripes
        global_feat: (B, D) - feature globale
        beta: (B, C) - poids de fiabilité
        temp: Température pour la similarité
    """
    B, C, D = Ehat.shape
    
    # Expansion du global pour matcher les stripes
    global_expanded = global_feat.unsqueeze(1).expand(-1, C, -1)  # (B, C, D)
    
    # Similarité cosinus
    sim = F.cosine_similarity(Ehat, global_expanded, dim=2)  # (B, C)
    
    # Perte: on veut maximiser la similarité
    loss = 1.0 - sim  # (B, C)
    
    # Pondération par beta
    weighted_loss = (loss * beta).sum() / (beta.sum() + 1e-9)
    
    return weighted_loss


def diversity_loss(Ehat: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Perte de diversité: encourage les stripes à être différentes entre elles.
    
    Args:
        Ehat: (B, C, D)
        beta: (B, C)
    """
    B, C, D = Ehat.shape
    
    # Matrice de similarité intra-image
    sim = torch.einsum("bcd,bkd->bck", Ehat, Ehat)  # (B, C, C)
    
    # Masque diagonal
    mask = torch.eye(C, device=Ehat.device).unsqueeze(0)
    sim = sim * (1 - mask)
    
    # Poids pour chaque paire de stripes
    w = beta.unsqueeze(2) * beta.unsqueeze(1) * (1 - mask)  # (B, C, C)
    
    # Pénalité sur les similarités élevées (hors diagonal)
    loss = (sim.pow(2) * w).sum() / (w.sum() + 1e-9)
    
    return loss


def augmentation_consistency_loss(E1: torch.Tensor, R1: torch.Tensor, O1: torch.Tensor,
                                  E2: torch.Tensor, R2: torch.Tensor, O2: torch.Tensor,
                                  labels: torch.Tensor, temp: float = 0.15, weights_source: str = "beta",
                                  use_ot: bool = False,
                                  use_cell_ot: bool = False,
                                  use_stripe_ot: bool = False,
                                  U1: Optional[torch.Tensor] = None, U2: Optional[torch.Tensor] = None,
                                  Gamma1: Optional[torch.Tensor] = None, Gamma2: Optional[torch.Tensor] = None,
                                  ot_epsilon: float = 0.01, ot_num_iters: int = 100,
                                  ot_margi_eps: float = 1e-9,
                                  use_cross_view_consistency: bool = False,
                                  cross_view_alpha: float = 0.7,
                                  cross_view_pos_lambda: float = 0.75,
                                  cross_view_phi_scale: float = 8.0,
                                  cross_view_phi_bias: float = 0.5,
                                  cross_view_norm_transitive: bool = True,
                                  use_cross_view_row_consistency: bool = False,
                                  cross_view_row_weight: float = 0.0,
                                  cross_view_row_pos_lambda: float = 0.75,
                                  use_view_prototype_propagation: bool = False,
                                  view_prototype_path: str = "",
                                  view_propagation_lambda: float = 0.15,
                                  view_prototype_temp: float = 10.0,
                                  use_view_prototype_span: bool = False,
                                  view_prototype_span_lambda: float = 1e-3,
                                  view_transition_self: float = 1.0,
                                  view_transition_neighbor1: float = 0.7,
                                  view_transition_neighbor2: float = 0.2,
                                  use_view_uncertainty_gate: bool = True,
                                  view_prototype_repository: Optional[ViewPrototypeRepository] = None) -> torch.Tensor:
    """
    Perte de cohérence entre deux vues augmentées (set-to-set, robuste aux permutions de stripes).
    On matche l'image originale (E1) avec sa vue augmentée (E2) via set_to_set_similarity,
    sans supposer l'alignement c<->c.
    """
    # Similarité set-level entre original et vue augmentée
    S = set_to_set_similarity(
        E1, R1, O1,
        E2, R2, O2,
        sim_mode="mix",
        alpha=0.5,
        temp=temp,              # température pour les softmax de matching
        use_wp=True,
        use_zscore=False,
        zscore_kappa=2.5,
        weights_source=weights_source,
        use_ot=use_ot,
        use_cell_ot=use_cell_ot,
        use_stripe_ot=use_stripe_ot,
        Uq_cells=U1,
        Ug_cells=U2,
        Gamma_q=Gamma1,
        Gamma_g=Gamma2,
        ot_epsilon=ot_epsilon,
        ot_num_iters=ot_num_iters,
        ot_margi_eps=ot_margi_eps,
        use_cross_view_consistency=use_cross_view_consistency,
        cross_view_alpha=cross_view_alpha,
        cross_view_pos_lambda=cross_view_pos_lambda,
        cross_view_phi_scale=cross_view_phi_scale,
        cross_view_phi_bias=cross_view_phi_bias,
        cross_view_norm_transitive=cross_view_norm_transitive,
        use_cross_view_row_consistency=use_cross_view_row_consistency,
        cross_view_row_weight=cross_view_row_weight,
        cross_view_row_pos_lambda=cross_view_row_pos_lambda,
        use_view_prototype_propagation=use_view_prototype_propagation,
        view_prototype_path=view_prototype_path,
        view_propagation_lambda=view_propagation_lambda,
        view_prototype_temp=view_prototype_temp,
        use_view_prototype_span=use_view_prototype_span,
        view_prototype_span_lambda=view_prototype_span_lambda,
        view_transition_self=view_transition_self,
        view_transition_neighbor1=view_transition_neighbor1,
        view_transition_neighbor2=view_transition_neighbor2,
        use_view_uncertainty_gate=use_view_uncertainty_gate,
        view_prototype_repository=view_prototype_repository,
    )

    # On force le positif uniquement sur la paire (même index dans le batch)
    B = S.size(0)
    device = S.device
    labels = labels.to(device=device).view(-1)
    if labels.numel() != B:
        raise ValueError(f"augmentation_consistency_loss labels shape {tuple(labels.shape)} does not match batch {B}")

    same_id = labels[:, None].eq(labels[None, :])
    pos_vals = S[same_id]
    neg_vals = S[~same_id]

    zero = S.sum() * 0.0
    pos_loss = ((1.0 - pos_vals) ** 2).mean() if pos_vals.numel() > 0 else zero
    neg_loss = (F.relu(neg_vals - temp) ** 2).mean() if neg_vals.numel() > 0 else zero
    return pos_loss + neg_loss


class DOCLoss(nn.Module):
    """
    Wrapper pour calculer toutes les pertes DOC en une passe.
    """
    
    def __init__(
        self,
        cfg,
        num_classes: int,
        center_feat_dim: Optional[int] = None,
        center_g_dim: Optional[int] = None,
        view_prototype_repository: Optional[ViewPrototypeRepository] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.view_prototype_repository = view_prototype_repository or ViewPrototypeRepository()
        self.center_loss: Optional[CenterLoss] = None
        self.center_loss_g: Optional[CenterLoss] = None
        
        # Critères de base
        self.criterion_id = CrossEntropyLabelSmooth(epsilon=cfg.label_smoothing)
        
        # TripletLoss locale (évite dépendance torchreid)
        self.criterion_tri = TripletLoss(margin=cfg.triplet_margin) if cfg.use_L_tri_g else None

        # Center loss optionnelle
        if cfg.use_L_center and center_feat_dim is not None:
            self.center_loss = CenterLoss(num_classes, center_feat_dim)
        if cfg.use_L_center_g and center_g_dim is not None:
            self.center_loss_g = CenterLoss(num_classes, center_g_dim)
    
    def forward(self, outputs: dict, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Calcule toutes les pertes activées.
        
        Args:
            outputs: Dict contenant les tensors du forward
            labels: (B,) labels des identités
            
        Returns:
            total_loss: Perte totale pondérée
            loss_dict: Dictionnaire des pertes individuelles pour logging
        """
        cfg = self.cfg
        device = labels.device
        stripe_only_mode = bool(getattr(cfg, "stripe_only_mode", False))
        # Configuration OT (cell-level and stripe-level explicit flags)
        cfg_cell_ot = bool(getattr(cfg, "use_cell_ot_matching", False))
        cfg_stripe_ot_raw = getattr(cfg, "use_stripe_ot_matching", None)
        use_cell_ot, use_stripe_ot = resolve_ot_modes(cfg_cell_ot, cfg_stripe_ot_raw)
        use_ot = use_cell_ot or use_stripe_ot
        ot_epsilon = float(getattr(cfg, "ot_epsilon", 0.01))
        ot_num_iters = int(getattr(cfg, "ot_num_iters", 100))
        ot_margi_eps = float(getattr(cfg, "ot_margi_eps", 1e-9))
        use_cross_view_consistency = bool(getattr(cfg, "use_cross_view_consistency", False))
        cross_view_alpha = float(getattr(cfg, "cross_view_alpha", 0.7))
        cross_view_pos_lambda = float(getattr(cfg, "cross_view_pos_lambda", 0.75))
        cross_view_phi_scale = float(getattr(cfg, "cross_view_phi_scale", 8.0))
        cross_view_phi_bias = float(getattr(cfg, "cross_view_phi_bias", 0.5))
        cross_view_norm_transitive = bool(getattr(cfg, "cross_view_norm_transitive", True))
        use_cross_view_row_consistency = bool(getattr(cfg, "use_cross_view_row_consistency", False))
        cross_view_row_weight = float(getattr(cfg, "cross_view_row_weight", 0.0))
        cross_view_row_pos_lambda = float(getattr(cfg, "cross_view_row_pos_lambda", 0.75))
        use_view_prototype_propagation = bool(getattr(cfg, "use_view_prototype_propagation", False))
        view_prototype_path = str(getattr(cfg, "view_prototype_path", ""))
        view_propagation_lambda = float(getattr(cfg, "view_propagation_lambda", 0.15))
        view_prototype_temp = float(getattr(cfg, "view_prototype_temp", 10.0))
        use_view_prototype_span = bool(getattr(cfg, "use_view_prototype_span", False))
        view_prototype_span_lambda = float(getattr(cfg, "view_prototype_span_lambda", 1e-3))
        view_transition_self = float(getattr(cfg, "view_transition_self", 1.0))
        view_transition_neighbor1 = float(getattr(cfg, "view_transition_neighbor1", 0.7))
        view_transition_neighbor2 = float(getattr(cfg, "view_transition_neighbor2", 0.2))
        use_view_uncertainty_gate = bool(getattr(cfg, "use_view_uncertainty_gate", True))
        use_orientation_guided_ot = bool(getattr(cfg, "use_orientation_guided_ot", False))
        orientation_ot_cost_weight = float(getattr(cfg, "orientation_ot_cost_weight", 0.05))
        orientation_ot_mass_weight = float(getattr(cfg, "orientation_ot_mass_weight", 1.0))
        crg_lambda = float(getattr(cfg, "crg_lambda", 0.5))

        # Initialise avec un scalaire différentiable pour éviter des pertes sans grad_fn
        total_loss = torch.zeros((), device=device)
        loss_dict = {}

        if use_cross_view_consistency and use_stripe_ot and "Ehat" in outputs:
            cv_stats = cross_view_similarity_stats(
                outputs["Ehat"],
                outputs["Ehat"],
                cross_view_alpha=cross_view_alpha,
                cross_view_pos_lambda=cross_view_pos_lambda,
                cross_view_phi_scale=cross_view_phi_scale,
                cross_view_phi_bias=cross_view_phi_bias,
                cross_view_norm_transitive=cross_view_norm_transitive,
            )
            if "Ehat2" in outputs:
                cv_stats2 = cross_view_similarity_stats(
                    outputs["Ehat2"],
                    outputs["Ehat2"],
                    cross_view_alpha=cross_view_alpha,
                    cross_view_pos_lambda=cross_view_pos_lambda,
                    cross_view_phi_scale=cross_view_phi_scale,
                    cross_view_phi_bias=cross_view_phi_bias,
                    cross_view_norm_transitive=cross_view_norm_transitive,
                )
                for k in cv_stats.keys():
                    cv_stats[k] = 0.5 * (cv_stats[k] + cv_stats2[k])
            loss_dict.update(cv_stats)
        
        # === Pertes globales ===
        if (not stripe_only_mode) and cfg.use_L_ID_g and "logits" in outputs:
            lid = self.criterion_id(outputs["logits"], labels)
            
            # Moyenne si deux vues
            if "logits2" in outputs:
                lid = 0.5 * (lid + self.criterion_id(outputs["logits2"], labels))
            
            weighted = cfg.w_ID_g * lid
            total_loss = total_loss + weighted
            loss_dict["L_ID_g"] = lid.item()
            loss_dict["L_ID_g_w"] = weighted.item()
        
        if (not stripe_only_mode) and cfg.use_L_tri_g and "global_feat" in outputs and self.criterion_tri:
            ltri = self.criterion_tri(outputs["global_feat"].float(), labels)
            
            if "global_feat2" in outputs:
                ltri = 0.5 * (ltri + self.criterion_tri(outputs["global_feat2"].float(), labels))
            
            weighted = cfg.w_tri_g * ltri
            total_loss = total_loss + weighted
            loss_dict["L_tri_g"] = ltri.item()
            loss_dict["L_tri_g_w"] = weighted.item()
        
        # === Pertes sur stripes ===
        if cfg.use_L_ID_s and "Ehat" in outputs:
            # Classification par stripe
            stripe_cls = outputs.get("stripe_classifier")
            if stripe_cls is None:
                stripe_cls = getattr(outputs.get("backbone", {}), "classifier", None)
            if stripe_cls is None:
                # Fallback: utiliser le classifier global
                stripe_cls = outputs.get("classifier")
            
            if stripe_cls is not None:
                B, C, D = outputs["Ehat"].shape
                feats_s = F.normalize(outputs["Ehat"].float(), dim=-1)
                # Si la tête a son propre scale (StripeHead), on n'en ajoute pas
                logits_s = stripe_cls(feats_s.view(-1, D))
                labels_s = labels.repeat_interleave(C)
                
                lid_s = self.criterion_id(logits_s, labels_s)
                
                if "Ehat2" in outputs:
                    B2, C2, D2 = outputs["Ehat2"].shape
                    feats_s2 = F.normalize(outputs["Ehat2"].float(), dim=-1)
                    logits_s2 = stripe_cls(feats_s2.view(-1, D2))
                    lid_s = 0.5 * (lid_s + self.criterion_id(logits_s2, labels.repeat_interleave(C2)))
                
                weighted = cfg.w_ID_s * lid_s
                total_loss = total_loss + weighted
                loss_dict["L_ID_s"] = lid_s.item()
                loss_dict["L_ID_s_w"] = weighted.item()

        if getattr(cfg, "use_L_ID_mod_abs", False) and "Ehat_mod_abs" in outputs:
            stripe_cls = outputs.get("stripe_classifier")
            if stripe_cls is None:
                stripe_cls = outputs.get("classifier")
            if stripe_cls is not None:
                B, C, D = outputs["Ehat_mod_abs"].shape
                logits_mod_abs = stripe_cls(F.normalize(outputs["Ehat_mod_abs"].float(), dim=-1).view(-1, D))
                labels_s = labels.repeat_interleave(C)
                lid_mod_abs = self.criterion_id(logits_mod_abs, labels_s)
                if "Ehat_mod_abs2" in outputs:
                    B2, C2, D2 = outputs["Ehat_mod_abs2"].shape
                    logits_mod_abs2 = stripe_cls(F.normalize(outputs["Ehat_mod_abs2"].float(), dim=-1).view(-1, D2))
                    lid_mod_abs = 0.5 * (lid_mod_abs + self.criterion_id(logits_mod_abs2, labels.repeat_interleave(C2)))
                weighted = float(getattr(cfg, "w_ID_mod_abs", 0.3)) * lid_mod_abs
                total_loss = total_loss + weighted
                loss_dict["L_ID_mod_abs"] = lid_mod_abs.item()
                loss_dict["L_ID_mod_abs_w"] = weighted.item()

        if getattr(cfg, "use_L_ID_mod_rel", False) and "Ehat_mod_rel" in outputs:
            stripe_cls = outputs.get("stripe_classifier")
            if stripe_cls is None:
                stripe_cls = outputs.get("classifier")
            if stripe_cls is not None:
                B, C, D = outputs["Ehat_mod_rel"].shape
                logits_mod_rel = stripe_cls(F.normalize(outputs["Ehat_mod_rel"].float(), dim=-1).view(-1, D))
                labels_s = labels.repeat_interleave(C)
                lid_mod_rel = self.criterion_id(logits_mod_rel, labels_s)
                if "Ehat_mod_rel2" in outputs:
                    B2, C2, D2 = outputs["Ehat_mod_rel2"].shape
                    logits_mod_rel2 = stripe_cls(F.normalize(outputs["Ehat_mod_rel2"].float(), dim=-1).view(-1, D2))
                    lid_mod_rel = 0.5 * (lid_mod_rel + self.criterion_id(logits_mod_rel2, labels.repeat_interleave(C2)))
                weighted = float(getattr(cfg, "w_ID_mod_rel", 0.3)) * lid_mod_rel
                total_loss = total_loss + weighted
                loss_dict["L_ID_mod_rel"] = lid_mod_rel.item()
                loss_dict["L_ID_mod_rel_w"] = weighted.item()

        if cfg.use_L_tri_s_ot and cfg.w_tri_s_ot > 0 and "Ehat" in outputs and "Omega" in outputs:
            weights_source = "gamma" if getattr(cfg, "use_gamma_weights_for_matching", False) else "beta"
            ltri_s_ot, valid_s_ot = structural_triplet_ot_loss(
                Ehat=outputs["Ehat"],
                Omega=outputs["Omega"],
                labels=labels,
                rel_vec=outputs.get("rel_vec", None),
                margin=cfg.triplet_margin,
                sim_mode=getattr(cfg, "sim_mode", "mix"),
                alpha=getattr(cfg, "alpha_mix", 0.5),
                temp=getattr(cfg, "set_match_temp", 0.45),
                use_wp=cfg.use_wp_in_agg,
                use_zscore_if_C9=cfg.use_zscore_if_C9,
                zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                weights_source=weights_source,
                        use_ot=use_ot,
                        use_cell_ot=use_cell_ot,
                        use_stripe_ot=use_stripe_ot,
                    Uhat=outputs.get("Uhat"),
                    Gamma=outputs.get("Gamma"),
                    ot_epsilon=ot_epsilon,
                        ot_num_iters=ot_num_iters,
                    uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                        ot_margi_eps=ot_margi_eps,
                    use_cross_view_consistency=use_cross_view_consistency,
                    cross_view_alpha=cross_view_alpha,
                    cross_view_pos_lambda=cross_view_pos_lambda,
                    cross_view_phi_scale=cross_view_phi_scale,
                    cross_view_phi_bias=cross_view_phi_bias,
                    cross_view_norm_transitive=cross_view_norm_transitive,
                    use_cross_view_row_consistency=use_cross_view_row_consistency,
                    cross_view_row_weight=cross_view_row_weight,
                    cross_view_row_pos_lambda=cross_view_row_pos_lambda,
                    Orient=outputs.get("Orient"),
                    use_orientation_guided_ot=use_orientation_guided_ot,
                    orientation_ot_cost_weight=orientation_ot_cost_weight,
                    orientation_ot_mass_weight=orientation_ot_mass_weight,
                    use_view_prototype_propagation=use_view_prototype_propagation,
                    view_prototype_path=view_prototype_path,
                    view_propagation_lambda=view_propagation_lambda,
                    view_prototype_temp=view_prototype_temp,
                    use_view_prototype_span=use_view_prototype_span,
                    view_prototype_span_lambda=view_prototype_span_lambda,
                    view_transition_self=view_transition_self,
                    view_transition_neighbor1=view_transition_neighbor1,
                    view_transition_neighbor2=view_transition_neighbor2,
                    use_view_uncertainty_gate=use_view_uncertainty_gate,
                    view_prototype_repository=self.view_prototype_repository,
            )

            if "Ehat2" in outputs and "Omega2" in outputs:
                ltri_s_ot2, valid_s_ot2 = structural_triplet_ot_loss(
                    Ehat=outputs["Ehat2"],
                    Omega=outputs["Omega2"],
                    labels=labels,
                    rel_vec=outputs.get("rel_vec2", None),
                    margin=cfg.triplet_margin,
                    sim_mode=getattr(cfg, "sim_mode", "mix"),
                    alpha=getattr(cfg, "alpha_mix", 0.5),
                    temp=getattr(cfg, "set_match_temp", 0.45),
                    use_wp=cfg.use_wp_in_agg,
                    use_zscore_if_C9=cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                    weights_source=weights_source,
                        use_ot=use_ot,
                        use_cell_ot=use_cell_ot,
                        use_stripe_ot=use_stripe_ot,
                        Uhat=outputs.get("Uhat2"),
                        Gamma=outputs.get("Gamma2"),
                        ot_epsilon=ot_epsilon,
                        ot_num_iters=ot_num_iters,
                        uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                        ot_margi_eps=ot_margi_eps,
                        Orient=outputs.get("Orient2"),
                        use_orientation_guided_ot=use_orientation_guided_ot,
                        orientation_ot_cost_weight=orientation_ot_cost_weight,
                        orientation_ot_mass_weight=orientation_ot_mass_weight,
                        use_cross_view_consistency=use_cross_view_consistency,
                        cross_view_alpha=cross_view_alpha,
                        cross_view_pos_lambda=cross_view_pos_lambda,
                        cross_view_phi_scale=cross_view_phi_scale,
                        cross_view_phi_bias=cross_view_phi_bias,
                        cross_view_norm_transitive=cross_view_norm_transitive,
                        use_cross_view_row_consistency=use_cross_view_row_consistency,
                        cross_view_row_weight=cross_view_row_weight,
                        cross_view_row_pos_lambda=cross_view_row_pos_lambda,
                        use_view_prototype_propagation=use_view_prototype_propagation,
                        view_prototype_path=view_prototype_path,
                        view_propagation_lambda=view_propagation_lambda,
                        view_prototype_temp=view_prototype_temp,
                        use_view_prototype_span=use_view_prototype_span,
                        view_prototype_span_lambda=view_prototype_span_lambda,
                        view_transition_self=view_transition_self,
                        view_transition_neighbor1=view_transition_neighbor1,
                        view_transition_neighbor2=view_transition_neighbor2,
                        use_view_uncertainty_gate=use_view_uncertainty_gate,
                        view_prototype_repository=self.view_prototype_repository,
                )
                ltri_s_ot = 0.5 * (ltri_s_ot + ltri_s_ot2)
                valid_s_ot = 0.5 * (valid_s_ot + valid_s_ot2)

            weighted = cfg.w_tri_s_ot * ltri_s_ot
            total_loss = total_loss + weighted
            loss_dict["L_tri_s_ot"] = ltri_s_ot.item()
            loss_dict["L_tri_s_ot_w"] = weighted.item()
            loss_dict["tri_s_ot_valid"] = valid_s_ot

        if getattr(cfg, "use_L_tri_mod_abs", False) and float(getattr(cfg, "w_tri_mod_abs", 0.0)) > 0 and "Ehat_mod_abs" in outputs and "Omega" in outputs:
            weights_source = "gamma" if getattr(cfg, "use_gamma_weights_for_matching", False) else "beta"
            ltri_mod_abs, valid_mod_abs = structural_triplet_ot_loss(
                Ehat=outputs["Ehat_mod_abs"],
                Omega=outputs["Omega"],
                labels=labels,
                rel_vec=None,
                margin=cfg.triplet_margin,
                sim_mode="app",
                alpha=getattr(cfg, "alpha_mix", 0.5),
                temp=getattr(cfg, "set_match_temp", 0.45),
                use_wp=cfg.use_wp_in_agg,
                use_zscore_if_C9=cfg.use_zscore_if_C9,
                zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                weights_source=weights_source,
                use_ot=use_ot,
                use_cell_ot=use_cell_ot,
                use_stripe_ot=use_stripe_ot,
                Uhat=outputs.get("Uhat"),
                Gamma=outputs.get("Gamma"),
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                ot_margi_eps=ot_margi_eps,
                Orient=outputs.get("Orient"),
                use_orientation_guided_ot=use_orientation_guided_ot,
                orientation_ot_cost_weight=orientation_ot_cost_weight,
                orientation_ot_mass_weight=orientation_ot_mass_weight,
            )
            if "Ehat_mod_abs2" in outputs and "Omega2" in outputs:
                ltri_mod_abs2, valid_mod_abs2 = structural_triplet_ot_loss(
                    Ehat=outputs["Ehat_mod_abs2"],
                    Omega=outputs["Omega2"],
                    labels=labels,
                    rel_vec=None,
                    margin=cfg.triplet_margin,
                    sim_mode="app",
                    alpha=getattr(cfg, "alpha_mix", 0.5),
                    temp=getattr(cfg, "set_match_temp", 0.45),
                    use_wp=cfg.use_wp_in_agg,
                    use_zscore_if_C9=cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                    weights_source=weights_source,
                    use_ot=use_ot,
                    use_cell_ot=use_cell_ot,
                    use_stripe_ot=use_stripe_ot,
                    Uhat=outputs.get("Uhat2"),
                    Gamma=outputs.get("Gamma2"),
                    ot_epsilon=ot_epsilon,
                    ot_num_iters=ot_num_iters,
                    uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                    ot_margi_eps=ot_margi_eps,
                    Orient=outputs.get("Orient2"),
                    use_orientation_guided_ot=use_orientation_guided_ot,
                    orientation_ot_cost_weight=orientation_ot_cost_weight,
                    orientation_ot_mass_weight=orientation_ot_mass_weight,
                )
                ltri_mod_abs = 0.5 * (ltri_mod_abs + ltri_mod_abs2)
                valid_mod_abs = 0.5 * (valid_mod_abs + valid_mod_abs2)
            weighted = float(getattr(cfg, "w_tri_mod_abs", 0.4)) * ltri_mod_abs
            total_loss = total_loss + weighted
            loss_dict["L_tri_mod_abs"] = ltri_mod_abs.item()
            loss_dict["L_tri_mod_abs_w"] = weighted.item()
            loss_dict["tri_mod_abs_valid"] = valid_mod_abs

        if getattr(cfg, "use_L_tri_mod_rel", False) and float(getattr(cfg, "w_tri_mod_rel", 0.0)) > 0 and "Ehat_mod_rel" in outputs and "Omega" in outputs:
            weights_source = "gamma" if getattr(cfg, "use_gamma_weights_for_matching", False) else "beta"
            ltri_mod_rel, valid_mod_rel = structural_triplet_ot_loss(
                Ehat=outputs["Ehat_mod_rel"],
                Omega=outputs["Omega"],
                labels=labels,
                rel_vec=None,
                margin=cfg.triplet_margin,
                sim_mode="app",
                alpha=getattr(cfg, "alpha_mix", 0.5),
                temp=getattr(cfg, "set_match_temp", 0.45),
                use_wp=cfg.use_wp_in_agg,
                use_zscore_if_C9=cfg.use_zscore_if_C9,
                zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                weights_source=weights_source,
                use_ot=use_ot,
                use_cell_ot=use_cell_ot,
                use_stripe_ot=use_stripe_ot,
                Uhat=outputs.get("Uhat"),
                Gamma=outputs.get("Gamma"),
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                ot_margi_eps=ot_margi_eps,
                Orient=outputs.get("Orient"),
                use_orientation_guided_ot=use_orientation_guided_ot,
                orientation_ot_cost_weight=orientation_ot_cost_weight,
                orientation_ot_mass_weight=orientation_ot_mass_weight,
            )
            if "Ehat_mod_rel2" in outputs and "Omega2" in outputs:
                ltri_mod_rel2, valid_mod_rel2 = structural_triplet_ot_loss(
                    Ehat=outputs["Ehat_mod_rel2"],
                    Omega=outputs["Omega2"],
                    labels=labels,
                    rel_vec=None,
                    margin=cfg.triplet_margin,
                    sim_mode="app",
                    alpha=getattr(cfg, "alpha_mix", 0.5),
                    temp=getattr(cfg, "set_match_temp", 0.45),
                    use_wp=cfg.use_wp_in_agg,
                    use_zscore_if_C9=cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                    weights_source=weights_source,
                    use_ot=use_ot,
                    use_cell_ot=use_cell_ot,
                    use_stripe_ot=use_stripe_ot,
                    Uhat=outputs.get("Uhat2"),
                    Gamma=outputs.get("Gamma2"),
                    ot_epsilon=ot_epsilon,
                    ot_num_iters=ot_num_iters,
                    uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                    ot_margi_eps=ot_margi_eps,
                    Orient=outputs.get("Orient2"),
                    use_orientation_guided_ot=use_orientation_guided_ot,
                    orientation_ot_cost_weight=orientation_ot_cost_weight,
                    orientation_ot_mass_weight=orientation_ot_mass_weight,
                )
                ltri_mod_rel = 0.5 * (ltri_mod_rel + ltri_mod_rel2)
                valid_mod_rel = 0.5 * (valid_mod_rel + valid_mod_rel2)
            weighted = float(getattr(cfg, "w_tri_mod_rel", 0.2)) * ltri_mod_rel
            total_loss = total_loss + weighted
            loss_dict["L_tri_mod_rel"] = ltri_mod_rel.item()
            loss_dict["L_tri_mod_rel_w"] = weighted.item()
            loss_dict["tri_mod_rel_valid"] = valid_mod_rel

        if getattr(cfg, "use_L_tri_complex", False) and float(getattr(cfg, "w_tri_complex", 0.0)) > 0 and "Ehat_mod_abs" in outputs and "Ehat_mod_rel" in outputs and "Omega" in outputs:
            ltri_c, valid_c = structural_triplet_complex_ot_loss(
                Ehat_mod_abs=outputs["Ehat_mod_abs"],
                Ehat_mod_rel=outputs["Ehat_mod_rel"],
                Omega=outputs["Omega"],
                labels=labels,
                margin=cfg.triplet_margin,
                use_stripe_ot=use_stripe_ot or use_ot,
                use_wp=cfg.use_wp_in_agg,
                temp=getattr(cfg, "set_match_temp", 0.45),
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                ot_margi_eps=ot_margi_eps,
                crg_lambda=crg_lambda,
            )
            if "Ehat_mod_abs2" in outputs and "Ehat_mod_rel2" in outputs and "Omega2" in outputs:
                ltri_c2, valid_c2 = structural_triplet_complex_ot_loss(
                    Ehat_mod_abs=outputs["Ehat_mod_abs2"],
                    Ehat_mod_rel=outputs["Ehat_mod_rel2"],
                    Omega=outputs["Omega2"],
                    labels=labels,
                    margin=cfg.triplet_margin,
                    use_stripe_ot=use_stripe_ot or use_ot,
                    use_wp=cfg.use_wp_in_agg,
                    temp=getattr(cfg, "set_match_temp", 0.45),
                    ot_epsilon=ot_epsilon,
                    ot_num_iters=ot_num_iters,
                    uniform_ot_marginals=getattr(cfg, "use_uniform_ot_marginals", False),
                    ot_margi_eps=ot_margi_eps,
                    crg_lambda=crg_lambda,
                )
                ltri_c = 0.5 * (ltri_c + ltri_c2)
                valid_c = 0.5 * (valid_c + valid_c2)
            weighted = float(getattr(cfg, "w_tri_complex", 0.4)) * ltri_c
            total_loss = total_loss + weighted
            loss_dict["L_tri_complex"] = ltri_c.item()
            loss_dict["L_tri_complex_w"] = weighted.item()
            loss_dict["tri_complex_valid"] = valid_c
        
        if cfg.use_L_setNCE and cfg.w_setNCE > 0 and "Ehat" in outputs and "Omega" in outputs and "rel_vec" in outputs:
            weights_source = "gamma" if getattr(cfg, "use_gamma_weights_for_matching", False) else "beta"
            if getattr(cfg, "setnce_use_unified", True):
                lset, valid_frac = unified_set_loss(
                    outputs["Ehat"], outputs["Omega"], outputs["rel_vec"], labels,
                    temp=cfg.set_nce_temp, hard_ratio=0.7,
                    lambda_local=getattr(cfg, "setnce_lambda_local", 0.6),
                    sim_mode=getattr(cfg, "sim_mode", "mix"),
                    alpha=getattr(cfg, "alpha_mix", 0.5),
                    use_wp=cfg.use_wp_in_agg,
                    use_zscore_if_C9=cfg.use_zscore_if_C9,
                    zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                    weights_source=weights_source,
                    use_ot=use_ot,
                    use_cell_ot=use_cell_ot,
                    use_stripe_ot=use_stripe_ot,
                    Uhat=outputs.get("Uhat"),
                    Gamma=outputs.get("Gamma"),
                    ot_epsilon=ot_epsilon,
                    ot_num_iters=ot_num_iters,
                    ot_margi_eps=ot_margi_eps,
                    use_cross_view_consistency=use_cross_view_consistency,
                    cross_view_alpha=cross_view_alpha,
                    cross_view_pos_lambda=cross_view_pos_lambda,
                    cross_view_phi_scale=cross_view_phi_scale,
                    cross_view_phi_bias=cross_view_phi_bias,
                    cross_view_norm_transitive=cross_view_norm_transitive,
                    use_cross_view_row_consistency=use_cross_view_row_consistency,
                    cross_view_row_weight=cross_view_row_weight,
                    cross_view_row_pos_lambda=cross_view_row_pos_lambda,
                    Orient=outputs.get("Orient"),
                    use_orientation_guided_ot=use_orientation_guided_ot,
                    orientation_ot_cost_weight=orientation_ot_cost_weight,
                    orientation_ot_mass_weight=orientation_ot_mass_weight,
                    use_view_prototype_propagation=use_view_prototype_propagation,
                    view_prototype_path=view_prototype_path,
                    view_propagation_lambda=view_propagation_lambda,
                    view_prototype_temp=view_prototype_temp,
                    use_view_prototype_span=use_view_prototype_span,
                    view_prototype_span_lambda=view_prototype_span_lambda,
                    view_transition_self=view_transition_self,
                    view_transition_neighbor1=view_transition_neighbor1,
                    view_transition_neighbor2=view_transition_neighbor2,
                    use_view_uncertainty_gate=use_view_uncertainty_gate,
                    view_prototype_repository=self.view_prototype_repository,
                )
            else:
                # Set-only path: no local row/col fusion.
                S_set = set_to_set_similarity(
                    outputs["Ehat"].float(), outputs["rel_vec"].float(), outputs["Omega"].float(),
                    outputs["Ehat"].float(), outputs["rel_vec"].float(), outputs["Omega"].float(),
                    sim_mode=getattr(cfg, "sim_mode", "mix"), alpha=getattr(cfg, "alpha_mix", 0.5), temp=cfg.set_nce_temp,
                    use_wp=cfg.use_wp_in_agg, use_zscore=cfg.use_zscore_if_C9, zscore_kappa=getattr(cfg, "zscore_kappa", 2.5),
                    weights_source=weights_source,
                    use_view_prototype_propagation=use_view_prototype_propagation,
                    view_prototype_path=view_prototype_path,
                    view_propagation_lambda=view_propagation_lambda,
                    view_prototype_temp=view_prototype_temp,
                    use_view_prototype_span=use_view_prototype_span,
                    view_prototype_span_lambda=view_prototype_span_lambda,
                    view_transition_self=view_transition_self,
                    view_transition_neighbor1=view_transition_neighbor1,
                    view_transition_neighbor2=view_transition_neighbor2,
                    use_view_uncertainty_gate=use_view_uncertainty_gate,
                    view_prototype_repository=self.view_prototype_repository,
                )
                S_set = torch.nan_to_num(S_set, nan=0.0, posinf=50.0, neginf=-50.0)
                lset, valid_frac = set_contrastive_loss_hard(
                    S_set, labels, temp=cfg.set_nce_temp, hard_ratio=0.7
                )

            weighted = cfg.w_setNCE * lset
            total_loss = total_loss + weighted
            loss_dict["L_set_contrastive"] = lset.item()
            loss_dict["L_set_contrastive_w"] = weighted.item()
            loss_dict["set_contrastive_valid"] = valid_frac

        if cfg.use_L_local_match and cfg.w_local_match > 0 and "Ehat" in outputs:
            llocal, valid_local = local_rowcol_max_loss(
                F.normalize(outputs["Ehat"].float(), dim=-1),
                labels,
                margin=cfg.local_match_margin,
                center_only=cfg.local_match_center_only,
            )

            if "Ehat2" in outputs:
                llocal2, valid_local2 = local_rowcol_max_loss(
                    F.normalize(outputs["Ehat2"].float(), dim=-1),
                    labels,
                    margin=cfg.local_match_margin,
                    center_only=cfg.local_match_center_only,
                )
                llocal = 0.5 * (llocal + llocal2)
                valid_local = 0.5 * (valid_local + valid_local2)

            weighted = cfg.w_local_match * llocal
            total_loss = total_loss + weighted
            loss_dict["L_local_match"] = llocal.item()
            loss_dict["L_local_match_w"] = weighted.item()
            loss_dict["local_match_valid"] = valid_local

        if cfg.use_L_attach and "Ehat" in outputs and "global_feat" in outputs:
            latt = attach_loss(
                outputs["Ehat"].float(),
                outputs["global_feat"].float(),
                outputs["Beta_ng"].float() if "Beta_ng" in outputs else outputs["Beta"].float(),
                cfg.attach_temp
            )
            weighted = cfg.w_attach * latt
            total_loss = total_loss + weighted
            loss_dict["L_attach"] = latt.item()
            loss_dict["L_attach_w"] = weighted.item()
        
        if cfg.use_L_div and "Ehat" in outputs:
            ldiv = diversity_loss(
                outputs["Ehat"].float(),
                outputs["Beta_ng"].float() if "Beta_ng" in outputs else outputs["Beta"].float()
            )
            weighted = cfg.w_div * ldiv
            total_loss = total_loss + weighted
            loss_dict["L_div"] = ldiv.item()
            loss_dict["L_div_w"] = weighted.item()
        
        if cfg.use_L_center and "Ehat" in outputs:
            # Init tardive si besoin
            if self.center_loss is None:
                feat_dim = outputs["Ehat"].shape[-1]
                self.center_loss = CenterLoss(self.num_classes, feat_dim).to(device)
            B, C, D = outputs["Ehat"].shape
            feats = outputs["Ehat"].float().view(-1, D)
            labels_s = labels.repeat_interleave(C)
            lcenter = self.center_loss(feats, labels_s)
            weighted = cfg.w_center * lcenter
            total_loss = total_loss + weighted
            loss_dict["L_center"] = lcenter.item()
            loss_dict["L_center_w"] = weighted.item()
        
        if cfg.use_L_aug and "Ehat2" in outputs and "rel_vec" in outputs and "rel_vec2" in outputs:
            weights_source = "gamma" if getattr(cfg, "use_gamma_weights_for_matching", False) else "beta"
            laug = augmentation_consistency_loss(
                outputs["Ehat"].float(),
                outputs["rel_vec"].float(),
                outputs["Omega_ng"].float() if "Omega_ng" in outputs else outputs["Omega"].float(),
                outputs["Ehat2"].float(),
                outputs["rel_vec2"].float(),
                outputs["Omega2_ng"].float() if "Omega2_ng" in outputs else outputs["Omega2"].float(),
                labels, cfg.aug_match_temp, weights_source,
                use_ot=use_ot,
                use_cell_ot=use_cell_ot,
                use_stripe_ot=use_stripe_ot,
                U1=outputs.get("Uhat"),
                U2=outputs.get("Uhat2"),
                Gamma1=outputs.get("Gamma"),
                Gamma2=outputs.get("Gamma2"),
                ot_epsilon=ot_epsilon,
                ot_num_iters=ot_num_iters,
                ot_margi_eps=ot_margi_eps,
                        use_cross_view_consistency=use_cross_view_consistency,
                        cross_view_alpha=cross_view_alpha,
                        cross_view_pos_lambda=cross_view_pos_lambda,
                        cross_view_phi_scale=cross_view_phi_scale,
                        cross_view_phi_bias=cross_view_phi_bias,
                        cross_view_norm_transitive=cross_view_norm_transitive,
                        use_cross_view_row_consistency=use_cross_view_row_consistency,
                        cross_view_row_weight=cross_view_row_weight,
                        cross_view_row_pos_lambda=cross_view_row_pos_lambda,
                        use_view_prototype_propagation=use_view_prototype_propagation,
                        view_prototype_path=view_prototype_path,
                        view_propagation_lambda=view_propagation_lambda,
                        view_prototype_temp=view_prototype_temp,
                        use_view_prototype_span=use_view_prototype_span,
                        view_prototype_span_lambda=view_prototype_span_lambda,
                        view_transition_self=view_transition_self,
                        view_transition_neighbor1=view_transition_neighbor1,
                        view_transition_neighbor2=view_transition_neighbor2,
                        use_view_uncertainty_gate=use_view_uncertainty_gate,
                        view_prototype_repository=self.view_prototype_repository,
            )
            weighted = cfg.w_aug * laug
            total_loss = total_loss + weighted
            loss_dict["L_aug"] = laug.item()
            loss_dict["L_aug_w"] = weighted.item()


        # Relational NCE (relation-only): rel_vec uniquement
        if cfg.use_L_relNCE and "rel_vec" in outputs:
            lrel, valid_rel = rel_nce_loss(outputs["rel_vec"].float(), labels, cfg.set_nce_temp)
            weighted = cfg.w_relNCE * lrel
            total_loss = total_loss + weighted
            loss_dict["L_relNCE"] = lrel.item()
            loss_dict["L_relNCE_w"] = weighted.item()
            loss_dict["relNCE_valid"] = valid_rel

        # Center loss globale
        if cfg.use_L_center_g and "global_feat" in outputs:
            if self.center_loss_g is None:
                feat_dim_g = outputs["global_feat"].shape[-1]
                self.center_loss_g = CenterLoss(self.num_classes, feat_dim_g).to(device)
            lcenter_g = self.center_loss_g(outputs["global_feat"].float(), labels)
            weighted = cfg.w_center_g * lcenter_g
            total_loss = total_loss + weighted
            loss_dict["L_center_g"] = lcenter_g.item()
            loss_dict["L_center_g_w"] = weighted.item()
        
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


__all__ = ['sinkhorn_algorithm']
