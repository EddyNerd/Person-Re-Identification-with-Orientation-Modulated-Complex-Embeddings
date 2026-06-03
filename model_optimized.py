import math
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle different versions of torchreid (OSNet required)
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    torchreid = None
    TORCHREID_AVAILABLE = False
    warnings.warn("torchreid not available. Install it to use OSNet.")


@dataclass(frozen=True)
class StripeExtractionConfig:
    """Validated subset of cfg used by stripe extraction."""

    C_stripes: int
    R_rows: int
    beta_energy: str
    beta_thresh: float
    beta_slope: float
    force_a_one: bool = False
    force_extract_stripe_gnn: bool = True
    force_gnn_gamma_sigmoid: bool = True
    interstripe_transformer: Optional[bool] = None
    use_residual_film_orientation: Optional[bool] = None

    @classmethod
    def from_any(cls, cfg, C: int, R: int) -> "StripeExtractionConfig":
        if isinstance(cfg, cls):
            cls._validate(cfg)
            return cfg

        allowed = {field.name for field in fields(cls)}
        required = {"beta_energy", "beta_thresh", "beta_slope"}
        values: Dict[str, Any] = {"C_stripes": int(C), "R_rows": int(R)}

        if isinstance(cfg, dict):
            missing = required - set(cfg)
            if missing:
                raise ValueError(cls._missing_message(cfg, missing))
            unknown = set(cfg) - allowed
            if unknown:
                raise ValueError(f"Unknown stripe extraction config fields: {sorted(unknown)}")
            source_get = cfg.get
        else:
            missing = [name for name in required if not hasattr(cfg, name)]
            if missing:
                raise ValueError(cls._missing_message(cfg, missing))
            source_get = lambda name, default=None: getattr(cfg, name, default)

        for field in fields(cls):
            if field.name in {"C_stripes", "R_rows"}:
                continue
            default = field.default
            values[field.name] = source_get(field.name, default)

        parsed = cls(**values)
        cls._validate(parsed)
        return parsed

    @staticmethod
    def _missing_message(cfg, missing) -> str:
        suffix = ""
        if "beta_slope" in missing and (
            (isinstance(cfg, dict) and "beta_slop" in cfg) or hasattr(cfg, "beta_slop")
        ):
            suffix = " Did you mean 'beta_slope' instead of 'beta_slop'?"
        return f"Missing required stripe extraction config fields: {sorted(missing)}.{suffix}"

    @staticmethod
    def _validate(cfg: "StripeExtractionConfig") -> None:
        if cfg.C_stripes < 1 or cfg.R_rows < 1:
            raise ValueError("C_stripes and R_rows must be >= 1")
        if cfg.beta_energy not in {"l2", "amax"}:
            raise ValueError("beta_energy must be 'l2' or 'amax'")
        if float(cfg.beta_slope) <= 0:
            raise ValueError("beta_slope must be > 0")


class FeatureMapHook:
    """
    Forward hook for extracting intermediate feature maps.
    """
    
    def __init__(self, module: nn.Module):
        self.hook: Optional[Any] = module.register_forward_hook(self._hook_fn)
        self.feat_map: Optional[torch.Tensor] = None
    
    def _hook_fn(self, module, input, output):
        # Garde le graph en train, mais détache en eval/no_grad pour limiter la rétention mémoire.
        self.feat_map = output if torch.is_grad_enabled() else output.detach()
    
    def close(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
        self.feat_map = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        # Do not serialize the feature map to avoid oversized checkpoints
        state = self.__dict__.copy()
        state["feat_map"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class GridGraphConvolution(nn.Module):
    """
    Graph convolution over grid cells.

    Treats cells as a 2D grid graph where:
    - Each stripe k has R cells (nodes) arranged vertically
    - Default edges connect vertically adjacent cells (i, i+1) within a stripe
    - Optional horizontal edges connect adjacent stripes when use_horizontal=True

    Args:
        Uhat : (B, C, R, D) where B = batch, C = stripes, R = cells/stripe, D = feature dim
    Returns:
        Uhat_prime : (B, C, R, D) after message passing
    """
    def __init__(
        self,
        feat_dim: int,
        use_horizontal: bool = False,
        use_sigmoid_gamma: bool = False,
        residual_init: float = 0.7,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_horizontal = use_horizontal
        self.use_sigmoid_gamma = use_sigmoid_gamma
        residual_init = min(max(float(residual_init), 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        
        # Transformations pour aggregation: ρEu) et ρEu) pour chaque noeud
        self.phi_transform = nn.Linear(feat_dim, feat_dim, bias=True)
        self.psi_transform = nn.Linear(feat_dim, feat_dim, bias=True)
        # Gate to combine local and aggregated information
        self.gate = nn.Linear(2 * feat_dim, feat_dim, bias=True)
        # Optionnel: poids dynamiques issus des features après sigmoid
        self.gamma_proj = nn.Linear(feat_dim, 1, bias=True)
        
        nn.init.xavier_uniform_(self.phi_transform.weight)
        nn.init.xavier_uniform_(self.psi_transform.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.gamma_proj.weight)
    
    def forward(self, Uhat: torch.Tensor, Gamma: torch.Tensor, use_sigmoid_gamma: Optional[bool] = None) -> torch.Tensor:
        """
        Uhat: (B, C, R, D) - cell descriptors
        Gamma: (B, C, R) - cell confidence weights
        use_sigmoid_gamma: if True, computes weights from features via sigmoid(W_gamma u)
        Returns: Uhat_prime (B, C, R, D) after graph convolution
        """
        if use_sigmoid_gamma is None:
            use_sigmoid_gamma = self.use_sigmoid_gamma
        
        if use_sigmoid_gamma:
            gamma_all = torch.sigmoid(self.gamma_proj(Uhat)).squeeze(-1)  # (B, C, R)
        else:
            gamma_all = Gamma.to(device=Uhat.device, dtype=Uhat.dtype)

        weighted = Uhat * gamma_all.unsqueeze(-1)
        num = weighted.clone()
        den = gamma_all.clone()

        # Vertical 4-neighborhood: self, row-1, row+1.
        num[:, :, 1:, :] = num[:, :, 1:, :] + weighted[:, :, :-1, :]
        den[:, :, 1:] = den[:, :, 1:] + gamma_all[:, :, :-1]
        num[:, :, :-1, :] = num[:, :, :-1, :] + weighted[:, :, 1:, :]
        den[:, :, :-1] = den[:, :, :-1] + gamma_all[:, :, 1:]

        # Optional horizontal neighbors: stripe-1, stripe+1.
        if self.use_horizontal:
            num[:, 1:, :, :] = num[:, 1:, :, :] + weighted[:, :-1, :, :]
            den[:, 1:, :] = den[:, 1:, :] + gamma_all[:, :-1, :]
            num[:, :-1, :, :] = num[:, :-1, :, :] + weighted[:, 1:, :, :]
            den[:, :-1, :] = den[:, :-1, :] + gamma_all[:, 1:, :]

        aggregated = num / den.clamp_min(1e-9).unsqueeze(-1)
        phi_u = self.phi_transform(Uhat)
        psi_agg = self.psi_transform(aggregated)
        u_updated = self.gate(torch.cat([phi_u, psi_agg], dim=-1))
        alpha = torch.sigmoid(self.residual_logit).to(device=Uhat.device, dtype=Uhat.dtype)
        return alpha * u_updated + (1.0 - alpha) * Uhat


class InterStripeTransformer(nn.Module):
    """Coordinate stripe descriptors globally with self-attention."""
    def __init__(
        self,
        dim: int,
        num_stripes: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stripes = int(num_stripes)

        # Keep the module valid even if dim is not divisible by the requested heads.
        if dim % max(1, int(num_heads)) != 0:
            valid_heads = [h for h in range(int(num_heads), 0, -1) if dim % h == 0]
            num_heads = valid_heads[0] if valid_heads else 1

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_stripes, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(num_heads),
            dim_feedforward=dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(dim)

    def _pos_embed_for(self, seq_len: int) -> torch.Tensor:
        if seq_len == self.pos_embed.size(1):
            return self.pos_embed
        # Interpolate positional embeddings when the runtime stripe count differs.
        pe = self.pos_embed.transpose(1, 2)  # (1, D, C)
        pe = F.interpolate(pe, size=seq_len, mode="linear", align_corners=False)
        return pe.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._pos_embed_for(x.size(1)).to(dtype=x.dtype, device=x.device)
        x = self.transformer(x)
        return self.norm(x)


class ResidualFiLMOrientation(nn.Module):
    """
    Orientation-conditioned residual FiLM for stripe descriptors.

    Flow used in this project:
    1) Orientation signal v_i = (cos(theta_i), sin(theta_i)) is provided per stripe.
       - view_prototype and stripe_estimator sources produce stripe-level (B, C, 2)
         conditioning.
       - external image-level vectors may be passed as (B, 2) and are broadcast to
         all stripes.
    2) A small MLP maps v_i to 2D values.
    3) The output is split into gamma_i and delta_i.
    4) Residual FiLM modulation is applied:
       e'_i = (1 + gamma_i) * e_i + delta_i

        Zero initialization:
        - The last FiLM generator layer can be initialized with zeros.
        - At the beginning of training, gamma_i ~= 0 and delta_i ~= 0.
        - The architecture then behaves like the baseline and progressively learns
            orientation usage only when it improves the Re-ID objective.
        - This stabilizes early training and reduces feature-collapse risk.

        Interpretation:
        - Residual FiLM is an adaptive channel re-weighting mechanism.
        - In side views, unreliable frontal cues can be down-weighted while
            side-relevant cues are amplified.
        - In frontal views, torso symmetry and frontal texture cues can be emphasized.
        - FiLM does not invent identity information; it modulates information already
            present in the descriptor.

        Matching after FiLM:
        - After FiLM, stripe descriptors are normalized.
        - Similarity is computed with cosine inner product: S_ij = <e'_qi, e'_gj>.
        - OT cost uses C_ij = 1 - S_ij, so Sinkhorn receives an
            orientation-aware cost matrix.
    """

    # FiLM uses a 10-view anatomical basis shared with the visible-sector demo.
    VIEW_ORDER = (
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
    )
    VIEW_ANGLES_DEG = {
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
    }

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        zero_init: bool = True,
        formula: str = "residual",
        orientation_source: str = "view_prototype",
        view_prototype_path: str = "",
        prototype_temp: float = 10.0,
        use_view_prototype_span: bool = False,
        view_prototype_span_lambda: float = 1e-3,
        gamma_activation: str = "tanh",
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.formula = str(formula)
        self.orientation_source = str(orientation_source)
        self.prototype_temp = float(prototype_temp)
        self.gamma_activation = str(gamma_activation).lower()
        if self.formula not in {"residual", "beta_full"}:
            raise ValueError("FiLM formula must be 'residual' or 'beta_full'")
        if self.gamma_activation not in {"none", "identity", "tanh", "sigmoid2"}:
            raise ValueError("FiLM gamma_activation must be 'none', 'identity', 'tanh', or 'sigmoid2'")

        self.generator = nn.Sequential(
            nn.Linear(2, int(hidden_dim), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 2 * self.feat_dim, bias=True),
        )
        if zero_init:
            # Zero-init keeps FiLM close to identity at training start.
            last = self.generator[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

        angles = []
        for name in self.VIEW_ORDER:
            radians = math.radians(float(self.VIEW_ANGLES_DEG[name]))
            angles.append([math.cos(radians), math.sin(radians)])
        self.register_buffer("view_orientation_basis", torch.tensor(angles, dtype=torch.float32))

        prototypes = torch.empty(0, self.feat_dim, dtype=torch.float32)
        if self.orientation_source == "view_prototype" and view_prototype_path:
            prototypes = self._load_view_prototypes(
                view_prototype_path,
                feat_dim=self.feat_dim,
                use_span=bool(use_view_prototype_span),
                span_lambda=float(view_prototype_span_lambda),
            )
        self.register_buffer("view_prototypes", prototypes)

    @classmethod
    def _resolve_path(cls, prototype_path: str) -> Path:
        path = Path(str(prototype_path)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @classmethod
    def _load_view_prototypes(
        cls,
        prototype_path: str,
        feat_dim: int,
        use_span: bool,
        span_lambda: float,
    ) -> torch.Tensor:
        path = cls._resolve_path(prototype_path)
        if not path.exists():
            raise FileNotFoundError(f"View prototype bundle not found for FiLM: {path}")

        bundle = torch.load(str(path), map_location="cpu")
        if not isinstance(bundle, dict) or "prototypes" not in bundle or "view_names" not in bundle:
            raise ValueError(f"Invalid view prototype bundle for FiLM: {path}")

        prototypes = bundle["prototypes"]
        view_names = bundle["view_names"]
        if not torch.is_tensor(prototypes):
            prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        prototypes = prototypes.float()
        if prototypes.dim() != 2:
            raise ValueError(f"FiLM view prototypes must be 2D, got {tuple(prototypes.shape)}")
        if prototypes.size(1) != int(feat_dim):
            raise ValueError(
                f"FiLM prototype dimension mismatch: expected {feat_dim}, got {prototypes.size(1)}"
            )
        if not isinstance(view_names, list):
            raise ValueError(f"Invalid view_names in FiLM prototype bundle: {path}")

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
        missing = [name for name in cls.VIEW_ORDER if name not in name_to_idx]
        if missing:
            raise ValueError(f"FiLM prototype bundle is missing views: {missing}")

        ordered = torch.stack([prototypes[name_to_idx[name]] for name in cls.VIEW_ORDER], dim=0)
        ordered = F.normalize(ordered, dim=1)
        if use_span:
            ordered = cls._span_residualize(ordered, max(0.0, float(span_lambda)))
        return ordered.cpu().contiguous()

    @staticmethod
    def _span_residualize(prototypes: torch.Tensor, span_lambda: float, eps: float = 1e-9) -> torch.Tensor:
        P = F.normalize(prototypes.float(), dim=1)
        residuals = []
        for idx in range(P.size(0)):
            others = torch.cat([P[:idx], P[idx + 1:]], dim=0)
            gram = torch.matmul(others, others.t())
            if span_lambda > 0.0:
                eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
                gram = gram + float(span_lambda) * eye
                weights = torch.linalg.solve(gram, torch.matmul(others, P[idx]))
            else:
                weights = torch.matmul(torch.linalg.pinv(gram), torch.matmul(others, P[idx]))
            residual = P[idx] - torch.matmul(weights, others)
            residuals.append(F.normalize(residual, dim=0, eps=eps))
        return torch.stack(residuals, dim=0)

    def infer_orientation(self, stripes: torch.Tensor) -> Optional[torch.Tensor]:
        view_prototypes = getattr(self, "view_prototypes", None)
        if self.orientation_source != "view_prototype" or not torch.is_tensor(view_prototypes) or view_prototypes.numel() == 0:
            return None
        prototypes = view_prototypes.to(device=stripes.device, dtype=stripes.dtype)
        basis = self.view_orientation_basis.to(device=stripes.device, dtype=stripes.dtype)
        stripes_norm = F.normalize(stripes, dim=-1)
        logits = self.prototype_temp * torch.einsum("bcd,md->bcm", stripes_norm, prototypes)
        view_probs = torch.softmax(logits, dim=-1)
        # Weighted sum of canonical view directions; output is already in (cos, sin)-like space.
        orientation = torch.einsum("bcm,mv->bcv", view_probs, basis)
        return F.normalize(orientation, dim=-1, eps=1e-6)

    def forward(
        self,
        stripes: torch.Tensor,
        orientation_vec: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if orientation_vec is None:
            orientation_vec = self.infer_orientation(stripes)
        if orientation_vec is None:
            return stripes

        orientation_vec = orientation_vec.to(device=stripes.device, dtype=stripes.dtype)
        if self.orientation_source == "stripe_estimator" and orientation_vec.dim() == 2:
            raise ValueError(
                "film_orientation_source='stripe_estimator' requires stripe-level "
                "orientation vectors with shape (B, C, 2), not image-level (B, 2)."
            )
        if orientation_vec.dim() == 2:
            orientation_vec = orientation_vec.unsqueeze(1).expand(-1, stripes.size(1), -1)
        if orientation_vec.dim() != 3 or orientation_vec.size(-1) != 2:
            raise ValueError(
                "Residual FiLM orientation vector must have shape (B, 2) or (B, C, 2)"
            )
        if orientation_vec.size(0) != stripes.size(0) or orientation_vec.size(1) != stripes.size(1):
            raise ValueError(
                f"Residual FiLM orientation shape {tuple(orientation_vec.shape)} "
                f"is incompatible with stripes {tuple(stripes.shape)}"
            )

        # Ensure v_i is a unit orientation vector before conditioning.
        orientation_vec = F.normalize(orientation_vec, dim=-1, eps=1e-6)
        # MLP(v_i) -> [gamma_i, delta_i], then residual FiLM on stripe descriptor e_i.
        gamma, delta = self.generator(orientation_vec).chunk(2, dim=-1)
        if self.gamma_activation == "tanh":
            gamma = torch.tanh(gamma)
        elif self.gamma_activation == "sigmoid2":
            gamma = 2.0 * torch.sigmoid(gamma) - 1.0

        modulated = (1.0 + gamma) * stripes + delta
        if self.formula == "beta_full":
            if beta is None:
                raise ValueError("FiLM formula 'beta_full' requires stripe beta weights")
            beta = beta.to(device=stripes.device, dtype=stripes.dtype)
            if beta.dim() != 2 or beta.size(0) != stripes.size(0) or beta.size(1) != stripes.size(1):
                raise ValueError(
                    f"beta must have shape (B, C), got {tuple(beta.shape)} for stripes {tuple(stripes.shape)}"
                )
            modulated = stripes + beta.unsqueeze(-1) * (gamma * stripes + delta)
        return modulated


class StripeHead(nn.Module):
    """BN + Linear + scale pour classifier les stripes."""
    def __init__(self, in_dim: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        self.scale = scale
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        logits = self.fc(x)
        return logits * self.scale


class DOCModel(nn.Module):
    """
    Dynamic Object-Centric (DOC) re-identification model.

    Backbone: OSNet (torchreid).  A forward hook captures the last Conv2d feature map.

    Architecture stages:
        1. Spatial Decomposition     : feature map ↁEC×R grid cells via ``F.unfold``
        2. Object-Centric GNN        : message passing over the grid; learns γ confidence weights
        3. Inter-Stripe Transformer  : cross-stripe attention over purified stripe descriptors
        4. Global-Local Fusion       : [f_global  EE_k] ↁElinear projection ↁEÊ_k
          5. Dual FiLM Orientation (opt):
              - absolute-angle FiLM -> e'_i
              - relative-angle FiLM -> e''_i
          6. Complex Hermitian (opt)   : Z_i = e'_i + j·e''_i ∁E℁ED for Hermitian OT matching

    Key tensor shapes returned by ``get_stripes`` / ``extract_stripes_adaptive``::

        global_feat : (B, D)           L2-normalized global descriptor
        Ehat        : (B, C, D)        refined stripe descriptors
        Beta        : (B, C)           per-stripe foreground confidence weights
        Omega       : (B, C)           per-stripe OT marginal weights
        Z (opt)     : (B, C, D) complex Hermitian stripe embeddings

    For the full theoretical derivation (Hermitian inner product, FiLM conditioning,
    Optimal Transport formulation), refer to ``THEORY.md`` at the repository root.
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        hook_layer: str = "",
        interstripe_transformer: bool = True,
        interstripe_num_stripes: int = 5,
        interstripe_num_heads: int = 8,
        interstripe_num_layers: int = 2,
        interstripe_dropout: float = 0.1,
        interstripe_concat_global_local: bool = False,
        interstripe_concat_dropout: float = 0.0,
        gnn_use_horizontal: bool = False,
        gnn_residual_init: float = 0.7,
        use_residual_film_orientation: bool = True,
        film_hidden_dim: int = 128,
        film_zero_init: bool = True,
        film_formula: str = "residual",
        film_gamma_activation: str = "tanh",
        film_orientation_source: str = "view_prototype",
        film_view_prototype_path: str = "",
        film_prototype_temp: float = 10.0,
        film_use_view_prototype_span: bool = False,
        film_view_prototype_span_lambda: float = 1e-3,
        use_complex_hermitian: bool = True,
    ):
        super().__init__()
        
        if not TORCHREID_AVAILABLE:
            raise RuntimeError("torchreid is required for DOCModel (OSNet backbone). Install torchreid.")
        torchreid_module = cast(Any, torchreid)

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Build OSNet backbone via torchreid
        try:
            self.backbone = torchreid_module.models.build_model(
                name=backbone_name,
                num_classes=num_classes,
                loss="softmax",
                pretrained=pretrained,
            )
        except Exception:
            # Fallback for older versions of torchreid
            self.backbone = torchreid_module.models.build_model(
                name=backbone_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )
        
        # Locate the target layer for the feature hook
        self.hook_handle: Optional[FeatureMapHook] = None
        target_layer = self._find_target_layer(hook_layer)
        
        if target_layer is None:
            raise RuntimeError(
                f"No Conv2d layer found in backbone '{backbone_name}'"
            )
        
        self.hook_handle = FeatureMapHook(target_layer)
        self._hook_layer_name = str(target_layer)

        # Stripe classifier (feature dim = hooked layer out_channels)
        stripe_feat_dim = getattr(target_layer, "out_channels", None)
        if stripe_feat_dim is None:
            raise RuntimeError("Cannot retrieve out_channels to initialize stripe_classifier")
        self.stripe_classifier = StripeHead(stripe_feat_dim, num_classes, scale=10.0)        
        # GNN pour raffinage des cellules de stripes
        self.grid_gnn = GridGraphConvolution(
            feat_dim=stripe_feat_dim,
            use_horizontal=bool(gnn_use_horizontal),
            residual_init=float(gnn_residual_init),
        )
        self.use_interstripe_transformer = bool(interstripe_transformer)
        self.interstripe_transformer = InterStripeTransformer(
            dim=stripe_feat_dim,
            num_stripes=int(interstripe_num_stripes),
            num_heads=int(interstripe_num_heads),
            num_layers=int(interstripe_num_layers),
            dropout=float(interstripe_dropout),
        ) if self.use_interstripe_transformer else None
        self.use_interstripe_concat_global_local = bool(interstripe_concat_global_local)
        self.interstripe_concat_proj = nn.Sequential(
            nn.Linear(stripe_feat_dim * 2, stripe_feat_dim, bias=True),
            nn.GELU(),
            nn.Dropout(float(interstripe_concat_dropout)),
            nn.LayerNorm(stripe_feat_dim),
        ) if self.use_interstripe_concat_global_local else None
        if self.interstripe_concat_proj is not None:
            first = self.interstripe_concat_proj[0]
            if isinstance(first, nn.Linear):
                nn.init.xavier_uniform_(first.weight)
                if first.bias is not None:
                    nn.init.zeros_(first.bias)
        self.use_residual_film_orientation = bool(use_residual_film_orientation)
        self.residual_film_orientation_abs = ResidualFiLMOrientation(
            feat_dim=stripe_feat_dim,
            hidden_dim=int(film_hidden_dim),
            zero_init=bool(film_zero_init),
            formula=str(film_formula),
            gamma_activation=str(film_gamma_activation),
            orientation_source=str(film_orientation_source),
            view_prototype_path=str(film_view_prototype_path),
            prototype_temp=float(film_prototype_temp),
            use_view_prototype_span=bool(film_use_view_prototype_span),
            view_prototype_span_lambda=float(film_view_prototype_span_lambda),
        ) if self.use_residual_film_orientation else None
        self.residual_film_orientation_rel = ResidualFiLMOrientation(
            feat_dim=stripe_feat_dim,
            hidden_dim=int(film_hidden_dim),
            zero_init=bool(film_zero_init),
            formula=str(film_formula),
            gamma_activation=str(film_gamma_activation),
            orientation_source=str(film_orientation_source),
            view_prototype_path=str(film_view_prototype_path),
            prototype_temp=float(film_prototype_temp),
            use_view_prototype_span=bool(film_use_view_prototype_span),
            view_prototype_span_lambda=float(film_view_prototype_span_lambda),
        ) if self.use_residual_film_orientation else None
        
        # Complex Hermitian embeddings
        self.use_complex_hermitian = bool(use_complex_hermitian)
        self.real_proj = None
        self.imag_proj = None
        
        # Cache for complex embeddings from last extraction (for automatic OT usage)
        self._last_Z_q: Optional[torch.Tensor] = None
        self._last_Z_g: Optional[torch.Tensor] = None
    def _find_target_layer(self, hook_layer: str) -> Optional[nn.Module]:
        """Find the target layer for the feature map hook."""
        if hook_layer:
            layer = dict(self.backbone.named_modules()).get(hook_layer)
            if layer:
                return layer
            warnings.warn(f"Layer '{hook_layer}' not found, falling back to heuristic")
        
        # Heuristic: use the last Conv2d layer
        target = None
        for name, m in self.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                target = m
        return target
    
    def forward_global(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits, global features, and the spatial feature map.

        Returns:
            logits   : (B, num_classes)
            features : (B, D)        L2-normalized global descriptor
            feat_map : (B, D, H, W)  spatial feature map from the hooked layer
        """
        # Forward backbone
        out = self.backbone(x)
        
        # Extraction logits (compatibilité versions)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        
        # Retrieve feature map from hook
        if self.hook_handle is None:
            raise RuntimeError("Feature map hook is not initialized")
        feat_map = self.hook_handle.feat_map
        if feat_map is None:
            raise RuntimeError(
                "Feature map hook returned None. "
                "Ensure forward_global is called in normal eval/training mode."
            )
        
        # Compute global features from the feature map
        # More robust than relying on the backbone output directly
        global_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        global_feat = F.normalize(global_feat, dim=1)
        # Release the reference to avoid retaining the tensor between calls
        self.hook_handle.feat_map = None

        return logits, global_feat, feat_map
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (for compatibility)."""
        logits, _, _ = self.forward_global(x)
        return logits

    @staticmethod
    def adapt_legacy_single_film_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible checkpoint adapter.

        Old checkpoints store one FiLM module under keys:
          residual_film_orientation.*

        New architecture uses two FiLM modules:
          residual_film_orientation_abs.* and residual_film_orientation_rel.*

        This helper duplicates legacy FiLM weights into both branches when the
        dual keys are missing.
        """
        if not isinstance(state_dict, dict):
            return state_dict

        has_old = any(str(k).startswith("residual_film_orientation.") for k in state_dict.keys())
        has_new = any(
            str(k).startswith("residual_film_orientation_abs.")
            or str(k).startswith("residual_film_orientation_rel.")
            for k in state_dict.keys()
        )
        if not has_old or has_new:
            return state_dict

        adapted = dict(state_dict)
        for key, value in state_dict.items():
            key_str = str(key)
            if not key_str.startswith("residual_film_orientation."):
                continue
            suffix = key_str[len("residual_film_orientation."):]
            adapted[f"residual_film_orientation_abs.{suffix}"] = value
            adapted[f"residual_film_orientation_rel.{suffix}"] = value
        return adapted

    def close(self):
        """Clean up registered hooks."""
        if self.hook_handle:
            self.hook_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    # =====================================================================
    # Static DOC methods (optimized)
    # =====================================================================

    def traceable_forward(
        self,
        x: torch.Tensor,
        num_stripes: int = 5,
        num_rows: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Traceable forward pass for ONNX export with fixed-size pooling.

        Uses ``F.adaptive_avg_pool2d`` instead of the dynamic ``F.unfold`` path
        so all kernel sizes are compile-time constants during ``torch.jit.trace``.

        **Components NOT included** (require dynamic shapes or runtime branching):

        - GNN message passing (dynamic graph scatter operations)
        - FiLM orientation conditioning (runtime-dependent branching)
        - Complex Hermitian projection (post-stripe; add via a separate export step)

        Args:
            x           : ``(B, 3, H, W)`` input images; H/W must match training resolution.
            num_stripes : number of vertical stripes (C).  Must be a Python ``int``.
            num_rows    : number of horizontal rows per stripe (R).  Must be a Python ``int``.

        Returns:
            global_feat  : ``(B, D)``       L2-normalized global descriptor.
            stripe_descs : ``(B, C, D)``    L2-normalized stripe descriptors (simple pool, no GNN).
        """
        # Step 1: backbone forward + hook capture
        _ = self.backbone(x)
        if self.hook_handle is None:
            raise RuntimeError("Feature map hook is not initialized")
        feat_map = self.hook_handle.feat_map  # (B, D, H', W')
        if feat_map is None:
            raise RuntimeError("Feature map hook returned None during traceable_forward")

        # Step 2: global descriptor
        global_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B, D)
        global_feat = F.normalize(global_feat, dim=1)

        # Step 3: stripe descriptors via fixed-size pooling (no dynamic unfold / no padding)
        # adaptive_avg_pool2d handles arbitrary input size ↁEfixed (num_rows, num_stripes) output
        pooled = F.adaptive_avg_pool2d(feat_map, (num_rows, num_stripes))  # (B, D, R, C)
        stripe_descs = pooled.mean(dim=2).permute(0, 2, 1).contiguous()   # (B, C, D)
        stripe_descs = F.normalize(stripe_descs, dim=2)

        # Step 4: inter-stripe transformer (standard attention ↁEONNX-compatible)
        if self.use_interstripe_transformer and self.interstripe_transformer is not None:
            stripe_descs = self.interstripe_transformer(stripe_descs)
            stripe_descs = F.normalize(stripe_descs, dim=2)

        return global_feat, stripe_descs
    
    @staticmethod
    def extract_stripes(
        feat_map: torch.Tensor,
        C: int,
        R: int,
        beta_cfg,
        return_cells: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extraction vectorisée des embeddings et fiabilités par stripes.
        
        Version optimisée utilisant F.unfold avec padding pour garantir
        exactement R x C patches.
        """
        beta_cfg = StripeExtractionConfig.from_any(beta_cfg, C, R)
        C = beta_cfg.C_stripes
        R = beta_cfg.R_rows
        B, Ch, H, W = feat_map.shape
        
        # 1. Compute energy map
        if beta_cfg.beta_energy == "amax":
            E = feat_map.abs().amax(dim=1)  # (B, H, W)
        else:  # l2
            E = torch.norm(feat_map, dim=1)  # (B, H, W)
        
        # Normalisation par image (min-max)
        E_min = E.amin(dim=(1, 2), keepdim=True)
        E_max = E.amax(dim=(1, 2), keepdim=True)
        denom = (E_max - E_min).clamp_min(1e-6)
        E_norm = ((E - E_min) / denom).clamp(0.0, 1.0)
        
        # 2. Compute Beta (foreground confidence map)
        beta_map = torch.sigmoid(
            (E_norm - beta_cfg.beta_thresh) / beta_cfg.beta_slope
        )  # (B, H, W)
        
        # 3. Découpage en grille R x C avec unfold et padding
        # Compute kernel sizes by rounding up
        kernel_h = math.ceil(H / R)
        kernel_w = math.ceil(W / C)
        
        # Pad bas/droite pour tomber juste sur R*C cellules
        pad_h = kernel_h * R - H
        pad_w = kernel_w * C - W
        if pad_h > 0 or pad_w > 0:
            feat_map = F.pad(feat_map, (0, pad_w, 0, pad_h))
            beta_map = F.pad(beta_map, (0, pad_w, 0, pad_h))
            H, W = feat_map.shape[2], feat_map.shape[3]
        
        # Unfold pour feat_map: (B, Ch*kh*kw, R*C)
        feat_unfold = F.unfold(
            feat_map,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w)
        )
        # Reshape: (B, Ch, kh, kw, R, C) puis permute pour (B, R, C, Ch, kh, kw)
        feat_patches = feat_unfold.view(B, Ch, kernel_h, kernel_w, R, C)
        feat_patches = feat_patches.permute(0, 4, 5, 1, 2, 3)  # (B, R, C, Ch, kh, kw)
        
        # Unfold pour beta_map
        beta_unfold = F.unfold(
            beta_map.unsqueeze(1),
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w)
        )
        beta_patches = beta_unfold.view(B, 1, kernel_h, kernel_w, R, C)
        beta_patches = beta_patches.permute(0, 4, 5, 2, 3, 1).squeeze(-1)  # (B, R, C, kh, kw)

        # Optional eval/debug mode: force confidence map a(r,c) to one.
        if beta_cfg.force_a_one:
            beta_patches = torch.ones_like(beta_patches)
        
        # 4. Moyenne Beta par cellule
        alpha_cells = beta_patches.mean(dim=(3, 4))  # (B, R, C)
        
        # 5. Pooling pondéré des features
        beta_weights = beta_patches.unsqueeze(3)  # (B, R, C, 1, kh, kw)
        weighted_feat = (feat_patches * beta_weights).sum(dim=(4, 5))  # (B, R, C, Ch)
        norm_factor = beta_weights.sum(dim=(4, 5)).clamp_min(1e-9)  # (B, R, C, 1)
        
        E_cells = weighted_feat / norm_factor  # (B, R, C, Ch)
        
        # 6. Agrégation verticale (sur R) pour obtenir C stripes finales
        alpha_sum = alpha_cells.sum(dim=1, keepdim=True).clamp_min(1e-9)  # (B, 1, C)
        
        Ehat = (E_cells * alpha_cells.unsqueeze(-1)).sum(dim=1)  # (B, C, Ch)
        Ehat = Ehat / alpha_sum.permute(0, 2, 1)  # Normalisation
        
        Ehat = F.normalize(Ehat, dim=2)
        Beta = alpha_cells.mean(dim=1)  # (B, C)

        if not return_cells:
            return Ehat, Beta

        # Uhat: cellules verticales par stripe (B, C, R, D)
        Uhat = E_cells.permute(0, 2, 1, 3).contiguous()
        Uhat = F.normalize(Uhat, dim=3)
        Gamma = alpha_cells.permute(0, 2, 1).contiguous()  # (B, C, R)
        return Ehat, Beta, Uhat, Gamma
    
    def apply_gnn_to_cells(
        self,
        Uhat: torch.Tensor,
        Gamma: torch.Tensor,
        use_sigmoid_gamma: bool = False,
    ) -> torch.Tensor:
        """
        Apply the GNN to stripe cells for refinement via message passing.

        Args:
            Uhat            : (B, C, R, D) cell descriptors (C stripes, R cells/stripe, D dim)
            Gamma           : (B, C, R) cell confidence weights
            use_sigmoid_gamma: if True, use sigmoid(W_gamma u) weights instead of Beta weights

        Returns:
            Uhat_prime : (B, C, R, D) GNN-refined cell descriptors
        """
        return self.grid_gnn(Uhat, Gamma, use_sigmoid_gamma=use_sigmoid_gamma)

    def _store_complex_embedding_cache(self, Z: Optional[torch.Tensor], cache_role: str = "query") -> None:
        """Store last complex embedding in query/gallery cache according to extraction role."""
        if Z is not None:
            Z = Z.detach()
            if not torch.is_grad_enabled():
                Z = Z.cpu()
        role = str(cache_role).lower()
        if role in {"query", "q"}:
            self._last_Z_q = Z
            return
        if role in {"gallery", "g"}:
            self._last_Z_g = Z
            return
        if role in {"both", "all"}:
            self._last_Z_q = Z
            self._last_Z_g = Z
            return
        warnings.warn(f"Unknown cache_role '{cache_role}', defaulting to query cache")
        self._last_Z_q = Z

    def _get_complex_embedding_cache(self, cache_role: str = "query") -> Optional[torch.Tensor]:
        """Read query/gallery complex embedding cache according to extraction role."""
        role = str(cache_role).lower()
        if role in {"gallery", "g"}:
            return self._last_Z_g
        return self._last_Z_q
    
    def extract_stripes_with_gnn(
        self,
        feat_map: torch.Tensor,
        C: int,
        R: int,
        beta_cfg,
        orientation_vec: Optional[torch.Tensor] = None,
        return_complex: bool = False,
        cache_role: str = "query",
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extraction des stripes avec application du GNN pour affiner les cellules.
        
        Retourne: Ehat_gnn, Beta, Uhat_gnn, Gamma
        où Ehat_gnn et Uhat_gnn sont affinées par le GNN.
        """
        beta_cfg = StripeExtractionConfig.from_any(beta_cfg, C, R)
        C = beta_cfg.C_stripes
        R = beta_cfg.R_rows
        # 1. Extraction initiale avec cellules
        Ehat, Beta, Uhat, Gamma = self.extract_stripes(
            feat_map, C, R, beta_cfg, return_cells=True
        )
        
        # 2. Application du GNN aux cellules
        use_sigmoid_gamma = beta_cfg.force_gnn_gamma_sigmoid
        Uhat_gnn = self.apply_gnn_to_cells(Uhat, Gamma, use_sigmoid_gamma=use_sigmoid_gamma)
        
        # 3. Calcul des poids dynamiques γ à partir des cellules affinées si demandé
        if use_sigmoid_gamma:
            Gamma_gnn = torch.sigmoid(self.grid_gnn.gamma_proj(Uhat_gnn)).squeeze(-1)  # (B, C, R)
        else:
            Gamma_gnn = Gamma
        
        # 4. Ré-agrégation verticale avec cellules affinées
        # Pour chaque stripe k: E_k^(GNN) = Σ_i gamma_(i,k) * u'_(i,k) / Σ_i gamma_(i,k)
        Ehat_gnn = (Uhat_gnn * Gamma_gnn.unsqueeze(-1)).sum(dim=2)  # (B, C, D)
        gamma_sum = Gamma_gnn.sum(dim=2, keepdim=True).clamp_min(1e-9)  # (B, C, 1)
        Ehat_gnn = Ehat_gnn / gamma_sum

        # 5. Coordination globale inter-stripe via Transformer
        use_interstripe = (
            self.use_interstripe_transformer
            if beta_cfg.interstripe_transformer is None
            else bool(beta_cfg.interstripe_transformer)
        )
        if use_interstripe and self.interstripe_transformer is not None:
            Ehat_gnn = self.interstripe_transformer(Ehat_gnn)

        # Optional global-local fusion: concat each stripe descriptor with the global descriptor,
        # then project back to D so downstream losses keep the same shape.
        if use_interstripe and self.use_interstripe_concat_global_local and self.interstripe_concat_proj is not None:
            global_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
            global_feat = F.normalize(global_feat, dim=1)
            global_expanded = global_feat.unsqueeze(1).expand(-1, Ehat_gnn.size(1), -1)
            Ehat_gnn = self.interstripe_concat_proj(torch.cat([Ehat_gnn, global_expanded], dim=2))

        Ehat_base = Ehat_gnn
        use_film = (
            self.use_residual_film_orientation
            if beta_cfg.use_residual_film_orientation is None
            else bool(beta_cfg.use_residual_film_orientation)
        )
        orientation_abs = orientation_vec
        orientation_rel = orientation_vec
        if orientation_vec is not None and torch.is_tensor(orientation_vec):
            if orientation_vec.dim() == 3 and orientation_vec.size(-1) == 4:
                orientation_abs = orientation_vec[:, :, :2]
                orientation_rel = orientation_vec[:, :, 2:]
            elif orientation_vec.dim() == 2 and orientation_vec.size(-1) == 4:
                orientation_abs = orientation_vec[:, :2]
                orientation_rel = orientation_vec[:, 2:]

        if use_film and self.residual_film_orientation_abs is not None and self.residual_film_orientation_rel is not None:
            Ehat_abs = self.residual_film_orientation_abs(Ehat_base, orientation_vec=orientation_abs, beta=Beta)
            Ehat_rel = self.residual_film_orientation_rel(Ehat_base, orientation_vec=orientation_rel, beta=Beta)
        else:
            Ehat_abs = Ehat_base
            Ehat_rel = Ehat_base

        # Canonical naming: mod_abs/mod_rel branches.
        Ehat_abs_out = Ehat_abs
        Ehat_rel_out = Ehat_rel
        Ehat_gnn = Ehat_abs

        # Create complex Hermitian embedding if enabled (before normalization)
        Z = None
        if self.use_complex_hermitian:
            # Dual-FiLM complex embedding: real = absolute-angle FiLM, imag = relative-angle FiLM.
            real_part = Ehat_abs_out
            imag_part = Ehat_rel_out
            Z = torch.complex(real_part, imag_part)  # (B, C, D) complex tensor
            # Normalize each complex stripe vector to unit norm over feature dim D.
            Z_norm = torch.sqrt((torch.abs(Z) ** 2).sum(dim=-1, keepdim=True)).clamp_min(1e-9)  # (B, C, 1)
            Z = Z / Z_norm
            # Store in role-aware cache for automatic OT usage
            self._store_complex_embedding_cache(Z, cache_role=cache_role)

        # Normalized stripes are consumed by cosine-based matching; OT then uses C_ij = 1 - S_ij.
        Ehat_gnn = F.normalize(Ehat_gnn, dim=2)

        if return_complex:
            return Ehat_gnn, Beta, Uhat_gnn, Gamma_gnn, Ehat_abs_out, Ehat_rel_out

        return Ehat_gnn, Beta, Uhat_gnn, Gamma_gnn
    
    def extract_stripes_adaptive(
        self,
        feat_map: torch.Tensor,
        C: int,
        R: int,
        beta_cfg,
        orientation_vec: Optional[torch.Tensor] = None,
        return_cells: bool = False,
        return_complex: bool = False,
        return_complex_embeddings: bool = False,
        cache_role: str = "query",
    ) -> tuple:
        """
        Adaptive router: uses extract_stripes_with_gnn when force_extract_stripe_gnn=True,
        otherwise falls back to the legacy extract_stripes path.

        Args:
            feat_map, C, R, beta_cfg  : extraction parameters
            return_cells              : if True, also return Uhat and Gamma
            return_complex_embeddings : if True, append Z (complex Hermitian embeddings) as last item

        Returns:
            Tuple whose contents depend on flags:
            - default          : (Ehat, Beta)
            - return_cells     : (Ehat, Beta, Uhat, Gamma)
            - return_complex   : (Ehat, Beta[, Uhat, Gamma], Ehat_abs, Ehat_rel)
            - +embeddings      : above + (Z,)  where Z may be None on the legacy path
        """
        beta_cfg = StripeExtractionConfig.from_any(beta_cfg, C, R)
        C = beta_cfg.C_stripes
        R = beta_cfg.R_rows
        use_gnn = beta_cfg.force_extract_stripe_gnn
        
        if use_gnn:
            # Version avec GNN
            outputs = self.extract_stripes_with_gnn(
                feat_map,
                C,
                R,
                beta_cfg,
                orientation_vec=orientation_vec,
                return_complex=return_complex,
                cache_role=cache_role,
            )
            if return_complex:
                Ehat_gnn, Beta, Uhat_gnn, Gamma, Ehat_abs, Ehat_rel = outputs
                if not return_cells:
                    result = (Ehat_gnn, Beta, Ehat_abs, Ehat_rel)
                else:
                    result = (Ehat_gnn, Beta, Uhat_gnn, Gamma, Ehat_abs, Ehat_rel)
            else:
                Ehat_gnn, Beta, Uhat_gnn, Gamma = outputs
                if not return_cells:
                    result = (Ehat_gnn, Beta)
                else:
                    result = (Ehat_gnn, Beta, Uhat_gnn, Gamma)
            
            # Add Z if requested
            if return_complex_embeddings:
                result = result + (self._get_complex_embedding_cache(cache_role=cache_role),)
            return result
        else:
            # Legacy path
            if not return_complex:
                result = self.extract_stripes(feat_map, C, R, beta_cfg, return_cells=return_cells)
            else:
                legacy = self.extract_stripes(feat_map, C, R, beta_cfg, return_cells=return_cells)
                if return_cells:
                    Ehat, Beta, Uhat, Gamma = legacy
                    result = (Ehat, Beta, Uhat, Gamma, Ehat, Ehat)
                else:
                    Ehat, Beta = legacy
                    result = (Ehat, Beta, Ehat, Ehat)
            
            # Add Z if requested (None for legacy path)
            if return_complex_embeddings:
                result = result + (None,)
            return result
    
    def get_last_complex_embeddings(self):
        """
        Return the Hermitian complex embeddings stored in the cache after the
        last stripe extraction.

        Returns:
            (Z_q, Z_g): complex tensors if available, otherwise (None, None)
        """
        return self._last_Z_q, self._last_Z_g
    
    def set_complex_embeddings(self, Z_q=None, Z_g=None):
        """
        Manually set complex embeddings (for multi-batch evaluation).
        """
        if Z_q is not None:
            Z_q = Z_q.detach()
            if not torch.is_grad_enabled():
                Z_q = Z_q.cpu()
        if Z_g is not None:
            Z_g = Z_g.detach()
            if not torch.is_grad_enabled():
                Z_g = Z_g.cpu()
        self._last_Z_q = Z_q
        self._last_Z_g = Z_g

    @staticmethod
    def complex_hermitian_similarity(
        Z_q: torch.Tensor,
        Z_g: torch.Tensor,
        normalize: bool = True,
        crg_lambda: float = 0.5,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Similarité Hermitienne stripe-à-stripe.

        Implémente explicitement:
            H = einsum("qcd,gkd->qgck", conj(Z_q), Z_g)
            S = |Re(H)| * exp(-lambda * |Im(H)| / (|Re(H)| + eps))

        Args:
            Z_q: (Bq, Cq, D) tenseur complexe (query)
            Z_g: (Bg, Cg, D) tenseur complexe (gallery)
            normalize: si True, normalise par ||Z_q||*||Z_g|| pour borner S dans [0,1]
            eps: stabilité numérique

        Returns:
            S: (Bq, Bg, Cq, Cg)
        """
        if not torch.is_tensor(Z_q) or not torch.is_tensor(Z_g):
            raise TypeError("Z_q and Z_g must be tensors")
        if Z_q.dim() != 3 or Z_g.dim() != 3:
            raise ValueError(f"Expected (B,C,D) tensors, got {tuple(Z_q.shape)} and {tuple(Z_g.shape)}")
        if not Z_q.is_complex() or not Z_g.is_complex():
            raise TypeError(f"Expected complex tensors, got {Z_q.dtype} and {Z_g.dtype}")
        if Z_q.size(-1) != Z_g.size(-1):
            raise ValueError(f"Feature dim mismatch: {tuple(Z_q.shape)} vs {tuple(Z_g.shape)}")

        H = torch.einsum("qcd,gkd->qgck", torch.conj(Z_q), Z_g)

        if normalize:
            q_norm = torch.sqrt((torch.abs(Z_q) ** 2).sum(dim=-1)).clamp_min(eps)  # (Bq, Cq)
            g_norm = torch.sqrt((torch.abs(Z_g) ** 2).sum(dim=-1)).clamp_min(eps)  # (Bg, Cg)
            denom = q_norm[:, None, :, None] * g_norm[None, :, None, :]
            H = H / denom.clamp_min(eps)

        real_abs = H.real.abs()
        imag_abs = H.imag.abs()
        gate = torch.exp(-float(crg_lambda) * imag_abs / (real_abs + float(eps)))
        S = real_abs * gate

        return S.clamp(0.0, 1.0)

    @staticmethod
    def complex_hermitian_cost(
        Z_q: torch.Tensor,
        Z_g: torch.Tensor,
        normalize: bool = True,
        crg_lambda: float = 0.5,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Coût Hermitien pour OT/Sinkhorn:
            C = 1 - S,
        avec S = complex_hermitian_similarity(Z_q, Z_g).
        """
        S = DOCModel.complex_hermitian_similarity(
            Z_q,
            Z_g,
            normalize=normalize,
            crg_lambda=crg_lambda,
            eps=eps,
        )
        return (1.0 - S).clamp(0.0, 2.0)
    
    @staticmethod
    def compute_omega(
        Ehat: torch.Tensor,
        Beta: torch.Tensor,
        omega_mode: str = "beta_rel",
        Uhat: Optional[torch.Tensor] = None,
        Gamma: Optional[torch.Tensor] = None,
        cfg=None,
    ) -> torch.Tensor:
        """
        Compute Omega stripe weights.

        Relational computation (mode rel/beta_rel):
        - inter-stripe score per stripe
        - intra-stripe score (adjacent cells) if Uhat/Gamma are available
        - intra-image normalization (z-score) then sigmoid

        In stripe OT with dynamic marginals, the reference behavior uses Beta as
        the marginal.  Gamma is only used if use_gamma_weights_for_matching is
        explicitly enabled.
        """
        eps = float(getattr(cfg, "relvec_eps", 1e-6)) if cfg is not None else 1e-6
        eta_inter_cfg = float(getattr(cfg, "relvec_eta_inter", 0.5)) if cfg is not None else 0.5
        eta_intra_cfg = float(getattr(cfg, "relvec_eta_intra", 0.5)) if cfg is not None else 0.5
        if cfg is not None:
            use_cell_ot_cfg = bool(getattr(cfg, "use_cell_ot_matching", False))
            use_stripe_ot_cfg = getattr(cfg, "use_stripe_ot_matching", None)
            if use_stripe_ot_cfg is None:
                # Legacy behavior: old use_cell_ot_matching toggled stripe OT.
                use_cell_ot = False
                use_stripe_ot = use_cell_ot_cfg
            else:
                use_cell_ot = use_cell_ot_cfg
                use_stripe_ot = bool(use_stripe_ot_cfg)
        else:
            use_cell_ot = False
            use_stripe_ot = False
        use_any_ot = use_cell_ot or use_stripe_ot
        skip_omega_rel_when_ot = bool(getattr(cfg, "skip_omega_rel_when_ot", True)) if cfg is not None else True
        use_gamma_weights = bool(getattr(cfg, "use_gamma_weights_for_matching", False)) if cfg is not None else False
        use_uniform_ot_marginals = bool(getattr(cfg, "use_uniform_ot_marginals", False)) if cfg is not None else False
        # Choix du weighting source : Beta par défaut, Gamma seulement sur demande.
        # En mode GNN, Gamma correspond aux poids appris pour les cellules.
        if Gamma is not None and use_gamma_weights:
            # Aggregated Gamma: mean over cells (R) for each stripe.
            beta_w = Gamma.float().mean(dim=2)  # (B, C) - aggregate over the R (cell) dimension
        else:
            beta_w = Beta.float()

        if use_uniform_ot_marginals:
            return torch.ones_like(beta_w)

        # Fast path for OT: Omega is only used as marginals, not in the OT cost itself.
        # Skip the relational inter/intra + z-score + sigmoid computation.
        if use_any_ot and skip_omega_rel_when_ot:
            if omega_mode == "ones":
                return torch.ones_like(beta_w)
            if omega_mode == "rel":
                warnings.warn(
                    "omega_mode='rel' requested with OT while skip_omega_rel_when_ot=True; "
                    "using uniform OT marginals. Set skip_omega_rel_when_ot=False to compute relational marginals.",
                    stacklevel=2,
                )
                return torch.ones_like(beta_w)
            return beta_w

        E = F.normalize(Ehat.float(), dim=2)
        
        _, C, _ = E.shape

        # Inter-stripes par stripe i: moyenne ponderee sur j != i.
        if C > 1:
            sim_inter = torch.einsum("bcd,bkd->bck", E, E)  # (B, C, C), cos in [-1,1]
            phi_inter = (sim_inter + 1.0) * 0.5  # [0,1]
            mask_offdiag = (1.0 - torch.eye(C, device=E.device, dtype=E.dtype)).unsqueeze(0)
            w_inter = beta_w.unsqueeze(2) * beta_w.unsqueeze(1) * mask_offdiag
            num_inter = (w_inter * phi_inter).sum(dim=2)  # (B, C)
            den_inter = w_inter.sum(dim=2).clamp_min(eps)  # (B, C)
            s_inter = num_inter / den_inter
        else:
            s_inter = torch.ones_like(Beta, dtype=E.dtype)

        # Intra-stripe par stripe i: moyenne ponderee sur cellules adjacentes k,k+1.
        intra_available = Uhat is not None and Gamma is not None and Uhat.size(2) > 1
        if intra_available:
            assert Uhat is not None and Gamma is not None
            U = F.normalize(Uhat.float(), dim=3)
            gamma_w = Gamma.float()
            uk = U[:, :, :-1, :]
            uk1 = U[:, :, 1:, :]
            sim_intra_adj = (uk * uk1).sum(dim=3)  # (B, C, K-1)
            phi_intra = (sim_intra_adj + 1.0) * 0.5  # [0,1]
            w_intra = gamma_w[:, :, :-1] * gamma_w[:, :, 1:]
            num_intra = (w_intra * phi_intra).sum(dim=2)  # (B, C)
            den_intra = w_intra.sum(dim=2).clamp_min(eps)  # (B, C)
            s_intra = num_intra / den_intra
            eta_sum = max(eta_inter_cfg + eta_intra_cfg, eps)
            eta_inter = eta_inter_cfg / eta_sum
            eta_intra = eta_intra_cfg / eta_sum
        else:
            # Fallback robuste si cellules non disponibles.
            s_intra = torch.zeros_like(s_inter)
            eta_inter = 1.0
            eta_intra = 0.0

        rel_raw = eta_inter * s_inter + eta_intra * s_intra  # (B, C)

        # z-score intra-image puis sigmoid: omega_rel dans [0,1].
        mu = rel_raw.mean(dim=1, keepdim=True)
        sigma = rel_raw.std(dim=1, unbiased=False, keepdim=True)
        rel_z = (rel_raw - mu) / (sigma + eps)
        omega_rel = torch.sigmoid(rel_z)

        if omega_mode == "ones":
            Omega = torch.ones_like(beta_w)
        elif omega_mode == "beta":
            Omega = beta_w  # Use beta_w which may be Gamma-aggregated or Beta
        elif omega_mode == "rel":
            Omega = omega_rel
        else:
            Omega = beta_w * omega_rel
        return Omega

    @staticmethod
    def compute_hierarchical_relation_vector(
        Ehat: torch.Tensor,
        Beta: torch.Tensor,
        Uhat: torch.Tensor,
        Gamma: torch.Tensor,
        cfg,
    ) -> torch.Tensor:
        """
        Construit un vecteur relationnel hiérarchique en R^D (même dimension que global_feat).
        """
        eps = float(getattr(cfg, "relvec_eps", 1e-6))
        l_inter_h = float(getattr(cfg, "relvec_lambda_inter_had", 1.0))
        l_inter_d = float(getattr(cfg, "relvec_lambda_inter_diff", 0.5))
        l_intra_h = float(getattr(cfg, "relvec_mu_intra_had", 1.0))
        l_intra_d = float(getattr(cfg, "relvec_mu_intra_diff", 0.5))
        eta_inter = float(getattr(cfg, "relvec_eta_inter", 0.5))
        eta_intra = float(getattr(cfg, "relvec_eta_intra", 0.5))
        normalize_out = bool(getattr(cfg, "relvec_normalize", True))
        detach_weights = bool(getattr(cfg, "relvec_detach_weights", True))

        E = F.normalize(Ehat.float(), dim=2)
        U = F.normalize(Uhat.float(), dim=3)
        _, C, _ = E.shape
        K = U.shape[2]

        beta_w = Beta.float().detach() if detach_weights else Beta.float()
        gamma_w = Gamma.float().detach() if detach_weights else Gamma.float()

        zero = E.sum(dim=1) * 0.0

        # Inter-stripes: somme pondérée sur paires (i, j), i < j.
        if C > 1:
            ei = E.unsqueeze(2)  # (B, C, 1, D)
            ej = E.unsqueeze(1)  # (B, 1, C, D)
            rel_inter = l_inter_h * (ei * ej) + l_inter_d * (ei - ej).abs()

            w_inter = beta_w.unsqueeze(2) * beta_w.unsqueeze(1)  # (B, C, C)
            tri_mask = torch.triu(torch.ones(C, C, device=E.device, dtype=E.dtype), diagonal=1)
            w_inter = w_inter * tri_mask.unsqueeze(0)

            num_inter = (rel_inter * w_inter.unsqueeze(-1)).sum(dim=(1, 2))
            den_inter = w_inter.sum(dim=(1, 2)).clamp_min(eps).unsqueeze(-1)
            r_inter = num_inter / den_inter
        else:
            r_inter = zero

        # Intra-stripe: cellules adjacentes (k, k+1) par stripe.
        if K > 1:
            uk = U[:, :, :-1, :]
            uk1 = U[:, :, 1:, :]
            rel_intra = l_intra_h * (uk * uk1) + l_intra_d * (uk - uk1).abs()

            w_intra = gamma_w[:, :, :-1] * gamma_w[:, :, 1:]  # (B, C, K-1)
            num_intra = (rel_intra * w_intra.unsqueeze(-1)).sum(dim=(1, 2))
            den_intra = w_intra.sum(dim=(1, 2)).clamp_min(eps).unsqueeze(-1)
            r_intra = num_intra / den_intra
        else:
            r_intra = zero

        r_rel = eta_inter * r_inter + eta_intra * r_intra
        if normalize_out:
            r_rel = F.normalize(r_rel, dim=1)
        return r_rel

    @staticmethod
    def fuse_global_with_relvec(
        global_feat: torch.Tensor,
        rel_vec: torch.Tensor,
        alpha: float = 0.5,
        normalize_out: bool = True,
    ) -> torch.Tensor:
        """
        Fusion vectorielle en R^D: f = alpha*g + (1-alpha)*r_rel.
        """
        fused = alpha * global_feat + (1.0 - alpha) * rel_vec
        if normalize_out:
            fused = F.normalize(fused, dim=1)
        return fused
    
    def get_stripes(
        self,
        x: torch.Tensor,
        beta_cfg,
        orientation_vec: Optional[torch.Tensor] = None,
        cache_role: str = "query",
        return_complex_embeddings: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Utility method: obtain all DOC components in a single forward pass.

        Returns:
            global_feat: (B, D)
            Ehat: (B, C, D)
            Beta: (B, C)
            Omega: (B, C)
            Z (optional): (B, C, D) complex if return_complex_embeddings=True and available
        """
        _, global_feat, feat_map = self.forward_global(x)
        Z: Optional[torch.Tensor] = None
        
        need_cells_for_omega = str(getattr(beta_cfg, "omega_mode", "beta_rel")).lower() in {"rel", "beta_rel"}
        if need_cells_for_omega:
            if return_complex_embeddings:
                Ehat, Beta, Uhat, Gamma, Z = self.extract_stripes_adaptive(
                    feat_map,
                    beta_cfg.C_stripes,
                    beta_cfg.R_rows,
                    beta_cfg,
                    orientation_vec=orientation_vec,
                    return_cells=True,
                    return_complex_embeddings=True,
                    cache_role=cache_role,
                )
            else:
                Ehat, Beta, Uhat, Gamma = self.extract_stripes_adaptive(
                    feat_map,
                    beta_cfg.C_stripes,
                    beta_cfg.R_rows,
                    beta_cfg,
                    orientation_vec=orientation_vec,
                    return_cells=True,
                    cache_role=cache_role,
                )
            Omega = self.compute_omega(Ehat, Beta, beta_cfg.omega_mode, Uhat=Uhat, Gamma=Gamma, cfg=beta_cfg)
        else:
            if return_complex_embeddings:
                Ehat, Beta, Z = self.extract_stripes_adaptive(
                    feat_map,
                    beta_cfg.C_stripes,
                    beta_cfg.R_rows,
                    beta_cfg,
                    orientation_vec=orientation_vec,
                    return_complex_embeddings=True,
                    cache_role=cache_role,
                )
            else:
                Ehat, Beta = self.extract_stripes_adaptive(
                    feat_map,
                    beta_cfg.C_stripes,
                    beta_cfg.R_rows,
                    beta_cfg,
                    orientation_vec=orientation_vec,
                    cache_role=cache_role,
                )
            Omega = self.compute_omega(Ehat, Beta, beta_cfg.omega_mode, cfg=beta_cfg)

        if return_complex_embeddings:
            return global_feat, Ehat, Beta, Omega, Z
        return global_feat, Ehat, Beta, Omega


def test_model():
    """Quick smoke test for DOCModel."""
    if not TORCHREID_AVAILABLE:
        print("torchreid not available, test skipped")
        return
    
    print("Test DOCModel...")
    
    # Config test
    class DummyCFG:
        C_stripes = 4
        R_rows = 4
        beta_energy = "l2"
        beta_thresh = 0.45
        beta_slope = 0.125
        omega_mode = "beta_rel"
    
    cfg = DummyCFG()
    
    # Create model
    model = DOCModel("osnet_x1_0", num_classes=100, pretrained=False)
    model.eval()
    
    # Test forward
    x = torch.randn(2, 3, 256, 128)
    
    with torch.no_grad():
        logits, global_feat, feat_map = model.forward_global(x)
        print(f"  Logits: {logits.shape}")
        print(f"  Global features: {global_feat.shape}")
        print(f"  Feature map: {feat_map.shape}")
        
        # Test extraction stripes
        Ehat, Beta, Uhat, Gamma = DOCModel.extract_stripes(
            feat_map, cfg.C_stripes, cfg.R_rows, cfg, return_cells=True
        )
        print(f"  Ehat: {Ehat.shape}")
        print(f"  Beta: {Beta.shape}")
        
        # Test omega
        Omega = DOCModel.compute_omega(Ehat, Beta, cfg.omega_mode, Uhat=Uhat, Gamma=Gamma, cfg=cfg)
        print(f"  Omega: {Omega.shape}")
        
        # Test full method (no external orientation)
        g, e, b, o = model.get_stripes(x, cfg)
        print("  get_stripes: OK")

        # Test full method with external orientation vector
        orientation_vec = torch.randn(x.size(0), 2)
        orientation_vec = F.normalize(orientation_vec, dim=1)
        g, e, b, o = model.get_stripes(x, cfg, orientation_vec=orientation_vec)
        print("  get_stripes with external orientation_vec: OK")

        # Test complex output path with external orientation
        g, e, b, o, z = model.get_stripes(
            x,
            cfg,
            orientation_vec=orientation_vec,
            return_complex_embeddings=True,
        )
        print(f"  Complex Z: {z.shape if z is not None else None}")
    
    model.close()
    print("Test passed!")


if __name__ == "__main__":
    test_model()
