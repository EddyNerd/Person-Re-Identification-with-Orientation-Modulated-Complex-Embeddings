from __future__ import annotations

from typing import Tuple

import torch


def _normalize_complex_parts(
    real_part: torch.Tensor,
    imag_part: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real_part = real_part.float()
    imag_part = imag_part.float()
    norm = torch.sqrt(
        real_part.square().sum(dim=-1, keepdim=True)
        + imag_part.square().sum(dim=-1, keepdim=True)
    ).clamp_min(eps)
    return real_part / norm, imag_part / norm


def complex_hermitian_interaction(
    q_mod_abs: torch.Tensor,
    q_mod_rel: torch.Tensor,
    g_mod_abs: torch.Tensor,
    g_mod_rel: torch.Tensor,
    *,
    normalize_inputs: bool = True,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real, imaginary and magnitude tensors of the Hermitian product."""
    if normalize_inputs:
        q_mod_abs, q_mod_rel = _normalize_complex_parts(q_mod_abs, q_mod_rel, eps=eps)
        g_mod_abs, g_mod_rel = _normalize_complex_parts(g_mod_abs, g_mod_rel, eps=eps)
    else:
        q_mod_abs = q_mod_abs.float()
        q_mod_rel = q_mod_rel.float()
        g_mod_abs = g_mod_abs.float()
        g_mod_rel = g_mod_rel.float()

    real = torch.einsum("qcd,gkd->qgck", q_mod_abs, g_mod_abs) + torch.einsum("qcd,gkd->qgck", q_mod_rel, g_mod_rel)
    imag = torch.einsum("qcd,gkd->qgck", q_mod_abs, g_mod_rel) - torch.einsum("qcd,gkd->qgck", q_mod_rel, g_mod_abs)
    magnitude = torch.sqrt(real.square() + imag.square() + float(eps))
    return real, imag, magnitude


def complex_hermitian_similarity(
    q_mod_abs: torch.Tensor,
    q_mod_rel: torch.Tensor,
    g_mod_abs: torch.Tensor,
    g_mod_rel: torch.Tensor,
    *,
    normalize_inputs: bool = True,
    crg_lambda: float = 0.5,
    eps: float = 1e-9,
) -> torch.Tensor:
    real, imag, _ = complex_hermitian_interaction(
        q_mod_abs,
        q_mod_rel,
        g_mod_abs,
        g_mod_rel,
        normalize_inputs=normalize_inputs,
        eps=eps,
    )
    real_abs = real.abs()
    imag_abs = imag.abs()
    gate = torch.exp(-float(crg_lambda) * imag_abs / (real_abs + float(eps)))
    return (real_abs * gate).clamp(0.0, 1.0)


def complex_hermitian_cost(
    q_mod_abs: torch.Tensor,
    q_mod_rel: torch.Tensor,
    g_mod_abs: torch.Tensor,
    g_mod_rel: torch.Tensor,
    *,
    normalize_inputs: bool = True,
    crg_lambda: float = 0.5,
    eps: float = 1e-9,
) -> torch.Tensor:
    sim = complex_hermitian_similarity(
        q_mod_abs,
        q_mod_rel,
        g_mod_abs,
        g_mod_rel,
        normalize_inputs=normalize_inputs,
        crg_lambda=crg_lambda,
        eps=eps,
    )
    return (1.0 - sim).clamp(0.0, 2.0)


def complex_set_to_set_similarity(
    Eq_mod_abs: torch.Tensor,
    Eq_mod_rel: torch.Tensor,
    Oq: torch.Tensor,
    Eg_mod_abs: torch.Tensor,
    Eg_mod_rel: torch.Tensor,
    Og: torch.Tensor,
    *,
    temp: float = 0.15,
    use_wp: bool = True,
    use_stripe_ot: bool = False,
    ot_epsilon: float = 0.1,
    ot_num_iters: int = 100,
    uniform_ot_marginals: bool = False,
    ot_margi_eps: float = 1e-9,
    crg_lambda: float = 0.5,
) -> torch.Tensor:
    """Complex CRG Hermitian stripe matching with optional Sinkhorn OT."""
    sim = complex_hermitian_similarity(
        Eq_mod_abs,
        Eq_mod_rel,
        Eg_mod_abs,
        Eg_mod_rel,
        crg_lambda=crg_lambda,
        eps=ot_margi_eps,
    )

    if use_stripe_ot:
        from loss_optimized import sinkhorn_algorithm  # type: ignore

        Bq, Bg, C, _ = sim.shape
        Cost = (1.0 - sim).clamp(0.0, 2.0)

        if uniform_ot_marginals:
            a = torch.ones(Bq, C, device=Eq_mod_abs.device, dtype=Eq_mod_abs.dtype) / C
            b = torch.ones(Bg, C, device=Eg_mod_abs.device, dtype=Eg_mod_abs.dtype) / C
        else:
            a = Oq.float() / (Oq.float().sum(dim=1, keepdim=True) + ot_margi_eps)
            b = Og.float() / (Og.float().sum(dim=1, keepdim=True) + ot_margi_eps)

        Cost_reshaped = Cost.reshape(Bq * Bg, C, C)
        a_expanded = a.repeat_interleave(Bg, dim=0)
        b_expanded = b.repeat(Bq, 1)

        P = sinkhorn_algorithm(
            Cost_reshaped,
            a_expanded,
            b_expanded,
            epsilon=ot_epsilon,
            num_iterations=ot_num_iters,
            eps=ot_margi_eps,
        )
        D_ot = (P * Cost_reshaped).sum(dim=(1, 2)).reshape(Bq, Bg)
        return 1.0 - D_ot.clamp(0.0, 2.0)

    if use_wp:
        wp = (Oq.unsqueeze(1).unsqueeze(-1) * Og.unsqueeze(0).unsqueeze(2)).clamp_min(1e-6)
        log_wp = torch.log(wp + 1e-12)
        wp_agg = wp
    else:
        log_wp = 0.0
        wp_agg = torch.ones_like(sim)

    att_k = torch.softmax(sim / temp + log_wp, dim=3)
    score_1 = (att_k * sim * wp_agg).sum(dim=3) / (att_k * wp_agg).sum(dim=3).clamp_min(1e-6)

    att_c = torch.softmax(sim / temp + log_wp, dim=2)
    score_2 = (att_c * sim * wp_agg).sum(dim=2) / (att_c * wp_agg).sum(dim=2).clamp_min(1e-6)

    return 0.5 * (score_1.mean(dim=2) + score_2.mean(dim=2))
