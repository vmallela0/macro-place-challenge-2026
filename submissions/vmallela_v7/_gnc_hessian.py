"""Graduated Non-Convexity (GNC) extension to Hessian escape — Cand #2.

Reference: Blake-Zisserman 1987, "Visual Reconstruction" (Mumford-Shah
GNC). Mokhtarian-Mackworth 1992, "Scale-based shape description."

The core idea: at larger smoothing scale σ, the cost landscape's
Hessian eigenvectors capture LARGER-SCALE structural features. Smaller
σ (sharper cost) gives finer-scale eigvecs. By combining escape
directions from multiple scales, we get a richer set of "escape
candidates" — some moves are best-described at coarse scale (which
basin to go to), others at fine scale (where exactly within a basin).

Math
====
Define a family of smoothed cost functions f_τ(x) where τ is the
smoothing scale. For our problem:
    f_τ(x) = HPWL_LSE(x; τ_LSE=τ) + 0.5 · CVaR_density(x; μ=τ)

Lower τ = wider/smoother. Higher τ = sharper/closer to original.

Compute Hessian eigvec at each τ:
    H_τ = ∂²f_τ / ∂x²
    v_τ = eigvec_min(H_τ)

These v_τ generally point in DIFFERENT directions (different scales of
landscape structure). Test all of them as escape candidates.

Why this might add value beyond single-scale Hessian
----------------------------------------------------
The Hessian-escape baseline uses a fixed τ (50 for LSE, 100 for
softplus). That captures one specific scale of the cost landscape.
GNC says: the GLOBAL structure is best seen at COARSE scale, the
LOCAL structure at FINE scale. For escaping a basin, one of those
scales might encode the right escape direction better than the other.

Empirically on placement:
- Coarse τ (e.g., τ=10): the eigvec might say "rotate the whole
  cluster of softs by 30°"
- Fine τ (e.g., τ=200): the eigvec might say "swap macros 47 and 891"
- Different scales → different escape attempts.
"""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import torch


def gnc_hessian_escape_multi_scale(
    macro_pos: torch.Tensor,
    proxy_call_factory: Callable[[float, float], Callable],
    *,
    tau_scales: List[Tuple[float, float]] = ((10.0, 50.0), (50.0, 100.0),
                                                (200.0, 200.0)),
    step_sizes: list = (0.02, 0.05),
    canvas_diag: float = 1.0,
    n_lanczos_iters: int = 50,
    n_hard: int = 0,
    soft_only_perturb: bool = True,
    verbose: bool = False,
) -> tuple[list, dict]:
    """Compute Hessian min-eigvec at MULTIPLE smoothing scales; emit
    candidate perturbations from all.

    Parameters
    ----------
    macro_pos : (n, 2) torch tensor (current placement)
    proxy_call_factory : (tau_lse, mu_softplus) → callable that returns
        smooth-surrogate scalar loss given macro_pos. Lets us swap
        smoothing scales without rebuilding the closure.
    tau_scales : list of (tau_lse, mu_softplus) pairs to evaluate at.
        Default explores: (10, 50) coarse, (50, 100) medium,
        (200, 200) fine.
    step_sizes : as for single-scale Hessian.

    Returns
    -------
    candidates : list of (label, perturbed_pos_np) — label includes the
        scale tag for tracking which scale produced it.
    diagnostics : dict with per-scale lambda_min and v_norm.
    """
    from _hessian_escape import hessian_min_eigvec
    n_total = macro_pos.shape[0]

    all_candidates = []
    diagnostics = {"per_scale": []}

    for i, (tau_lse, mu_softplus) in enumerate(tau_scales):
        proxy_call = proxy_call_factory(tau_lse, mu_softplus)
        if verbose:
            print(f"  [gnc] scale {i}: τ_lse={tau_lse}, μ_softplus="
                  f"{mu_softplus}", flush=True)
        try:
            lam_min, v_min = hessian_min_eigvec(
                proxy_call, macro_pos,
                n_lanczos_iters=n_lanczos_iters,
                verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"    [gnc] scale {i} eigsh failed: {e}", flush=True)
            continue

        v_min_xy = v_min.reshape(n_total, 2)
        v_norm = np.linalg.norm(v_min_xy)
        if v_norm < 1e-12:
            diagnostics["per_scale"].append(
                {"tau_lse": tau_lse, "mu": mu_softplus,
                 "lambda_min": lam_min, "v_norm": v_norm,
                 "warn": "degenerate"})
            continue
        v_min_xy = v_min_xy / v_norm
        if soft_only_perturb and n_hard > 0:
            v_min_xy[:n_hard] = 0.0
            v_norm_post = np.linalg.norm(v_min_xy)
            if v_norm_post > 1e-12:
                v_min_xy = v_min_xy / v_norm_post

        diagnostics["per_scale"].append({
            "tau_lse": tau_lse, "mu": mu_softplus,
            "lambda_min": lam_min, "v_norm": float(v_norm)})

        base = macro_pos.detach().cpu().numpy()
        for s in step_sizes:
            label = f"τ{int(tau_lse)}μ{int(mu_softplus)}_s{s:+.3f}"
            delta = s * canvas_diag * v_min_xy
            all_candidates.append((label, base + delta))
            label_neg = f"τ{int(tau_lse)}μ{int(mu_softplus)}_s{-s:+.3f}"
            all_candidates.append((label_neg, base - delta))

    return all_candidates, diagnostics
