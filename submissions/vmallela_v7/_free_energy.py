"""Free-energy / Gaussian-smoothed objective — zeus B7.

Variational thermodynamics: minimize
    F(x) = E_{ε ~ N(0, σ²·I)}[U(x + ε)]
instead of U(x). This is the free energy at temperature T = σ² for a
Gaussian variational family q_x(·) = N(x, σ²·I), under the Laplace
approximation. The mode of F is at the "flattest" basin — wide,
low-curvature regions of U(x), not just point minima.

Why this should help here
=========================
The exact proxy is non-smooth (discrete placement). Our smooth surrogate
approximates it but has spurious local minima — sharp basins that look
attractive to gradient descent but evaporate when you discretize. Free
energy smoothing penalizes sharp minima: a basin of width w² has
F(x_center) ≈ U(x_center) + (σ²/w²)·(extra cost from sharpness).

For a quadratic well U(x) ≈ ½ λ ||x-x*||²:
    F(x*) = U(x*) + (σ²/2) · λ_max(∇²U)
Sharper wells (large λ) cost more F. Flat wells survive.

Also related: this is exactly what Sharpness-Aware Minimization (SAM)
implements in deep learning, where it improved generalization. Same
principle here: a "good" placement should be robust to small
perturbations, not perched on a spike.

Math
====
F(x) = E_ε U(x+ε) ≈ U(x) + (σ²/2) tr(∇²U(x)) + O(σ⁴)
The σ²·tr(H) regularizer is the dominant high-σ term — it pulls toward
small-trace Hessians, i.e., low average curvature.

Implementation
==============
Monte Carlo with K samples:
    F̂(x) = (1/K) Σ_k U(x + ε_k),     ε_k ~ N(0, σ²·I)
∇F̂(x) = (1/K) Σ_k ∇U(x + ε_k)  via autograd reparameterization.

Pros vs cons
------------
+ One-line wrapper — drops into any caller of smooth_proxy_call.
+ Symmetric: positive and negative ε are equally weighted.
+ σ is a knob: σ→0 recovers raw U; σ large washes everything.
+ Composes with all other bets (HMC, DMC, RUDY, etc.).
- K× cost per evaluation. K=4 is the practical sweet spot.
- Need σ small enough that x+ε stays in the feasible region.

Usage
-----
    raw_proxy = smooth_proxy_for_v7(...)
    fe_proxy  = make_free_energy_proxy(raw_proxy, sigma=5.0, K=4, seed=0)
    # fe_proxy can replace raw_proxy in Lanczos, HMC, DMC, etc.
"""

from __future__ import annotations
import torch


def make_free_energy_proxy(smooth_proxy_call,
                            sigma: float = 5.0,
                            K: int = 4,
                            seed: int = 0,
                            soft_only: bool = True,
                            n_hard: int = 0):
    """Wrap smooth_proxy_call(x_tensor) into F̂(x) = mean_k U(x + ε_k).

    Each call uses a fresh batch of K Gaussian samples drawn from a
    rejection-free Gaussian (no clamping — assume σ is small enough
    to stay near canvas).

    Parameters
    ----------
    smooth_proxy_call : closure x_tensor (n,2) → scalar.
    sigma : Gaussian σ in microns. Default 5.0 = ~1% of canvas diag
        for typical IBM benches.
    K : Monte Carlo sample count. Default 4 — small enough to be
        affordable, large enough to suppress noise on the gradient.
    seed : initial RNG seed. Internal state is incremented per call
        so successive calls get fresh noise.
    soft_only : if True, don't perturb the first n_hard rows.
    n_hard : leading rows to leave unperturbed.

    Returns
    -------
    A closure F̂(x) callable like smooth_proxy_call.
    """
    state = {"counter": int(seed)}

    def fe_call(x_tensor: torch.Tensor) -> torch.Tensor:
        if sigma <= 0 or K <= 1:
            return smooth_proxy_call(x_tensor)
        device = x_tensor.device
        dtype = x_tensor.dtype
        n_total = x_tensor.shape[0]
        # Generate K samples (deterministic given counter) so that
        # repeated calls at the same x are reproducible-ish across
        # eigvec / line-search probes within one Lanczos iter.
        g = torch.Generator(device=device)
        g.manual_seed(int(state["counter"]) & 0x7FFFFFFF)
        # We DON'T increment counter per-call to keep noise frozen across
        # autograd; instead, caller is expected to bump the counter
        # between Lanczos iters. This is critical: if the noise varies
        # within a single Lanczos call, the Hessian/grad probes see
        # different surrogates and Lanczos fails.
        losses = []
        for k in range(int(K)):
            eps = torch.randn(n_total, 2, generator=g,
                               device=device, dtype=dtype) * float(sigma)
            if soft_only and n_hard > 0:
                eps[:n_hard, :] = 0.0
            # Reparameterization: gradient flows through x_tensor, not eps.
            U_k = smooth_proxy_call(x_tensor + eps.detach())
            losses.append(U_k)
        return torch.stack(losses).mean()

    # Expose a way to bump the seed between iterations.
    fe_call.bump_seed = lambda: state.update(counter=state["counter"] + 1)
    return fe_call
