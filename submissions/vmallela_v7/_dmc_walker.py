"""Diffusion Monte Carlo escape — zeus B5.

Quantum-inspired alternative to subspace HMC.

Why this exists
===============
Subspace HMC produces T deterministic trajectories from random momenta in
the K-dim negative-eigenspace, each a smooth Hamiltonian flow. It works
well when the basin is locally "well-curved" but struggles when:
  - the basin has narrow corridors (HMC trajectories overshoot)
  - U(x) has cusps / kinks where ∇U is large but the basin is narrow
  - the K-dim subspace doesn't contain the true escape direction

Diffusion Monte Carlo (DMC) replaces deterministic dynamics with a
walker-based importance sampling of the imaginary-time Schrödinger
equation. Key differences:
  • Each walker is a placement; we maintain N=32-128 walkers.
  • Drift = ½ · ∇log ψ_T(x) where ψ_T(x) ∝ exp(-β U(x)/2) is the trial
    wavefunction. So drift = -½ β ∇U — half the gradient step.
  • Diffusion = Brownian motion with σ = sqrt(τ) (τ = imaginary-time step).
  • Branching: each walker carries a weight w_t = exp(-τ (U(x) - E_T)).
    walkers with w > 1 split (replicate), w < 1 walkers can die.
  • E_T (trial energy) is adapted to keep the walker count near a target.

Why this should help here
-------------------------
DMC samples from the GROUND-STATE distribution of U, not just one
trajectory. At the end, walkers concentrate near minima of U.
Unlike HMC, DMC doesn't need a smooth surrogate — branching handles
rugged landscapes. Unlike SA, DMC's population covers diverse basins
in parallel; we extract the best basin's centroid as the candidate.

Math
====
Imaginary-time Schrödinger: ∂ψ/∂t = -(H - E_T) ψ where H = -½∇² + U.
With trial wavefunction ψ_T(x) = exp(-S(x)/2), let f(x,t) = ψ(x,t) ψ_T(x).
Then f evolves as:
    ∂f/∂t = -½ ∇²f + ∇·(b f) + (E_T - U) f
where b(x) = -∇S/2 is the drift. We discretize via Trotter:
    x' = x + b(x)·τ + N(0, τ·I)
    w' = w · exp(-τ · (U(x') - E_T))     [growth/decay weight]
and resample walkers proportional to weight every step.

Convergence
-----------
As τ→0 and N→∞, walker density → ψ_T(x)² ∝ exp(-S(x)) = exp(-β U(x)).
For our use (escape from saddle), one DMC sweep with τ small enough is
already useful: the diffusion lets walkers cross small barriers.

Failure modes
-------------
  - Walker collapse: if E_T is wrong, all walkers die or one walker
    dominates. Adapt E_T via E_T ← E_T - α log(N_active/N_target).
  - Drift instability: for very steep U, x' shoots out. Cap step.
  - Bias from finite τ: O(τ²) bias is OK at our scale.

Usage
-----
Provide a population of K starting placements (we typically start from
the current x_0 perturbed by small Gaussian) and the smooth_proxy_call.
Returns a list of (label, pos_np) candidates: each walker's endpoint.
"""

from __future__ import annotations
import time
import numpy as np
import torch


def diffusion_monte_carlo_candidates(
    macro_pos: torch.Tensor,             # (n_total, 2) base placement
    smooth_proxy_call,                   # callable: x_tensor → scalar U
    *,
    n_walkers: int = 32,
    n_steps: int = 40,                   # imaginary-time steps
    tau: float = 0.5,                    # time step (raw position units²)
    beta: float = 1.0,                   # inverse temperature; controls drift strength
    init_jitter: float = 5.0,            # microns; std of initial walker scatter
    n_hard: int = 0,                     # don't move hard macros
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    e_t_adapt_alpha: float = 0.1,        # E_T feedback gain
    step_cap_microns: float = 50.0,      # per-step displacement cap
    seed: int = 42,
    verbose: bool = False,
) -> tuple[list, dict]:
    """Run DMC for n_steps, return one candidate per surviving walker.

    Parameters
    ----------
    macro_pos : (n_total, 2) base placement (numpy or torch).
    smooth_proxy_call : closure x_tensor → scalar U_smooth.
    n_walkers : initial population size. Adapted via branching.
    n_steps : number of imaginary-time steps (each is drift+diffuse+branch).
    tau : time step. Drift magnitude is β·τ·∇U. Larger τ = larger steps
        but more bias. Default 0.5.
    beta : inverse temperature in the trial wavefunction.
    init_jitter : Gaussian σ for initial walker scatter from base.
    n_hard : leading rows that should NOT be perturbed.
    canvas_w/h : for clamping walkers inside canvas.
    e_t_adapt_alpha : feedback gain for E_T to maintain walker count.
    step_cap_microns : cap |Δx| per walker per step.

    Returns
    -------
    candidates : list of (label, pos_np) — one per surviving walker,
        sorted by U_final ascending.
    diag : dict of stats.
    """
    device = macro_pos.device
    dtype = macro_pos.dtype
    n_total = macro_pos.shape[0]
    base_np = macro_pos.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Initialize walkers around base with Gaussian jitter (soft macros only).
    walkers = np.tile(base_np[None, :, :], (n_walkers, 1, 1))
    if n_total - n_hard > 0:
        walkers[:, n_hard:, :] += rng.standard_normal(
            (n_walkers, n_total - n_hard, 2)) * float(init_jitter)
        # Clamp into canvas.
        walkers[:, :, 0] = np.clip(walkers[:, :, 0], 0.0, float(canvas_w))
        walkers[:, :, 1] = np.clip(walkers[:, :, 1], 0.0, float(canvas_h))
    weights = np.ones(n_walkers, dtype=np.float64)
    target_pop = n_walkers

    def _eval_grad(x_np):
        """Return (U, ∇U) for a single placement."""
        x_t = torch.tensor(x_np, dtype=dtype, device=device,
                            requires_grad=True)
        U_t = smooth_proxy_call(x_t)
        g_t = torch.autograd.grad(U_t, x_t)[0]
        return float(U_t.item()), g_t.detach().cpu().numpy()

    def _eval_batch(xs):
        """Evaluate U for a batch of walkers (no grad)."""
        with torch.no_grad():
            out = np.empty(len(xs), dtype=np.float64)
            for i, x_np in enumerate(xs):
                x_t = torch.tensor(x_np, dtype=dtype, device=device)
                out[i] = float(smooth_proxy_call(x_t).item())
            return out

    Us_init = _eval_batch(walkers)
    E_T = float(np.median(Us_init))   # initialize at median energy
    history_pop = [int(n_walkers)]
    history_ET = [E_T]
    history_U_mean = [float(Us_init.mean())]
    history_U_min = [float(Us_init.min())]

    for step in range(int(n_steps)):
        n_active = walkers.shape[0]
        if n_active == 0:
            break
        # Drift + diffuse, gradient per walker (sequential — most walkers
        # are O(few hundred macros); n_walkers loop is ~32 steps).
        new_walkers = np.empty_like(walkers)
        new_Us = np.empty(n_active, dtype=np.float64)
        for i in range(n_active):
            U_i, g_i = _eval_grad(walkers[i])
            drift = -0.5 * beta * tau * g_i
            noise = rng.standard_normal(g_i.shape) * np.sqrt(tau)
            if n_hard > 0:
                drift[:n_hard, :] = 0.0
                noise[:n_hard, :] = 0.0
            dx = drift + noise
            # Step cap (per walker).
            r = np.linalg.norm(dx)
            if r > step_cap_microns:
                dx = dx * (step_cap_microns / r)
            x_new = walkers[i] + dx
            x_new[:, 0] = np.clip(x_new[:, 0], 0.0, float(canvas_w))
            x_new[:, 1] = np.clip(x_new[:, 1], 0.0, float(canvas_h))
            new_walkers[i] = x_new
            new_Us[i] = U_i        # we use U(x_old) for the weight (consistent w/ DMC)
        # Update weights (use a stabilized form: log-domain, then renormalize).
        log_w_step = -tau * (new_Us - E_T)
        log_w = np.log(np.maximum(weights, 1e-30)) + log_w_step
        # Clamp to avoid runaway weights.
        log_w = np.clip(log_w, -20.0, 20.0)
        weights = np.exp(log_w)

        # Stochastic branching: number of copies for walker i = floor(w_i + U(0,1)).
        u = rng.random(n_active)
        n_copies = np.floor(weights + u).astype(np.int64)
        # Build new population.
        n_total_new = int(n_copies.sum())
        if n_total_new == 0:
            # Population collapse — restart from best-K and shrink E_T.
            k_keep = min(target_pop // 2, n_active)
            order = np.argsort(new_Us)[:k_keep]
            walkers = new_walkers[order]
            weights = np.ones(k_keep, dtype=np.float64)
            E_T = float(new_Us[order].min())
            history_pop.append(int(k_keep))
            history_ET.append(E_T)
            history_U_mean.append(float(new_Us[order].mean()))
            history_U_min.append(float(new_Us[order].min()))
            continue
        new_pop = np.empty((n_total_new, n_total, 2), dtype=np.float64)
        new_pop_U = np.empty(n_total_new, dtype=np.float64)
        idx = 0
        for i in range(n_active):
            c = int(n_copies[i])
            if c > 0:
                new_pop[idx:idx+c] = new_walkers[i]
                new_pop_U[idx:idx+c] = new_Us[i]
                idx += c
        # Cap population at 4×target to avoid blowup.
        cap = 4 * target_pop
        if n_total_new > cap:
            # Keep the cap lowest-U walkers.
            order = np.argsort(new_pop_U)[:cap]
            new_pop = new_pop[order]
            new_pop_U = new_pop_U[order]
            n_total_new = cap
        walkers = new_pop
        weights = np.ones(n_total_new, dtype=np.float64)
        # Adapt E_T to drive population toward target.
        E_T = E_T - e_t_adapt_alpha * np.log(
            max(n_total_new, 1) / max(target_pop, 1)) / max(tau, 1e-9)
        history_pop.append(int(n_total_new))
        history_ET.append(float(E_T))
        history_U_mean.append(float(new_pop_U.mean()))
        history_U_min.append(float(new_pop_U.min()))

    # Final eval — return all surviving walkers sorted by U.
    if walkers.shape[0] == 0:
        diag = {
            "method": "dmc",
            "warn": "all walkers died",
            "history_pop": history_pop,
            "history_ET": history_ET,
            "history_U_mean": history_U_mean,
            "history_U_min": history_U_min,
            "wall_s": time.time() - t0,
        }
        return [], diag
    Us_final = _eval_batch(walkers)
    order = np.argsort(Us_final)
    walkers = walkers[order]
    Us_final = Us_final[order]
    # De-duplicate: drop walkers within 1 micron of each other (replicas).
    keep_mask = np.ones(len(walkers), dtype=bool)
    for i in range(1, len(walkers)):
        if not keep_mask[i]:
            continue
        for j in range(i):
            if not keep_mask[j]:
                continue
            disp = walkers[i] - walkers[j]
            if np.max(np.linalg.norm(disp, axis=1)) < 1.0:
                keep_mask[i] = False
                break
    walkers = walkers[keep_mask]
    Us_final = Us_final[keep_mask]

    candidates = []
    for i, x_np in enumerate(walkers):
        label = f"dmc_w{i:02d}_U{Us_final[i]:+.4f}"
        candidates.append((label, x_np.astype(np.float64)))

    diag = {
        "method": "dmc",
        "n_walkers_init": int(target_pop),
        "n_walkers_final": int(walkers.shape[0]),
        "tau": float(tau),
        "beta": float(beta),
        "n_steps": int(n_steps),
        "U_final_min": float(Us_final.min()) if len(Us_final) else None,
        "U_final_median": float(np.median(Us_final)) if len(Us_final) else None,
        "history_pop": history_pop,
        "history_ET": history_ET,
        "history_U_mean": history_U_mean,
        "history_U_min": history_U_min,
        "wall_s": time.time() - t0,
    }
    if verbose:
        print(f"    [dmc] N0={target_pop} steps={n_steps} τ={tau:.2f} "
              f"survivors={walkers.shape[0]} "
              f"U_min={diag['U_final_min']:.4f} "
              f"({diag['wall_s']:.1f}s)", flush=True)
    return candidates, diag
