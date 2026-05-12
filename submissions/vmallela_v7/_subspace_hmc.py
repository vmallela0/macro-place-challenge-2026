"""Subspace Hamiltonian Monte Carlo escape — zeus addition.

Why this exists
===============
The existing post-Lanczos escape options in `_hessian_escape.py` are:
  - `adaptive_topk_candidates` : backtracking line search per eigvec.
    Yields ONE candidate per eigvec, at the single best step along ±v_j.
  - `kdim_trust_region_step`   : analytic K-dim Newton step.
    Yields ONE candidate from the quadratic-model minimum in
    span{v_1..v_K}.

Both are deterministic. When the Hessian eigvec direction is computed
against a stale or partially-wrong surrogate (ITERATIONS.md Iter 4d,
7), a single deterministic step lands in the locally-improved-but-
globally-stale state — and the strict-improvement gate then rejects it.
Symptom: 6-bench regression band at ±0.005 around verified.

Subspace HMC introduces RANDOMNESS in the K-dim subspace:
  - Sample p ~ 𝒩(0, M=|Λ_K|)         (Hessian-metric momentum prior)
  - Leapfrog L steps on H(a, p) = ½ p·M⁻¹p + U(x_0 + V a)
  - Validate exact proxy at endpoint
  - Strict-improvement gate; restart with fresh momentum

Math derivation
===============
We restrict motion to span{v_1, ..., v_K} where {(λ_j, v_j)} are the
K smallest Hessian eigenpairs. Let V = [v_1 | ... | v_K] ∈ ℝ^(2N × K),
columns orthonormal. The K-dim coordinates a ∈ ℝ^K give x = x_0 + V·a.

Hamiltonian on (a, p) ∈ ℝ^K × ℝ^K with mass M = diag(|λ_j| ∨ ε):
    H(a, p) = ½ p^T M⁻¹ p + U_smooth(x_0 + V a)

Leapfrog discretization (symplectic, volume-preserving):
    p_{t+1/2} = p_t - (h/2) · ∇_a U(a_t)
    a_{t+1}   = a_t + h · M⁻¹ · p_{t+1/2}
    p_{t+1}   = p_{t+1/2} - (h/2) · ∇_a U(a_{t+1})

Chain rule: ∇_a U = V^T ∇_x U_smooth(x_0 + V a). PyTorch autograd
computes this in one backward pass per step by tracking a → x → U.

Why subspace-HMC over plain HMC?
- Plain HMC samples p ~ 𝒩(0, I_{2N}) — every direction equally
  likely. Most directions are HIGH-CURVATURE (the placement has many
  stiff degrees of freedom). Random kicks in stiff directions =
  energy-wasting bounces against the well walls.
- Subspace HMC concentrates exploration on the LOW-CURVATURE
  directions (smallest λ_j). These are the manifold's "soft modes"
  — the natural axes for crossing into nearby basins.
- The Hessian-metric mass M = |Λ_K| further preconditions: directions
  with larger |λ| have lower velocity. Mixing time governed by
  1/cond(M), not 1/spectral_gap (Patterson-Teh 2013).

Why strict-improvement gate replaces Metropolis-Hastings?
- We are doing OPTIMIZATION, not sampling. We want argmin U_exact.
- The smooth surrogate diverges from U_exact (frozen routing, CVaR
  smoothing, etc.). MH on U_smooth would converge to the smooth-
  surrogate posterior — biased relative to U_exact.
- Strict gate against U_exact ensures every accepted move is a real
  exact-proxy improvement. The smooth gradient is reduced to a
  PROPOSAL mechanism, not a target. Correctness is invariant to
  surrogate quality (worse surrogate = lower acceptance, not regression).

Cost
====
Per trajectory: L autograd backward passes through smooth_proxy_call.
For ibm15-scale (~6 k pins, ~1.5 k macros), one backward ≈ 30 ms on
32-thread CPU. L=16, 100 trajectories: 100 × 16 × 30 ms ≈ 50 s,
well under the 1000 s Hessian budget. K-dim projections are O(K · 2N)
= O(N) per step — negligible.

Validation: every trajectory's endpoint requires ONE exact-proxy
evaluation (~100 ms in PlacementCost). 100 trajectories × 100 ms = 10 s.

Sanity test
===========
At K=1 with deterministic momentum (rng seed fixed), subspace HMC
should reproduce the existing line-search behavior — same eigvec
direction, same step (up to leapfrog discretization). See
tests/test_subspace_hmc.py for this validation.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def subspace_hmc_candidates(
    macro_pos: torch.Tensor,             # (n_total, 2) current state
    smooth_proxy_call,                   # callable: x_tensor → scalar U
    eigvals: np.ndarray,                 # (K,) Hessian eigenvalues
    eigvecs: np.ndarray,                 # (2 n_total, K) eigenvectors (cols)
    *,
    n_trajectories: int = 32,
    n_leapfrog: int = 12,
    step_size: float = 0.5,              # leapfrog stepsize (raw units)
    canvas_diag: float = 1.0,
    mass_floor: float = 1e-3,
    n_hard: int = 0,
    soft_only: bool = True,
    seed: int = 42,
    max_total_step_canvas: float = 0.30, # cap ||x-x_0||/canvas_diag per traj
    integrator: str = "leapfrog",        # zeus B1: "leapfrog" (2nd-order)
                                          # or "yoshida4" (4th-order symplectic)
    verbose: bool = False,
) -> tuple[list, dict]:
    """Generate candidate placements via subspace HMC.

    Math: x_traj = x_0 + V · a_L, where a_L is the K-dim coordinate
    after L leapfrog steps. The endpoint x_traj is one candidate;
    we generate `n_trajectories` independent samples (different
    momentum draws).

    Parameters
    ----------
    macro_pos : (n_total, 2) current placement. Detached for HMC.
    smooth_proxy_call : closure x_tensor → scalar U_smooth.
    eigvals, eigvecs : output of `hessian_min_eigvecs_topk`. Eigvecs
        must have shape (2 * n_total, K) — column j is v_j.
    n_trajectories : count of HMC trajectories (= count of candidates).
    n_leapfrog : leapfrog steps per trajectory.
    step_size : leapfrog stepsize h. Controls trajectory length.
    canvas_diag : sqrt(cw² + ch²) for scaling cap.
    mass_floor : minimum mass entry (regularizes 1/sqrt(|λ|) when
        eigvals → 0). Default 1e-3 means very flat directions get
        moderate (not infinite) velocity.
    soft_only / n_hard : if True, zero out the eigvec components
        corresponding to hard-macro positions BEFORE running HMC. This
        constrains motion to soft macros only, matching the existing
        Hessian phase convention.
    seed : RNG seed for momentum draws.
    max_total_step_canvas : per-trajectory cap on ||V·a_L||/canvas_diag.
        If a trajectory exceeds this radius, we scale a_L back to the
        cap (still a valid candidate, just with shorter step). Prevents
        runaway trajectories from a single bad gradient evaluation.

    Returns
    -------
    candidates : list of (label, perturbed_pos_np) tuples
    diag       : dict with timing / acceptance stats
    """
    if eigvecs is None or eigvals is None:
        return [], {"warn": "no eigeninfo"}
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvecs.ndim != 2 or eigvals.ndim != 1:
        return [], {"warn": f"bad shape {eigvecs.shape}/{eigvals.shape}"}
    N, K = eigvecs.shape
    n_total = macro_pos.shape[0]
    if N != 2 * n_total:
        return [], {"warn": f"eigvec dim {N} != 2·n_total {2*n_total}"}
    if K == 0:
        return [], {"warn": "K=0"}

    device = macro_pos.device
    dtype = macro_pos.dtype

    # Soft-only projection: zero out hard-macro rows of each eigvec.
    if soft_only and n_hard > 0:
        eigvecs_view = eigvecs.reshape(n_total, 2, K).copy()
        eigvecs_view[:n_hard, :, :] = 0.0
        eigvecs = eigvecs_view.reshape(N, K)
        # Renormalize each column (V's orthonormality is broken by
        # zeroing, but each column should still be roughly unit-norm
        # in the soft-only subspace).
        col_norms = np.linalg.norm(eigvecs, axis=0)
        col_norms = np.where(col_norms > 1e-12, col_norms, 1.0)
        eigvecs = eigvecs / col_norms[None, :]

    V_torch = torch.tensor(eigvecs, dtype=dtype, device=device)
    # Mass and inv_mass per eigenvalue. Hessian-metric: M_jj = max(|λ_j|, ε).
    mass = np.maximum(np.abs(eigvals), mass_floor)
    inv_mass = 1.0 / mass
    sqrt_mass = np.sqrt(mass)

    base_xy_torch = macro_pos.detach().clone()
    n_total_int = int(n_total)
    K_int = int(K)
    L = int(n_leapfrog)
    h = float(step_size)
    rng = np.random.default_rng(seed)

    candidates: list = []
    diagnostics_per_traj: list = []
    cap_radius = float(max_total_step_canvas) * float(canvas_diag)
    t0_total = time.time()

    def _grad_at(a_np: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute (∇_a U, U) at the position x = x_0 + V·a."""
        a_t = torch.tensor(a_np, dtype=dtype, device=device,
                            requires_grad=True)
        delta = (V_torch @ a_t).reshape(n_total_int, 2)
        x_t = base_xy_torch + delta
        U_t = smooth_proxy_call(x_t)
        g_a_t = torch.autograd.grad(U_t, a_t)[0]
        return (g_a_t.detach().cpu().numpy().astype(np.float64),
                float(U_t.item()))

    # zeus B1 — Yoshida 4th-order symplectic composition.
    # A 2nd-order leapfrog Φ_h has error O(h^3) per step. The 4th-order
    # composition is
    #     Φ4_h  =  Φ_{w1 h}  ∘  Φ_{w2 h}  ∘  Φ_{w1 h}
    # with w1 = 1/(2-2^(1/3)) ≈ 1.3512, w2 = 1-2 w1 ≈ -1.7024
    # (Yoshida 1990, "Construction of higher order symplectic integrators").
    # Cost = 3× leapfrog substeps per "step", but error → O(h^5).
    # Net: at the same wall-clock, we can take ~sqrt(h) larger effective
    # step → longer trajectories without losing reversibility / symplecticity.
    cbrt2 = 2.0 ** (1.0 / 3.0)
    yoshida_w1 = 1.0 / (2.0 - cbrt2)
    yoshida_w2 = -cbrt2 / (2.0 - cbrt2)
    use_yoshida = (integrator == "yoshida4")

    def _leapfrog_substep(a, p, sub_h):
        # Kick-drift-kick form. Returns (a', p') after one Φ_{sub_h}.
        g_a, _ = _grad_at(a)
        p = p - 0.5 * sub_h * g_a
        a = a + sub_h * inv_mass * p
        g_a, _ = _grad_at(a)
        p = p - 0.5 * sub_h * g_a
        return a, p

    for traj in range(int(n_trajectories)):
        # Sample initial momentum p ~ N(0, M)
        p = rng.standard_normal(K_int) * sqrt_mass
        a = np.zeros(K_int, dtype=np.float64)

        _, U0 = _grad_at(a)
        if use_yoshida:
            # L outer steps, each a 3-substep Yoshida composition.
            for _step in range(L):
                a, p = _leapfrog_substep(a, p, yoshida_w1 * h)
                a, p = _leapfrog_substep(a, p, yoshida_w2 * h)
                a, p = _leapfrog_substep(a, p, yoshida_w1 * h)
        else:
            # Classic kick-drift-kick leapfrog, L steps total.
            g_a, _ = _grad_at(a)
            p = p - 0.5 * h * g_a
            for step in range(L):
                a = a + h * inv_mass * p
                g_a, _ = _grad_at(a)
                if step == L - 1:
                    p = p - 0.5 * h * g_a
                else:
                    p = p - h * g_a

        # Cap trajectory radius
        delta_xy = eigvecs @ a       # (N,)
        radius = float(np.linalg.norm(delta_xy))
        if radius > cap_radius > 0:
            scale = cap_radius / radius
            a = a * scale
            delta_xy = delta_xy * scale

        # Final position + endpoint surrogate
        x_final_np = (macro_pos.detach().cpu().numpy().astype(np.float64)
                      + delta_xy.reshape(n_total_int, 2))
        try:
            with torch.no_grad():
                x_t = torch.tensor(x_final_np, dtype=dtype, device=device)
                U_final = float(smooth_proxy_call(x_t).item())
        except Exception:
            U_final = float("nan")

        label = f"hmc_t{traj:02d}_K{K_int}_dU{(U_final-U0):+.4f}"
        candidates.append((label, x_final_np.astype(np.float64)))
        diagnostics_per_traj.append({
            "traj": traj, "U0": U0, "U_final": U_final,
            "delta_U": U_final - U0, "a_norm": float(np.linalg.norm(a)),
            "radius_microns": float(np.linalg.norm(delta_xy)),
        })

    diag = {
        "method": "subspace_hmc",
        "integrator": integrator,
        "n_trajectories": int(n_trajectories),
        "n_leapfrog": L,
        "step_size": h,
        "K": K_int,
        "mass_floor": mass_floor,
        "wall_s": time.time() - t0_total,
        "lambda_min": float(np.min(eigvals)),
        "trajectories": diagnostics_per_traj,
    }
    if verbose:
        n_improving = sum(1 for d in diagnostics_per_traj
                            if d["delta_U"] < -1e-9
                            and np.isfinite(d["delta_U"]))
        med_radius = float(np.median(
            [d["radius_microns"] for d in diagnostics_per_traj]))
        print(f"    [subspace-hmc] K={K_int} L={L} h={h:.3f} integ={integrator} "
              f"trajs={n_trajectories} surrogate-improving={n_improving} "
              f"med_radius={med_radius:.1f}μm  ({diag['wall_s']:.1f}s)",
              flush=True)
    return candidates, diag


# zeus B2 — Replica-overlap diverse subset selection (spin-glass inspired).
#
# The Parisi replica trick for spin glasses: when many replicas of the
# same system are simulated, their pairwise overlap distribution P(q)
# reveals the basin structure. If P(q) is peaked at high q, the
# replicas are stuck in one basin (replica-symmetric). If P(q) has
# multiple peaks, the system has many metastable basins (replica
# symmetry broken — RSB).
#
# In our subspace-HMC, each trajectory = one replica. The overlap
# between trajectory i and j is the inner product of their endpoints
# in the K-dim eigvec coord system, normalized. We want maximum spread
# of endpoints (cover diverse basins) rather than the lowest single
# surrogate value. Algorithm: greedy farthest-point selection.

def replica_diverse_select(
    candidates: list,                     # list of (label, pos_np)
    base_pos: np.ndarray,                 # original macro_pos shape (n,2)
    n_select: int = 8,                    # candidates to keep
    *,
    candidate_diagnostics: list | None = None,  # optional, per-traj dicts
) -> tuple[list, dict]:
    """Greedy farthest-point selection on the candidate endpoint set.

    Step 1: pick the candidate with lowest U_final as the seed.
    Step k: pick the candidate maximizing min-distance to the
        already-selected set (in placement-displacement L2 norm).

    Returns (selected_candidates, diag).

    Diag carries the pairwise overlap matrix and the chosen subset's
    minimum spread — both useful for understanding the basin structure.
    """
    if not candidates:
        return [], {"warn": "no candidates"}
    n_cand = len(candidates)
    if n_select >= n_cand:
        return candidates, {"warn": f"n_select {n_select} ≥ n_cand {n_cand}; pass through"}

    # Vectors of displacement from base, flattened to 1D per replica.
    base = base_pos.reshape(-1).astype(np.float64)
    disps = np.zeros((n_cand, base.size), dtype=np.float64)
    for i, (_, pos_np) in enumerate(candidates):
        disps[i, :] = pos_np.reshape(-1).astype(np.float64) - base
    # Pairwise L2-distance matrix (replica overlap, in micron units).
    sq = (disps ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * disps @ disps.T, 0))

    # Greedy seed: lowest U_final if diagnostics provided; else first.
    if candidate_diagnostics and len(candidate_diagnostics) == n_cand:
        us = np.array([d.get("U_final", np.inf) for d in candidate_diagnostics])
        us = np.where(np.isfinite(us), us, np.inf)
        seed_idx = int(np.argmin(us))
    else:
        seed_idx = 0

    selected_idx = [seed_idx]
    available = set(range(n_cand)) - {seed_idx}
    while len(selected_idx) < n_select and available:
        # For each available i, min-distance to the selected set.
        cands = np.array(sorted(available))
        sel_arr = np.array(selected_idx)
        min_d = D[np.ix_(cands, sel_arr)].min(axis=1)
        winner = int(cands[np.argmax(min_d)])
        selected_idx.append(winner)
        available.discard(winner)

    sub_cands = [candidates[i] for i in selected_idx]
    # Diag: overlap stats over the FULL pool, and the chosen subset's spread.
    upper_tri = D[np.triu_indices(n_cand, k=1)]
    sub_D = D[np.ix_(selected_idx, selected_idx)]
    sub_upper = sub_D[np.triu_indices(len(selected_idx), k=1)]
    diag = {
        "method": "replica_diverse_select",
        "n_cand": n_cand,
        "n_select": len(selected_idx),
        "selected_idx": selected_idx,
        "all_pairwise_median_microns": float(np.median(upper_tri)) if upper_tri.size else 0.0,
        "all_pairwise_max_microns": float(np.max(upper_tri)) if upper_tri.size else 0.0,
        "subset_pairwise_min_microns": float(np.min(sub_upper)) if sub_upper.size else 0.0,
        "subset_pairwise_median_microns": float(np.median(sub_upper)) if sub_upper.size else 0.0,
    }
    return sub_cands, diag
    return candidates, diag
