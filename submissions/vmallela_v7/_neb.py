"""Nudged Elastic Band (NEB) — zeus B11.

Henkelman & Jónsson (2000), "Improved tangent estimate in the nudged
elastic band method for finding minimum energy paths and saddle points"
(J. Chem. Phys. 113, 9978).

What this solves
================
Given two known low-cost placements x_A and x_B in DIFFERENT basins,
the NEB finds the MINIMUM ENERGY PATH (MEP) connecting them. The MEP's
highest-energy point is the optimal transition state — the smallest
saddle barrier separating the two basins. Knowledge of this barrier:
  (1) Tells us whether the basins are truly distinct or really one
      basin in disguise (low barrier ⇒ same basin).
  (2) Gives us a NEW candidate placement: the saddle itself, when
      perturbed slightly, decays to whichever basin we tilt toward.
  (3) Provides a "diversity certificate" for the v6 portfolio seeds.

Math
====
Discretize the path into K images x_0 = x_A, x_1, ..., x_{K-1} = x_B.
Each interior image is a placement; we relax all K-2 interior images
simultaneously. The NEB force on image i (for i in {1, ..., K-2}) is:

    F_i^NEB  =  F_i^perp - V'(x_i)·τ_i  +  F_i^spring·τ_i

where
    τ_i      = (unit) local tangent to the path at image i
    F_i^perp = -∇U(x_i) - (∇U(x_i) · τ_i)·τ_i
                  (component of -∇U perpendicular to τ_i)
    F_i^spring = k · (||x_{i+1} - x_i|| - ||x_i - x_{i-1}||)·τ_i
                  (longitudinal spring force keeping image spacing roughly equal)

Critical NEB ingredients:
  - PERPENDICULAR gradient pulls each image toward the MEP.
  - LONGITUDINAL spring keeps images evenly spaced (no piling up).
  - TANGENT computed from the up-slope side (H&J 2000 "improved tangent")
    handles cases near saddle/inflection where naive central difference
    of x gives an oscillating tangent.

Convergence
===========
F_i^NEB = 0 ⇒ images lie on the MEP. The image with highest U along
the converged path is the saddle (transition state). For a CONVEX
single-well, the MEP is the straight line (no saddle, all images
sit on one side).

For our problem, we use NEB as DIAGNOSTIC + CANDIDATE GENERATOR:
  - Run 30-50 NEB iterations from linear interpolant.
  - The image with highest U on the relaxed band → "saddle candidate".
  - The two adjacent images on either side of the saddle → "basin centers".
  - All 3 are added as candidates; the validation step picks the best.

Why this should help here
=========================
The v6 portfolio runs N independent Phase-1s. Some seeds end in the
same basin (degenerate); others in different basins. NEB ON THE TOP-2
SEEDS tells us:
  (a) If they're in the same basin: take the lower-cost seed, skip the
      other (save Hessian-phase budget).
  (b) If they're in different basins: the saddle between them is a
      novel candidate that NEITHER seed will find by local search.
  (c) Steepest-descent FROM the saddle into either basin gives us a
      DEEPER point than either original seed (the bottom of the
      "downhill side" of the saddle).

Failure modes
-------------
- High image count → expensive (K · grad eval per NEB step).
- Spring constant k mistuned: too soft → images bunch at endpoints;
  too stiff → straight-line band, no perpendicular relaxation.
- Same-basin seeds: NEB band relaxes to the lower basin, no saddle
  found. We detect via "no image has U > both endpoints + ε" and skip.

Implementation
==============
Pure NumPy on placement coords; one ∇U eval per image per NEB step via
the caller's smooth_proxy_call closure. Compute:
  (1) Tangents via H&J improved formula.
  (2) Perpendicular and spring forces.
  (3) Project F^NEB and step in that direction (no momentum — simple GD).
"""

from __future__ import annotations
import time
import numpy as np
import torch


def _improved_tangent(x_prev, x_cur, x_next, U_prev, U_cur, U_next):
    """Henkelman-Jónsson 2000 improved tangent.

    If U_{i+1} > U_i > U_{i-1}: tangent = x_{i+1} - x_i (uphill direction).
    If U_{i+1} < U_i < U_{i-1}: tangent = x_i - x_{i-1} (downhill direction).
    Else (peak or valley): weighted combination by ΔU magnitudes.

    All arrays flattened to 1D for arithmetic; we normalize at the end.
    """
    tau_plus = x_next - x_cur
    tau_minus = x_cur - x_prev
    if U_next > U_cur and U_cur > U_prev:
        tau = tau_plus
    elif U_next < U_cur and U_cur < U_prev:
        tau = tau_minus
    else:
        dU_max = max(abs(U_next - U_cur), abs(U_prev - U_cur))
        dU_min = min(abs(U_next - U_cur), abs(U_prev - U_cur))
        if U_next > U_prev:
            tau = tau_plus * dU_max + tau_minus * dU_min
        else:
            tau = tau_plus * dU_min + tau_minus * dU_max
    norm = np.linalg.norm(tau)
    if norm < 1e-30:
        tau = tau_plus  # fall back
        norm = max(np.linalg.norm(tau), 1e-30)
    return tau / norm


def neb_relax(
    x_A: np.ndarray,                     # (n, 2) start endpoint
    x_B: np.ndarray,                     # (n, 2) end endpoint
    U_grad_eval,                         # callable: (n,2) → (U, ∇U)
    *,
    n_images: int = 8,                   # K images including endpoints
    n_iters: int = 30,
    lr: float = 0.5,                     # gradient descent step
    spring_k: float = 0.1,               # longitudinal spring constant
    n_hard: int = 0,                     # leading rows not relaxed
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[float], dict]:
    """Relax a band of n_images images between x_A and x_B.

    Returns:
        images   : list of (n,2) arrays — relaxed band.
        Us       : list of float — U at each image.
        diag     : dict with stats.
    """
    K = int(n_images)
    if K < 3:
        raise ValueError(f"Need n_images >= 3, got {K}")
    n_total = x_A.shape[0]
    images = []
    # Linear interpolation between endpoints.
    for k in range(K):
        alpha = k / (K - 1)
        x_k = (1.0 - alpha) * x_A + alpha * x_B
        images.append(x_k.copy())
    images = np.stack(images)            # (K, n, 2)

    diag = {"method": "neb", "history_U_max": [], "history_max_force": []}
    t0 = time.time()

    for it in range(int(n_iters)):
        # Evaluate U and ∇U at every image.
        Us = np.empty(K, dtype=np.float64)
        grads = np.empty_like(images)
        for k in range(K):
            U_k, g_k = U_grad_eval(images[k])
            Us[k] = U_k
            grads[k] = g_k

        # Compute forces on interior images.
        forces = np.zeros_like(images)
        for k in range(1, K - 1):
            x_prev = images[k - 1].reshape(-1)
            x_cur = images[k].reshape(-1)
            x_next = images[k + 1].reshape(-1)
            tau = _improved_tangent(
                x_prev, x_cur, x_next, Us[k-1], Us[k], Us[k+1])
            g_flat = grads[k].reshape(-1)
            # Perpendicular gradient component:
            # F_perp = -∇U + (∇U · τ)·τ
            g_parallel = (g_flat @ tau) * tau
            F_perp = -g_flat + g_parallel
            # Spring force along tangent:
            l_plus = np.linalg.norm(x_next - x_cur)
            l_minus = np.linalg.norm(x_cur - x_prev)
            F_spring = spring_k * (l_plus - l_minus) * tau
            F = F_perp + F_spring
            forces[k] = F.reshape(n_total, 2)

        # Zero hard-macro components.
        if n_hard > 0:
            forces[:, :n_hard, :] = 0.0
        # Step (GD with fixed lr).
        images[1:K-1] = images[1:K-1] + lr * forces[1:K-1]

        max_F = float(np.linalg.norm(forces[1:K-1].reshape(K - 2, -1),
                                       axis=-1).max() if K > 2 else 0.0)
        diag["history_max_force"].append(max_F)
        diag["history_U_max"].append(float(Us.max()))

        if verbose and (it == 0 or it == n_iters - 1 or (it + 1) % 10 == 0):
            print(f"    [neb] iter {it+1}/{n_iters} max|F|={max_F:.4e} "
                  f"U_max={Us.max():.4f} U_min={Us.min():.4f}", flush=True)

    # Final eval of all images.
    Us_final = []
    for k in range(K):
        U_k, _ = U_grad_eval(images[k])
        Us_final.append(float(U_k))
    diag["U_final"] = Us_final
    diag["U_max_idx"] = int(np.argmax(Us_final[1:K-1]) + 1) if K > 2 else 0
    diag["U_max"] = float(Us_final[diag["U_max_idx"]])
    diag["U_endpoints"] = (Us_final[0], Us_final[-1])
    diag["barrier"] = diag["U_max"] - min(Us_final[0], Us_final[-1])
    diag["wall_s"] = time.time() - t0
    return [im for im in images], Us_final, diag


def neb_candidates(
    seeds: list[np.ndarray],             # list of (n, 2) candidate placements
    smooth_proxy_call,                   # callable: x_tensor → scalar U
    *,
    n_images: int = 7,
    n_iters: int = 30,
    lr: float = 0.5,
    spring_k: float = 0.1,
    n_hard: int = 0,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    barrier_eps_frac: float = 0.001,     # min relative barrier to call basins distinct
    verbose: bool = False,
) -> tuple[list, dict]:
    """Run NEB on PAIRS of seeds. For each pair (i, j) with i < j:
    - Relax a band.
    - If a barrier > eps · max(U_i, U_j) is found, add the saddle and
      its two adjacent images as candidates.
    - If no barrier: report "same basin" — no new candidate, but the
      lower of (i, j) is implicitly preferred.

    Returns (candidates_list_of_(label, pos), diag).
    """
    diag = {"method": "neb_candidates", "pairs": []}
    candidates: list = []
    if len(seeds) < 2:
        return [], {"warn": "need >=2 seeds"}
    device = torch.device("cpu")
    dtype = torch.float32

    def U_grad_eval(x_np):
        x_t = torch.tensor(x_np, dtype=dtype, device=device,
                            requires_grad=True)
        U_t = smooth_proxy_call(x_t)
        g_t = torch.autograd.grad(U_t, x_t)[0]
        return float(U_t.item()), g_t.detach().cpu().numpy().astype(np.float64)

    n_pairs_total = min(3, len(seeds) - 1)   # cap at 3 pairs to bound cost
    for p in range(int(n_pairs_total)):
        a, b = 0, p + 1
        x_A = seeds[a]
        x_B = seeds[b]
        images, Us_final, pair_diag = neb_relax(
            x_A, x_B, U_grad_eval,
            n_images=n_images, n_iters=n_iters, lr=lr, spring_k=spring_k,
            n_hard=n_hard, verbose=verbose,
        )
        # Detect barrier.
        endpoint_lo = min(pair_diag["U_endpoints"])
        barrier_rel = (pair_diag["U_max"] - endpoint_lo) / max(abs(endpoint_lo), 1e-9)
        pair_diag["barrier_rel"] = barrier_rel
        pair_diag["pair"] = (a, b)
        diag["pairs"].append(pair_diag)
        if barrier_rel > barrier_eps_frac:
            # Saddle image and its two neighbors.
            saddle_idx = pair_diag["U_max_idx"]
            for offs in [0, -1, 1]:
                idx = saddle_idx + offs
                if idx < 0 or idx >= len(images):
                    continue
                # Clamp to canvas.
                im = images[idx].copy()
                im[:, 0] = np.clip(im[:, 0], 0.0, float(canvas_w))
                im[:, 1] = np.clip(im[:, 1], 0.0, float(canvas_h))
                label = f"neb_p{a}{b}_img{idx}_U{Us_final[idx]:+.4f}"
                candidates.append((label, im))
    diag["n_candidates"] = len(candidates)
    return candidates, diag
