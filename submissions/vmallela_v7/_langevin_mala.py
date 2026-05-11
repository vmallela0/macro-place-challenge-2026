"""Metropolis-Adjusted Langevin proposal with exact-cost gate — zeus Plan B.

Idea
====
The smooth surrogate U_smooth(x) approximates the exact proxy U_exact(x)
but diverges from it (frozen routing in cong, CVaR smoothing, electro
density). Iter 4d–7 showed that gradient-descent-style optimization
on U_smooth (Adam, eigvec line search) is brittle: a step that
decreases U_smooth may *increase* U_exact when the surrogate is
miscalibrated.

MALA-with-exact-gate fixes this by using ∇U_smooth as a PROPOSAL
direction and U_exact as the ACCEPTANCE criterion:

  x' = x − h · ∇U_smooth(x) + √(2 h T) · ξ,   ξ ~ N(0, I)

  accept if U_exact(x') < U_exact(x) − tol

This is rejection-free Monte Carlo with smooth-gradient drift +
Gaussian noise. Theoretical guarantees (Roberts-Tweedie 1996):
- When U_smooth = U_exact and we accept by Metropolis ratio, MALA
  converges to π ∝ exp(−U_exact/T).
- When U_smooth ≠ U_exact, the strict-improvement gate filters bad
  proposals; convergence to argmin U_exact is preserved (worst case:
  pure random walk + exact gate ≈ slow SA).

Annealed: T decays geometrically T_t = T_0 · γ^t with γ ≈ 0.999. At
high T, broad exploration; as T → 0, concentrates on local mins.

Where to use
============
Two natural slots:
1. **Post-Hessian booster** (Phase 4): once the Hessian escape pulls
   us into a new basin, run 200–500 MALA steps to refine. Each step is
   ~150 ms (one smooth backward + one exact eval). Total ~30–75 s
   wall, well within the post-Hessian budget reserve.
2. **Replacement for Phase 3** (radical): skip Lanczos entirely; run
   3000+ MALA steps from post-Lap state. Each step has cost
   `O(1 backward + 1 exact)` ≈ 150 ms; 3000 steps ≈ 7.5 min.

Strict-improvement gate is critical: without it, the smooth-gradient
bias makes us drift toward smooth-min, not exact-min.

Cost
====
Per step: 1 autograd backward on smooth surrogate (~50 ms for ibm15)
+ 1 exact proxy via PlacementCost (~100 ms) ≈ 150 ms total.
3000 steps × 150 ms = 7.5 min.

Math derivation
===============
For target π ∝ exp(−U_exact / T), the Langevin SDE is
    dX_t = −∇U_exact(X_t) dt + √(2 T) dW_t.
Euler-Maruyama discretization:
    X_{t+1} = X_t − h ∇U_exact + √(2 h T) ξ_t,  ξ_t ~ N(0, I).
We approximate ∇U_exact by ∇U_smooth (cheap, differentiable). The
proposal x' is biased toward U_smooth-descent, ergodic under
Gaussian noise.

The gate (accept iff U_exact(x') < U_exact(x) − tol) makes the
algorithm a **greedy improvement walk** rather than a sampler. Each
accepted move strictly decreases the exact proxy.

Optional: replace strict gate with Metropolis-Hastings using a
SMOOTH-Hastings ratio (the q-correction term). This recovers
samples from π_smooth, biased away from π_exact. For optimization
purposes the strict gate is preferred.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def mala_search(
    macro_pos: torch.Tensor,                # (n_total, 2) current state
    smooth_proxy_call,                       # callable: x → scalar U_smooth
    exact_proxy_call,                        # callable: x_np → (cost, ov)
    canvas_diag: float,
    *,
    n_steps: int = 1000,
    step_size_frac: float = 0.005,           # h as frac of canvas_diag
    temp_init_frac: float = 0.005,           # T_0 as frac of canvas_diag²
    temp_decay: float = 0.999,
    soft_only: bool = True,
    n_hard: int = 0,
    seed: int = 42,
    n_burn: int = 50,                         # burn-in steps (no accept)
    cap_displacement_frac: float = 0.30,    # cap ||x_new - x_0||/canvas_diag
    eval_every: int = 1,                     # validate exact every K steps
    verbose: bool = False,
) -> tuple[np.ndarray, float, dict]:
    """Run MALA-with-exact-gate from `macro_pos`, return best (pos, cost).

    Parameters
    ----------
    macro_pos : (n_total, 2) starting placement.
    smooth_proxy_call : closure returning scalar U_smooth (autograd-aware).
    exact_proxy_call  : callable x_np → (cost, n_overlap). Should use the
        official PlacementCost. Caller is responsible for caching the
        evaluator.
    canvas_diag       : sqrt(cw² + ch²) for unit-scaling.
    n_steps           : total Langevin steps.
    step_size_frac    : h as fraction of canvas_diag.
    temp_init_frac    : T_0 as fraction of canvas_diag² (so √(2hT_0)·ξ
        gives ~step_size·canvas_diag displacement per coord).
    temp_decay        : T_t = T_0 · γ^t.
    n_burn            : initial steps that are not eligible for acceptance
        — protects against early-stage stale gradient.
    cap_displacement_frac : if a proposal's ||x' - x_0||/canvas_diag exceeds
        this, reject; protects against catastrophic gradient explosions.
    eval_every        : exact-proxy evaluated every K Langevin steps to
        amortize cost when steps are small.

    Returns
    -------
    best_pos_np : (n_total, 2) best placement seen
    best_cost   : exact proxy at best_pos
    diag        : dict with iteration stats
    """
    device = macro_pos.device
    dtype = macro_pos.dtype
    n_total = int(macro_pos.shape[0])
    rng = np.random.default_rng(seed)

    h = float(step_size_frac) * float(canvas_diag)
    T0 = float(temp_init_frac) * (float(canvas_diag) ** 2)
    sigma_step = np.sqrt(2.0 * h * T0)         # initial noise std per coord
    cap = float(cap_displacement_frac) * float(canvas_diag)

    x_t = macro_pos.detach().clone()
    x0_np = x_t.cpu().numpy().astype(np.float64)
    best_pos_np = x0_np.copy()
    try:
        best_cost, best_ov = exact_proxy_call(best_pos_np)
    except Exception as e:
        if verbose:
            print(f"  [mala] init exact eval err: {e}; bail", flush=True)
        return best_pos_np, float("inf"), {"err": str(e)}
    if best_ov != 0:
        if verbose:
            print(f"  [mala] init has {best_ov} overlaps; bail", flush=True)
        return best_pos_np, best_cost, {"warn": "init has overlaps"}

    accepted = 0
    n_eval = 1
    log = []
    t_start = time.time()
    T = T0

    for step in range(int(n_steps)):
        x_var = x_t.detach().clone().requires_grad_(True)
        try:
            U = smooth_proxy_call(x_var)
            g_t = torch.autograd.grad(U, x_var)[0]
        except Exception as e:
            if verbose:
                print(f"  [mala] step {step}: grad err {e}; halt",
                      flush=True)
            break
        g_np = g_t.detach().cpu().numpy().astype(np.float64)
        if soft_only and n_hard > 0:
            g_np[:n_hard] = 0.0
        # Proposal
        sigma_t = np.sqrt(2.0 * h * T)
        noise = rng.standard_normal(g_np.shape) * sigma_t
        if soft_only and n_hard > 0:
            noise[:n_hard] = 0.0
        x_prop = x_t.detach().cpu().numpy().astype(np.float64) \
                 - h * g_np + noise
        # Cap displacement
        delta = x_prop - x0_np
        rad = float(np.linalg.norm(delta))
        if rad > cap:
            x_prop = x0_np + delta * (cap / rad)

        # Acceptance check (every eval_every steps)
        accept = False
        if (step + 1) % eval_every == 0 or step >= int(n_steps) - 5:
            try:
                cost_prop, ov_prop = exact_proxy_call(x_prop)
                n_eval += 1
            except Exception as e:
                if verbose:
                    print(f"  [mala] step {step}: exact err {e}; skip",
                          flush=True)
                cost_prop, ov_prop = float("inf"), -1
            if (ov_prop == 0 and cost_prop < best_cost - 1e-7
                    and step >= n_burn):
                best_cost = cost_prop
                best_pos_np = x_prop.copy()
                accepted += 1
                x_t = torch.tensor(x_prop, dtype=dtype, device=device)
                accept = True
        log.append({
            "step": step, "U_smooth": float(U.item()),
            "||g||": float(np.linalg.norm(g_np)),
            "T": T, "h": h, "accept": accept,
        })
        T *= temp_decay
        if verbose and (step % max(1, n_steps // 10) == 0 or accept):
            ag = "ACC" if accept else "..."
            print(f"  [mala] step {step:4d}: U_smooth={float(U.item()):.3f} "
                  f"||g||={float(np.linalg.norm(g_np)):.3e} "
                  f"T={T:.4f} best={best_cost:.5f} {ag}",
                  flush=True)

    diag = {
        "n_steps": n_steps,
        "n_eval": n_eval,
        "accepted": accepted,
        "wall_s": time.time() - t_start,
        "T_final": T,
        "h": h,
        "log": log,
    }
    if verbose:
        print(f"  [mala] DONE: {accepted}/{n_steps} accepted, "
              f"best={best_cost:.6f}, wall={diag['wall_s']:.1f}s",
              flush=True)
    return best_pos_np, float(best_cost), diag
