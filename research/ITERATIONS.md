# Autonomous iteration log — albania1

Started: 2026-05-05 23:10 PDT.
Goal: push proxy < 1.0109. End: verified ≤ 0.95 OR 3× <0.005 OR ~12h.

---

## Iter 0 — initial state (verified)

Branch albania1 (off v7-combinatorial-submission, verified at 1.0109).
Code added: cong-aware Hessian, per-component weights, Klein-4 orientation,
halo, CVaR knobs, benchmark.py forward-compat.

---

## Iter 1 — structural floor (verified)

Netlist demand/supply ratio predicts 81% of v7 cong variance (R² for cong
= 0.66, R² for proxy = 0.72, quadratic fit R² = 0.79). Per-bench
algorithmic room (v7 - structural floor):

| High room | residual | predicted Δ proxy |
|---|---:|---:|
| ibm06 | +0.262 | +0.131 |
| ibm12 | +0.269 | +0.135 |
| ibm18 | +0.234 | +0.117 |
| ibm07 | +0.080 | +0.040 |
| ibm03 | +0.062 | +0.031 |
| ibm08 | +0.045 | +0.023 |
| ibm15 | +0.032 | +0.016 |
| ibm02 | +0.032 | +0.016 |

| Below floor | residual |
|---|---:|
| ibm09 | -0.232 |
| ibm17 | -0.207 |
| ibm11 | -0.181 |
| ibm01 | -0.165 |
| ibm14 | -0.086 |
| ibm10 | -0.072 |
| ibm13 | -0.062 |
| ibm04 | -0.004 |
| ibm16 | -0.007 |

**Best-case algorithmic upside: ~0.030 mean improvement → 1.0003 → 0.97
dev box, 1.0109 → 0.98 verified.** Structural wall ≈ 0.95-0.98.

---

## Iter 2 — first cong-on test on ibm17 (LOW-room bench, predicted ~0 Δ)

Result: proxy=1.2805 (cong-on, k_dens=0.10) vs verified=1.2813 (cong-off).
Δ = -0.0008. **Within noise**. Confirmed ibm17 has no algorithmic room.

But: log revealed `ARPACK error -1: No convergence (51 iterations, 0/1
eigenvectors converged)`. **The Hessian phase silently no-op'd — the
saddle escape didn't fire.** The cong-included surrogate makes Lanczos
ill-conditioned on ibm17 at default 50 iters.

---

## Iter 3 — Lanczos convergence fix (committed)

Three changes to `_hessian_escape.py`:
1. **Tikhonov regularization**: `tikhonov` param. Adds ε·I to operator.
   `λ_min(H + εI) = λ_min(H) + ε`, recoverable. Default 1e-4.
2. **Auto-retry**: on convergence failure, retry with 4× maxiter. Free
   improvement on already-failed branches.
3. **Ladder fallback** in adaptive_topk_candidates: degenerate eigvec →
   retry k=1 + tikhonov ×100 + 4× iters.

Plus: bumped Lanczos default 50 → 100 in env.

Math tests still pass: saddle (λ=-2), min (λ=+2), top-3 orthogonality, etc.

---

## Iter 4 — focused cong-on sweep (running, ETA ~4.7h)

Benches chosen to span the room spectrum:
- ibm12, ibm06, ibm18 (high room — should show clear improvement)
- ibm15 (medium room — small improvement expected)
- ibm09 (below floor — should show ~0 Δ, validates no regression)

Cong-on default config: weight=0.5, K_cong_frac=0.05, Lanczos=100,
Tikhonov=1e-4. Comparing to verified baseline from sweep_results.csv.

Background: `/tmp/albania1_focused_cong_*/sweep.log`. Plus a 600s smoke
on ibm09 with fixed Lanczos to verify the convergence bug fix.

---

## Working theory

The 1.0109 verified mean has the structural wall at ~0.95-0.98 (per
linear/quadratic fit on netlist demand/supply). Cong-aware Hessian
should close the residuals on the 8 high-room benches, lifting mean by
~0.030. Realistic verified mean by morning: ~0.98.

To break below 0.95, need:
- Tier 2 levers (orientation, halos, timing weights) — affect routing
  supply, not Tier 1. Already partially implemented via orientations.pt
  sidecar.
- Architectural shifts: natural-gradient Langevin SDE
  (research/NATURAL_GRADIENT_IDEA.md), QP+overlap-soft-constraint
  relaxation, or recursive-bisection prior.

---

## Pipeline state (overnight)

```
KILLED:  k_dens A/B sweep (contaminated by Lanczos convergence bug)
KILLED:  chain stage 2/3 launchers (no longer needed)
RUNNING: ibm09 smoke (Lanczos fix verification)
RUNNING: focused cong-on sweep (5 benches × ~57min, ETA ~4.7h)
```

Expected results by ~5 AM PDT:
- 5 cong-on proxy values
- Per-bench Δ vs verified baseline
- Validation of structural-floor prediction (high-room benches show
  bigger improvements)

---

## Open follow-ups for morning

1. If focused sweep shows mean Δ ≥ 0.020 → run full 17-bench cong-on
   sweep (16h) for new submission baseline.
2. If focused sweep shows mean Δ < 0.010 → tune cong_weight (try 1.0,
   then 2.0). Iterative Hessian (MAX_ITERS=2 with shrunken per-iter).
3. Implement natural-gradient Langevin (one day, replaces Phase 1+3).
4. Build LP+QP convex relaxation for tight proxy lower bound.
