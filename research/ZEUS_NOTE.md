# Zeus — closing the surrogate-exact gap on the cong term

Status: in flight, branch `albania2` → push to `albania2` when committing.
Author: claude session, 2026-05-11.

## TL;DR

albania1's verified result is 0.9975 mean over 17 IBM. The brief
asks for ≥0.02 mean lift on top of that, with structural rather than
parametric work.

The empirical failure mode of every "include congestion in the Hessian
surrogate" experiment to date (ITERATIONS.md Iter 4d, 7) is that the
routing-demand contribution to the smooth proxy is **frozen at a
snapshot** — `IncrementalEvaluator.V_routing_smooth` from one starting
state, used unchanged inside `smooth_proxy_call` for every Hessian-
vector product. As macros move along the eigenvector direction during
Lanczos / line search, the frozen routing demand grows stale. The
eigvec is computed against a stale congestion map. Symptom:
cong-on with weight=0.5 nets to mean Δ −0.0007 (within noise) on 5
high-room benches; cong-off wins clean at w=0.0 on ibm06.

**Fix (this note):** make routing demand **differentiable** in macro
positions using Spindler-Johannes RUDY with LSE-smoothed bbox extents
and softplus-smoothed cell-overlap. Autograd flows ∂V_demand_c /
∂macro_pos through every (net, cell) pair the net's bbox touches.
The Hessian-vector product now sees the LIVE congestion gradient.

Combined with a subspace HMC escape (random momentum in the K-dim
smallest-eigvec subspace) for diversity beyond the existing axis-
aligned line search.

## What's wrong with the existing cong-aware Hessian

`placer.py:1469-1488` (pre-zeus):

```python
V_total = V_smooth_frozen + V_macro / max(grid_v_routes, 1e-9)
H_total = H_smooth_frozen + H_macro / max(grid_h_routes, 1e-9)
combined = torch.cat([V_total, H_total], dim=0)
cong_smooth = cvar_smooth(combined.unsqueeze(0), K_c, t_c.detach(), mu=100.0).squeeze()
loss = loss + cong_weight * cong_smooth
```

`V_smooth_frozen` is `torch.tensor(incr.V_routing_smooth)` — a
**constant** copied from the IncrementalEvaluator's post-init state.
`V_macro` is differentiable (macros' blockage of routing channels in
their footprint). The sum is differentiable only via `V_macro`. The
73 % proxy variance in congestion lives in V_routing_smooth, not
V_macro_smooth.

When Lanczos perturbs macros along the eigvec direction, the per-net
RUDY contribution would shift (some bboxes shrink, others grow, top-K
cells reshuffle). The frozen surrogate ignores that. The eigvec
direction `-∇U_smooth / ||·||` aligns with the wrong landscape — good
on the frozen map, often bad on the live map.

This is why Iter 7's `λ_min`-maximizing weight (w=0.75) produced WORSE
proxy than w=0.0 on ibm06: a deeper saddle on the frozen-RUDY surrogate
doesn't correspond to a deeper saddle on the live cost. The Hessian
phase is searching the wrong topography.

## The fix: differentiable RUDY

For each net n with smooth LSE bbox extents (Δx_n, Δy_n):
- horizontal-wire density per unit area = Δx_n / (Δx_n · Δy_n) = 1/Δy_n
- vertical-wire density per unit area   = 1/Δx_n
Per cell c in bbox_n:
- V_demand_c += net_weight_n · overlap(c, bbox_n) / Δy_n
- H_demand_c += net_weight_n · overlap(c, bbox_n) / Δx_n

All ops smoothed:
- bbox extents via per-net LSE scatter (τ=50, same as `lse_hpwl_vectorized`)
- cell overlap via softplus_μ (μ=100, same as `_cell_window`)
- division by (Δ + ε_bbox) with ε_bbox = 1 micron to handle degenerate
  bboxes (1-pin nets, all-pins-coincident nets)

Sparse COO storage of (net, cell) pairs: per net we precompute a
discrete bounding-box window with margin=4 cells, scatter contributions
only into window cells. Window is re-snapshot at the start of each
Hessian iteration (3 iters × 1 snapshot each, current default).
Total pair count ~1.1 M for ibm06 (vs the dense (n_nets, K_max) form's
8.6 M with K_max = 868 = whole grid because some net spans the canvas).

Code: `submissions/vmallela_v7/_rudy_smooth.py` —
- `build_net_window_indices_sparse`: returns (pair_net, pair_cell, n_pairs)
- `smooth_rudy_routing_sparse`: forward+backward differentiable

Validated by sanity tests at `tests/test_rudy_smooth.py`:
- Finite + gradient flow on a 3-macro / 3-net toy
- A net with only lower-left pins gives V[upper-right cell] ≈ 0
- For a 2-pin net spanning a known 40-wide bbox, ∑V ≈ net_weight · Δx
  (rel.err ≈ 0.024 from LSE smoothing of the bbox)
- Sparse output equals dense within 4.3e-8 relative error

## Subspace HMC escape

`submissions/vmallela_v7/_subspace_hmc.py`.

After Lanczos returns the K smallest eigpairs (λ_j, v_j), the existing
escape options are:
- `adaptive_topk_candidates` : per-eigvec backtracking line search →
  one candidate per eigvec at the single best step along ±v_j.
- `kdim_trust_region_step`   : analytic K-dim Newton step → one
  candidate from the quadratic-model minimum in span{v_1..v_K}.

Both are AXIS-ALIGNED in the K-dim Hessian eigenspace and DETERMINISTIC.
If the eigvec direction is computed against a stale or partially-wrong
surrogate (Iter 4d, 7), a single step lands in the locally-improved-
but-globally-stale state — strict-improvement gate then rejects it.

Subspace HMC adds RANDOMNESS in the K-dim subspace:
- Sample p ~ N(0, M = |Λ_K|)  (Hessian-metric momentum prior)
- Leapfrog L steps on H(a, p) = ½ p·M⁻¹p + U_smooth(x_0 + V a)
- Validate exact proxy at endpoint; strict-improvement gate

Math: `x_traj = x_0 + V · a_L` where V ∈ ℝ^(2N × K) has eigvecs as
columns. K-dim coords a evolve by leapfrog (symplectic, volume-
preserving). Chain rule via autograd: ∂U/∂a = V^T ∂U/∂x evaluated in
a single backward pass per leapfrog step.

Why subspace-HMC over plain HMC?
- Plain HMC samples p ~ N(0, I_{2N}); most random kicks bounce against
  stiff modes (high curvature in HPWL/density). Energy-wasting.
- Subspace HMC concentrates exploration on the smallest-eigval
  ("softest") modes — the natural manifold axes for crossing nearby
  basins.
- Mass M = |Λ_K| gives the Patterson-Teh 2013 SGRLD preconditioning:
  mixing time governed by 1/cond(M), not 1/spectral_gap.

Cost: L autograd backward passes per trajectory. T trajectories.
For ibm15-scale (~6 k pins) with K=6, L=12, T=16: ~600 ms × 16 = 10 s,
well within the 1000 s Hessian budget.

## Falsifiable predictions

Per ITERATIONS.md the residual upper bound on each high-room bench is
~0.13. Recovery rate of ~10-15 % of structural room with proper
gradient signal would give per-bench Δ:
- ibm06 (+0.262 residual) : −0.013 to −0.020
- ibm12 (+0.269 residual) : −0.013 to −0.020
- ibm18 (+0.234 residual) : −0.012 to −0.018
- ibm07 (+0.080)          : −0.004 to −0.008
- ibm03 (+0.062)          : −0.003 to −0.006

Plus +0.005 mean from end-to-end no-regression gate (eliminates 6-bench
regression band at ±0.005). Total expected mean Δ ≈ −0.012 to −0.020.

This is **below** the breakthrough threshold of −0.020 cited in the
brief. The honest math says one more leg (HMC + RUDY together, or a
follow-on architectural shift) is needed to clear the 0.02 bar.

## What this does NOT do

- It does not introduce Tier 2 levers (orientation, halos, timing).
  Those would loosen the structural-floor ceiling at 0.95-0.98.
- It does not replace the v4 SA Phase 1. v4 still produces the basin;
  the Hessian phase opens the door wider via cong-aware curvature.
- It does not "fix" surrogate-exact divergence — it narrows it on the
  cong term, where 73 % of proxy variance lives. CVaR-smooth-vs-exact
  and macro-blockage approximations remain.

## A/B plan

1. **Smoke**: BUDGET=600 single-bench (ibm06) with baseline vs rudy.
   Confirms no crash + ~rough Δ direction. ~10 min wall.
2. **3-bench A/B**: BUDGET=1800 with baseline vs rudy on ibm06, ibm12,
   ibm15. ~30 min wall (6 jobs parallel). Validates per-bench signal.
3. **Full 17 sweep**: ~17 h wall sequential, or run in parallel waves
   of 4 (60 cores / 8 worker each = ~7 placers parallel = 5h).
4. **If signal positive** at step 2: stack with HMC (K=6, T=16) on the
   3 high-room benches.

Env flags:
- `PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1` enables.
- `PLACER_V7_HESSIAN_RUDY_MARGIN=4` cell margin around per-net bbox.
- `PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64` drops giant-bbox nets (default
  64 cells; their per-cell contribution is dilute, dropping them is
  ~free in signal terms and ~5× faster).
- `PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16` enables
  subspace HMC additional candidates.
