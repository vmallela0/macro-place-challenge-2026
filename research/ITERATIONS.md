# Autonomous iteration log — albania1

Started: 2026-05-05 23:10 PDT (user asleep; goal: push proxy < 1.0109).
End conditions: verified mean ≤ 0.95 OR 3 consecutive < 0.005 improvements OR ~12h elapsed.
Style: bold math reformulations (Ramanujan), rigorous verification (Tao).

---

## Iter 0 — initial state

Branch: `albania1` (off `v7-combinatorial-submission`, the verified-1.0109 branch)
- Cherry-picked istanbul (adaptive line-search Hessian)
- Klein-4 orientation flip + sidecar (Tier 2)
- Halo knob (default off)
- CVaR k_dens knob (default 0.10)
- **Congestion-aware Hessian smooth_proxy_call** (the breakthrough — default ON)
- Per-component surrogate weight knobs (HPWL/dens/cong, default 1.0/0.5/0.5)
- benchmark.py forward-compat for unknown fields

Verified baseline (cong-off, from `submissions/vmallela_v7/sweep_results.csv`):
- 17-bench mean: 1.0003 dev box / 1.0109 LSJ verified

---

## Iter 1 — structural floor analysis (verified)

Computed netlist-only invariants per bench. Correlation of each with v7 achieved congestion:

| Invariant | Pearson r |
|---|---:|
| demand/supply ratio | **+0.81** |
| pin per perimeter | +0.80 |
| n_pins | +0.72 |
| total Steiner length | +0.66 |
| Fiedler value | +0.55 |

**Linear fit:** v7_cong ≈ 1.71 · demand_supply + 0.625 (R² = 0.66 on cong, R² = 0.72 on proxy directly).

Per-bench algorithmic room (v7_proxy - structural_floor_proxy):

| Bench | room | demand/supply | v7 cong | predicted | residual |
|---|---:|---:|---:|---:|---:|
| ibm12 | **+0.134** | 0.4141 | 1.601 | 1.332 | +0.269 |
| ibm06 | **+0.131** | 0.3173 | 1.429 | 1.167 | +0.262 |
| ibm18 | **+0.117** | 0.5106 | 1.731 | 1.497 | +0.234 |
| ibm07 | +0.040 | 0.3901 | 1.371 | 1.291 | +0.080 |
| ibm03 | +0.031 | 0.2682 | 1.145 | 1.083 | +0.062 |
| ibm08 | +0.023 | 0.3981 | 1.350 | 1.305 | +0.045 |
| ibm15 | +0.016 | 0.4643 | 1.450 | 1.418 | +0.032 |
| ibm02 | +0.016 | 0.2908 | 1.154 | 1.122 | +0.032 |
| ibm04 | -0.002 | 0.3249 | 1.176 | 1.180 | -0.004 |
| ibm16 | -0.004 | 0.4709 | 1.422 | 1.429 | -0.007 |
| ibm17 | -0.009 | 0.7076 | 1.626 | 1.833 | **-0.207** |
| ibm13 | -0.031 | 0.3121 | 1.096 | 1.158 | -0.062 |
| ibm10 | -0.036 | 0.4131 | 1.258 | 1.330 | -0.072 |
| ibm14 | -0.043 | 0.5767 | 1.524 | 1.610 | -0.086 |
| ibm01 | -0.073 | 0.2343 | 0.860 | 1.025 | -0.165 |
| ibm11 | -0.090 | 0.3069 | 0.968 | 1.149 | -0.181 |
| ibm09 | **-0.096** | 0.2591 | 0.835 | 1.067 | **-0.232** |

**Key finding**: 9/17 benches are already BELOW the structural floor (v7's algorithm beats netlist topology prediction). 8/17 have algorithmic room. The 3 high-room benches (ibm12, ibm06, ibm18) total 0.382 in proxy room — closing them entirely would lift mean by 0.022.

**Best-case algorithmic upside on Tier 1 (closing all positive residuals): ~0.030 mean reduction.** From 1.0003 → 0.97 dev box, 1.0109 → 0.98 verified.

This is a **calibrated wall**: the structural floor is at proxy ≈ 0.95-0.96. Going below that requires breaking the demand/supply structural relationship (architectural changes, Tier 2 lever).

`research/lower_bounds/cong_difficulty.py` ; `cong_difficulty.csv`.

---

## Iter 2 — Hessian eigvec component decomposition (FAILED)

Wrote `research/lower_bounds/hessian_decomp.py` to test which surrogate component drives the eigvec direction. Tested 7 weight combinations on ibm01.

**Result**: Lanczos returns λ_min=0, ||v_min||=0 on .plc-init placement for all variants. The init has zero or ill-conditioned gradient (macros at default positions, no saddle structure yet).

**Lesson**: Hessian phase only meaningfully fires post-v4. Need post-Laplacian placement to test eigvec decomposition. Deferred to after sweep produces post-Lap state.

---

## Pipeline status (running overnight)

3-stage chained pipeline:

```
Stage 1 (active):  k_dens A/B at /tmp/albania1_cvar_ab_*  (started 21:56 PDT)
                   Currently on ibm17/ctrl ~30 min in
                   ETA: ~3h
Stage 2 (queued):  cong validation at /tmp/albania1_cong_validation_*
                   Triggers on stage 1 DONE
                   Tests: ibm15-off, ibm15-on, ibm17-off, ibm08-off
                   ETA: 3.8h
Stage 3 (queued):  decision-tree based on stage 2 cong delta
                   - ≥0.01: high-room A/B (ibm12/06/18 with cong_weight=1.0)
                   - ≥0.005: cong_weight=1.0 retry on ibm15/17/08
                   - <0.005: high-room A/B as fallback
                   ETA: 3.8-5.7h
```

Total wall ≤ 12.5h. By 7-8h (~6 AM PDT), stage 1 + stage 2 complete; stage 3 mid-run.

---

## Hypotheses to test in stage 2/3

H1 (PRIMARY): Adding cong to Hessian surrogate lowers proxy on benches with positive room. Stage 2 tests on ibm15/17/08; stage 3 tests on ibm12/06/18.

H2: cong_weight=1.0 (boost over default 0.5) gives larger improvement than cong_weight=0.5. Stage 3 tests via auto-fallback.

H3 (POSTPONED): multi-eigvec Hessian (k=3 instead of 1). Already supported by code via PLACER_V7_HESSIAN_ADAPTIVE_TOPK=3 with fallback. Test in iter 3 if H1/H2 succeed.

H4 (POSTPONED): iterative Hessian (MAX_ITERS=3). Code supports it but budget at 1000s/iter doesn't fit 3 passes within 1h cap. Need to shrink per-iter budget.

H5 (POSTPONED): Mérigot semi-discrete OT density refine (Phase 5 post-Hessian). Density already near floor, marginal expected lift. Defer.

---

## Open: real proxy lower bound via convex relaxation

Current LP-only LB is loose by 50-600× (drops overlap, allows collapse). To tighten, implement LP+QP joint relaxation:

```
min   ||·||₁ HPWL  +  α (x - x̄)ᵀ K (x - x̄)        # K = density-quadratic
s.t.  pairwise distance ≥ macro_size                # overlap as soft penalty
```

Solvable in cvxpy. Per bench ~minutes. Would tell us how close 0.95-0.96 wall is to true optimum. Defer until pipeline yields data.
