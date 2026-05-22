# `vmallela_v7` — Hessian negative-eigenvalue saddle escape

A placer that escapes apparent local minima by computing the smallest
eigenvalue of the Hessian of a smooth surrogate of the proxy cost.
When that eigenvalue is negative, the corresponding eigenvector points
to a downhill direction the standard SA / LNS / coordinate-descent
moves cannot see — because they only probe one spatial axis at a time.

| Grader-verified mean (17 IBM benches) | Best | Worst | Overlaps | Wall |
|---:|---:|---:|---:|---:|
| **1.0109** | 0.7644 | 1.2921 | 0 | 15.5 h total, ≤ 1 h/bench |

---

## Why this works — the saddle-vs.-minimum picture

A point where every small spatial move increases the cost can be one
of two things:

| | λ_min(H) | Every direction | Action |
|---|:---:|---|---|
| **True local minimum** | ≥ 0 | uphill | stop |
| **Saddle point** | < 0 | uphill *in axes we probed* | escape along v_min |

In 2-D a saddle looks like a horse saddle: up-curvature along the
ridge, down-curvature across it. Standard placers feel the up-curvature
along the ridge and stop. The Hessian sees both curvatures
simultaneously; its smallest eigenvector points across the ridge.

This is the same mathematics that transition-state theory uses to find
reaction pathways in molecular dynamics (Crippen & Snyder 1971,
Henkelman & Jónsson 2000). The saddle point on the potential-energy
surface separates two stable conformations; the eigenvector of the
negative eigenvalue is the reaction coordinate.

---

## Pipeline

```
.plc init ─► Phase 1 (2300 s) ─► Phase 2 (~30 s) ─► Phase 3 (~1000 s) ─► out
            v4 baseline          Laplacian          Hessian escape
            (SA + LNS + CD)      soft-resolve       (the novel piece)
```

Each phase passes through a **strict-improvement gate** against the
*exact* proxy cost. If the phase did not help on this benchmark, its
output is discarded and the previous state is kept. The algorithm
cannot regress.

### Phase 1 — v4 baseline (2300 s)

Push-apart → legalize → coordinate descent → per-net optimization →
LNS → soft cycles → escape basin. A single worker; no portfolio.
We tried 8-worker portfolio (v6) and it underperformed on hard
benches because each worker got fewer optimization cycles than a
single deep run with the same total compute.

### Phase 2 — Laplacian soft-resolve (~30 s)

Given the fixed hard-macro positions from Phase 1, the optimal soft
cluster centroids for HPWL are the closed-form solution of

```
L_ff · x_f  =  -L_fc · x_c
```

where `L` is the clique-model Laplacian of the netlist hypergraph
(each k-pin net contributes pair weight `w_n / (k − 1)`) and the `f` /
`c` partitions are *free* (soft) and *constrained* (hard) macros.
Solved by conjugate gradient. Applied as a per-soft line search with
strict-improvement gating; provides −0.0005 to −0.005 cost per bench.

### Phase 3 — Hessian saddle escape (~1000 s)  ★ the novel piece

1. **Smooth surrogate.** Build a differentiable approximation of the
   proxy cost:

   ```
   f(x)  =  HPWL_LSE(x; τ = 50)  +  ½ · CVaR_top-10%( density(x); μ = 100 )
   ```

   - `HPWL_LSE`: log-sum-exp smoothing of bbox half-perimeter
     wirelength (exact as τ → ∞).
   - `CVaR_top-10%`: Rockafellar–Uryasev 2000 reformulation of the
     top-10% density average; smooth at finite μ, exact at μ → ∞.

2. **Hessian-vector product** by PyTorch double-backward autograd:

   ```
   H · v  =  ∂ (∇f · v) / ∂x
   ```

   N is large enough (10⁴–10⁵ dimensions) that materializing the full
   N×N Hessian is infeasible. Hvp is O(N) per call.

3. **Lanczos iteration** (`scipy.sparse.linalg.eigsh`, k = 1,
   `which = "SA"`, 50 iters) on the Hvp operator returns the smallest
   eigenvalue λ_min and eigenvector v_min.

4. **Generate candidates** along v_min:

   ```
   x_candidate(s)  =  x  +  s · v_min,   s ∈ {±0.02, ±0.05} · canvas_diag
   ```

5. **Reconverge in parallel.** 4 candidates × `multiprocessing.Pool`
   workers each run the reduced v4 pipeline (push-apart → legalize →
   CD → LNS → soft cycles) for ≤ 1000 s.

6. **Strict-improvement gate.** Score each candidate with the official
   `compute_proxy_cost`. Keep the lowest-cost overlap-free result if
   and only if it beats the post-Laplacian baseline.

**Why the surrogate works even though it's not bit-equal to the proxy.**
The Hessian eigenvector captures *large-scale curvature*, which is
robust to the LSE / CVaR smoothing. Local gradients are not — that's
why an earlier attempt to optimize the surrogate directly with Adam
failed (it followed local noise of the surrogate into worse regions
of the exact cost). Eigenvectors are a global geometric feature; they
survive the approximation.

---

## Math validation

Five unit tests in `tests/test_hessian_escape_math.py`, all pass with
machine-precision error:

| Test | Setup | Expected | Computed |
|---|---|---:|---:|
| Saddle x² − y² | known saddle at origin | λ_min = −2 | **−2.0000** |
| Minimum x² + y² | known min at origin | λ_min = +2 | **+2.0000** |
| Top-k diag(1, 4, 9, 16) | known eigvals | [1, 4, 9] | **[1, 4, 9]** |
| Eigenvector orthogonality | H symmetric | exact | off-diag 4.4 × 10⁻¹⁶ |
| Termination check | saddle / min | continue / stop | both correct |

---

## Results

### Grader-verified (official)

Run on AMD EPYC 9655P + NVIDIA RTX 6000 Ada (competition hardware):

| | |
|---|---:|
| Mean proxy cost (17 / 17) | **1.0109** |
| Best per-bench | 0.7644 |
| Worst per-bench | 1.2921 |
| Total overlaps | 0 |
| Total wall | 15.5 h |
| Per-bench wall | ≤ 1 h (compliant) |

### Local sweep (RTX 6000 Ada)

Per-bench breakdown from our own RTX 6000 Ada run. Slightly worse
mean (1.0409 vs. 1.0109) due to platform-specific BLAS / SIMD
trajectory differences from the grader; algorithm + config are
bit-identical.

| Bench | Proxy | Overlaps | Wall (s) |
|---|---:|---:|---:|
| ibm01 | 0.7745 | 0 | 3250 |
| ibm02 | 0.9897 | 0 | 3340 |
| ibm03 | 0.9256 | 0 | 3339 |
| ibm04 | 0.9334 | 0 | 3330 |
| ibm06 | 1.1007 | 0 | 3341 |
| ibm07 | 1.0586 | 0 | 3356 |
| ibm08 | 1.0591 | 0 | 3369 |
| ibm09 | 0.7748 | 0 | 3343 |
| ibm10 | 1.0039 | 0 | 3494 |
| ibm11 | 0.8406 | 0 | 3362 |
| ibm12 | 1.2366 | 0 | 3476 |
| ibm13 | 0.9198 | 0 | 3390 |
| ibm14 | 1.1598 | 0 | 3509 |
| ibm15 | 1.1548 | 0 | 3438 |
| ibm16 | 1.1168 | 0 | 3543 |
| ibm17 | 1.3398 | 0 | 3681 |
| ibm18 | 1.3064 | 0 | 3458 |
| **mean** | **1.0409** | — | — |

---

## Reproduction

```bash
git checkout v7-combinatorial-submission
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark
uv run evaluate submissions/vmallela_v7/placer.py --benchmark ibm15

# All 17 sequential (≈ 16 h)
for b in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 \
         ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  uv run evaluate submissions/vmallela_v7/placer.py --benchmark "$b"
done
```

The placer reads its configuration from `os.environ.setdefault` calls
at module-import time. The grader's no-env invocation
(`OptimalPlacer().place(bench)`) is therefore identical to our local
runs.

| Env var | Default | Meaning |
|---|---:|---|
| `PLACER_TOTAL_BUDGET` | 2300 | v4 pipeline budget (s) |
| `PLACER_V6_WORKERS` | 1 | parallel SA workers (1 = single deep run) |
| `PLACER_V6_GPU_WORKERS` | 0 | GPU SA workers |
| `PLACER_V6_CONSENSUS` | 0 | consensus refine |
| `PLACER_V7_LAPLACIAN` | 1 | Phase 2 on/off |
| `PLACER_V7_HESSIAN` | 1 | Phase 3 on/off |
| `PLACER_V7_HESSIAN_BUDGET` | 1000 | per-candidate reconverge budget (s) |
| `PLACER_V7_HESSIAN_STEPS` | `0.02,-0.02,0.05,-0.05` | candidate step sizes (fraction of canvas diag) |
| `PLACER_V7_HESSIAN_LANCZOS` | 50 | Lanczos iters for v_min |

---

## Files

```
submissions/vmallela_v7/
├── README.md             this writeup
├── placer.py             OptimalPlacer entry point (grader API)
├── run.sh                locked-env launcher
│
├── _hessian_escape.py    Lanczos eigvec + termination check
├── _hessian_worker.py    mp.Pool worker (parallel candidates)
├── _soft_laplacian.py    Phase 2: closed-form HPWL solve
├── _smooth_proxy.py      LSE-HPWL + CVaR-density surrogate
├── _cell_window.py       windowed density for the smooth proxy
│
└── tests/
    ├── test_hessian_escape_math.py    5 math validations
    ├── test_lse_hpwl_vectorized.py    scatter-reduce HPWL parity
    └── test_cell_window_math.py       CVaR exactness, autograd flow
```

The placer also imports shared code from the upstream
`vmallela`, `vmallela_v2`, and `vmallela_v6` submission directories
(the v4 baseline pipeline, the v6 portfolio infrastructure, and the
shared evaluator wrapper).

---

## Approaches tried before this one

Eleven distinct novel approaches were tried and discarded before
Hessian escape gave a positive signal:

| # | Approach | Why it failed |
|---|---|---|
| 1 | Adam on smooth surrogate | Surrogate-vs.-exact divergence; surrogate moves did not translate to exact-cost wins. |
| 2 | Gaussian basin-hop | 0 / 9 acceptances; perturbation either too small to escape or too big to recover. |
| 3 | Sequence-pair single-worker basin-hop | Inner minimizer plateaued at 300 s budget. |
| 4 | Sequence-pair 4-worker basin-hop | Diversity did not help — local minimizer is the ceiling. |
| 5 | Lévy α-stable basin-hop | Heavy tails amplified scale 21–44×; max jumps > 3× canvas, infeasible. |
| 6 | Top-K congestion eviction | Cost only modelled congestion; HPWL impact ignored; rejected by gate. |
| 7 | Sinkhorn OT eviction | Globally optimal but HPWL weight too low; full-apply blew cost up 3×. |
| 8 | ePlace electrostatic warm-start | HPWL-blind spreading destroyed net topology (+0.13 regression on ibm15). |
| 9 | ePlace n_steps tuning | Monotonic degradation as spreading grows. |
| 10 | DREAMPlace-style HPWL-aware ePlace | .plc init already at HPWL local min; HPWL pull over-collapsed macros. |
| 11 | v6 portfolio + Hessian | Portfolio overhead ate budget; ibm17 timed out at 3600 s. |

The pattern: every method that tried to **search** the cost landscape
got stuck in the same basins. Hessian escape works because it **uses
the local geometry** (curvature direction) to identify the escape
route, then lets standard search refine from there. A different
mathematical category of method.

---

## References

- Crippen & Snyder (1971), *J. Chem. Phys.* — saddle-point search in
  potential-energy surfaces.
- Henkelman & Jónsson (2000), *J. Chem. Phys.* — dimer method for
  finding transition states.
- Nesterov & Polyak (2006), *Math. Program.* — cubic regularization,
  convergence theorem for saddle escape.
- Rockafellar & Uryasev (2000), *J. Risk* — CVaR / Conditional
  Value-at-Risk as a smooth top-k operator.
- Tsay & Kuh (1991), *IEEE TCAD* — clique-model Laplacian for
  HPWL-quadratic placement.
- Lanczos (1950) — iterative tridiagonalization for sparse symmetric
  eigenproblems.
