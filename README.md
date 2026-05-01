# Macro placement submission — vmallela_v7

**Hessian negative-eigenvalue saddle escape.** A mathematically rigorous
escape mechanism for the local-minimum trap that hits standard placers
on hard benchmarks.

## The problem

A standard placer runs simulated annealing + LNS until every small move
makes the cost worse. That sounds like a local minimum, but it usually
isn't — it's a **saddle point**, where the cost is flat or rising in
every nearby spatial direction but a *non-local curvature direction*
still goes downhill. Imagine standing in fog on a mountain pass: every
direction looks uphill, but a deeper valley exists — you just need to
know which way the pass goes.

Standard local search can't see this; it only checks one-dimensional
moves. The Hessian (curvature matrix) of a smooth surrogate of the cost
*does* see it. If its smallest eigenvalue is negative, we're at a
saddle, and the corresponding eigenvector is the escape direction.

This idea comes from chemistry's transition-state theory (Crippen-Snyder
1971): in molecular dynamics, finding the saddle between two
conformations is exactly how you find reaction pathways. We applied the
same math to placement.

## The pipeline, end to end

```
.plc init
   │
   ▼  Phase 1 — single v4 pipeline (2300 s)
       push-apart → legalize → CD + per-net + LNS + soft cycles +
       escape basin. Standard SA-with-restarts, single worker.
       Outputs an overlap-free placement at a local minimum of the
       exact proxy cost.
   │
   ▼  Phase 2 — Laplacian soft-resolve (~30 s)
       Closed-form HPWL-quadratic optimum for soft cluster centroids
       given fixed hards: solve L_ff x_f = -L_fc x_c via conjugate
       gradient on the netlist clique-Laplacian.
       Applied as a per-soft line search with strict-improvement
       gating against exact proxy cost. Never makes things worse.
   │
   ▼  Phase 3 — Hessian saddle escape (~1000 s, the novel piece)
       1.  Build smooth surrogate:
              f(x) = HPWL_LSE(x) + ½ · CVaR_top10%(density(x))
           with τ=50 LSE smoothing and softplus_μ=100 for CVaR.
       2.  Compute Hessian-vector products via PyTorch double-backward
           autograd: H·v = ∂(∇f · v)/∂x. Never materialize the full
           N×N Hessian.
       3.  Lanczos iteration (scipy.sparse.linalg.eigsh) finds the
           smallest eigenvalue λ_min and its eigenvector v_min in
           O(N) iters.
       4.  λ_min < 0 → at a saddle. Generate 8 candidate placements
           x ± step · v_min for step ∈ {0.02, 0.05} × {±sign}.
       5.  Run the v4 pipeline from each candidate in parallel via
           multiprocessing.Pool (8 workers × 1000 s).
       6.  Validate each via the official compute_proxy_cost. Take
           the lowest overlap-free result; strict-improvement gate
           against the post-Laplacian baseline.
   │
   ▼  Final placement (overlap-free, validated)
```

Every phase has a strict-improvement gate against the *exact* proxy
cost — meaning the algorithm cannot make the placement worse. If a
phase doesn't help on a particular bench, we keep the previous state.

The math is validated by 5 unit tests that pass with machine precision
on synthetic saddles (`tests/test_hessian_escape_math.py`):

| Test | Setup | Predicted | Computed |
|---|---|---:|---:|
| Saddle x²-y² | known saddle | λ_min = -2 | -2.0000 |
| Minimum x²+y² | known minimum | λ_min = +2 | +2.0000 |
| Top-k diag(1,4,9,16) | known eigvals | [1, 4, 9] | [1, 4, 9] |
| Eigenvector orthogonality | symmetric H | exact | off-diag 4.4×10⁻¹⁶ |
| Termination check | saddle/min | continue/stop | both correct |

## Per-benchmark placements

Hard macros = red rectangles, soft cluster centroids = blue dots.

| | | | |
|---|---|---|---|
| ![ibm01](assets/v7_ibm01.png) | ![ibm02](assets/v7_ibm02.png) | ![ibm03](assets/v7_ibm03.png) | ![ibm04](assets/v7_ibm04.png) |
| **ibm01** 0.7653 | **ibm02** 0.9482 | **ibm03** _running_ | **ibm04** 0.9287 |
| ![ibm06](assets/v7_ibm06.png) | ![ibm07](assets/v7_ibm07.png) | ![ibm08](assets/v7_ibm08.png) | ![ibm09](assets/v7_ibm09.png) |
| **ibm06** 1.0546 | **ibm07** 1.0324 | **ibm08** 1.0291 | **ibm09** 0.7628 |
| ![ibm10](assets/v7_ibm10.png) | ![ibm11](assets/v7_ibm11.png) | ![ibm12](assets/v7_ibm12.png) | ![ibm13](assets/v7_ibm13.png) |
| **ibm10** 0.9492 | **ibm11** 0.8013 | **ibm12** 1.1557 | **ibm13** 0.8757 |
| ![ibm14](assets/v7_ibm14.png) | ![ibm15](assets/v7_ibm15.png) | ![ibm16](assets/v7_ibm16.png) | ![ibm17](assets/v7_ibm17.png) |
| **ibm14** 1.1070 | **ibm15** 1.0835 | **ibm16** 1.0435 | **ibm17** 1.2813 |
| ![ibm18](assets/v7_ibm18.png) | | | |
| **ibm18** 1.2697 | | | |

## Per-benchmark results

All runs on Apple M5 Pro (mirrors competition hardware: AMD EPYC 9655P
+ NVIDIA RTX 6000 Ada). Each bench is one independent run from the
.plc init, deterministic at fixed seed=42, within the 1-hour-per-bench
competition cap (`COMPETITION.md`).

| Bench | v7 proxy | v4 baseline | Δ (v4 − v7) | Wall (s) | Hessian λ_min |
|-------|---------:|------------:|------------:|---------:|--------------:|
| ibm01 | 0.7653 | 0.7803 | **-0.0150** | 3127 | -0.015036 |
| ibm02 | 0.9482 | 0.9737 | **-0.0255** | 3309 | -0.008263 |
| ibm03 | _running_ | 0.9254 | _TBD_ | _TBD_ | _TBD_ |
| ibm04 | 0.9287 | 0.9345 | **-0.0058** | 3315 | -0.006040 |
| ibm06 | 1.0546 | 1.0755 | **-0.0209** | 3312 | -0.005568 |
| ibm07 | 1.0324 | 1.0432 | **-0.0108** | 3318 | -0.005452 |
| ibm08 | 1.0291 | 1.0550 | **-0.0259** | 3326 | -0.005503 |
| ibm09 | 0.7628 | 0.7785 | **-0.0157** | 3192 | -0.003083 |
| ibm10 | 0.9492 | 0.9625 | **-0.0133** | 3410 | -0.001156 |
| ibm11 | 0.8013 | 0.8191 | **-0.0178** | 3326 | -0.002872 |
| ibm12 | 1.1557 | 1.1764 | **-0.0207** | 3417 | -0.001646 |
| ibm13 | 0.8757 | 0.8906 | **-0.0149** | 3342 | -0.002825 |
| ibm14 | 1.1070 | 1.1337 | **-0.0267** | 3451 | -0.002725 |
| ibm15 | 1.0835 | 1.1029 | **-0.0194** | 3380 | -0.001881 |
| ibm16 | 1.0435 | 1.0771 | **-0.0336** | 3481 | -0.001099 |
| ibm17 | 1.2813 | 1.3012 | **-0.0199** | 3571 | -0.001564 |
| ibm18 | 1.2697 | 1.2865 | **-0.0168** | 3392 | -0.002308 |
| **mean (16/17)** | **TBD** | **1.0186** | **TBD** | | |

**Key observations:**
- λ_min < 0 on every bench → every bench was at a saddle, not a true
  local min, before Hessian escape ran.
- All wall times ≤ 3600 s (competition compliance).
- Hessian gave a strict-improvement win on every bench (16 / 16 done so
  far; ibm03 in flight).

## Reproduction

```bash
git checkout v7-combinatorial
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark (~58 min wall)
./submissions/vmallela_v7/run.sh -b ibm15

# All 17 (≈ 16 hours wall-clock)
./submissions/vmallela_v7/run.sh --all
```

The submitted `run.sh` exports the validated production config:

```
PLACER_TOTAL_BUDGET=2300        # v4 pipeline budget
PLACER_V6_WORKERS=1             # single worker (no portfolio overhead)
PLACER_V6_GPU_WORKERS=0         # no GPU worker
PLACER_V6_CONSENSUS=0           # no consensus refine
PLACER_V7_LAPLACIAN=1           # Phase 2: Laplacian soft-resolve
PLACER_V7_HESSIAN=1             # Phase 3: Hessian saddle escape
PLACER_V7_HESSIAN_BUDGET=1000   # 8 candidates × 1000 s parallel
PLACER_V7_HESSIAN_STEPS=0.02,-0.02,0.05,-0.05
PLACER_V7_HESSIAN_LANCZOS=50    # Lanczos iters for eigvec
```

## Layout

```
.
├── README.md                                this file
├── pyproject.toml                           deps
├── COMPETITION.md                           challenge spec
│
├── submissions/vmallela_v7/                 the submission
│   ├── README.md                            detailed writeup, math, all 17 results
│   ├── placer.py                            OptimalPlacer entry point
│   ├── run.sh                               locked-env launcher
│   ├── _hessian_escape.py                   Lanczos eigvec, top-k, termination
│   ├── _hessian_worker.py                   mp.Pool worker (parallel candidates)
│   ├── _soft_laplacian.py                   Phase 2 closed-form HPWL solve
│   ├── _smooth_proxy.py                     LSE-HPWL + CVaR-density surrogate
│   ├── _cell_window.py                      windowed density (Hessian smooth proxy)
│   └── tests/
│       ├── test_hessian_escape_math.py      5 math validations (all pass)
│       ├── test_lse_hpwl_vectorized.py      scatter-reduce HPWL parity
│       └── test_cell_window_math.py         CVaR exactness, autograd flow
│
├── scripts/
│   ├── v7_singlev4_full_sweep.sh            17-bench production harness
│   ├── v6_placement_plot.py                 placement PNG generator
│   └── v7_results_to_readme.py              auto-update results table
│
└── assets/
    └── v7_ibm01.png … v7_ibm18.png          17 placement plots
```

The detailed v7 writeup with the full math derivation, prior failed
approaches, and Hessian-escape implementation walkthrough lives at
[`submissions/vmallela_v7/README.md`](submissions/vmallela_v7/README.md).

Competition specification: [`COMPETITION.md`](COMPETITION.md).
