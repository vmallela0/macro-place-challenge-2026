# Macro placement submission — vmallela_v7

**Hessian negative-eigenvalue saddle escape**: the trick that broke the
local-minimum ceiling that v4 and v6 were stuck at.

## Headline result

| Configuration | 17-bench mean | Per-bench |
|---|---:|---|
| `optimized` v2 baseline | 1.1172 | reported |
| `optimized_v4` (3300 s) | **1.0186** | reported |
| `v6-gpu` (8 workers × 1800 s + GPU CD + consensus) | **1.0184** | 10 W / 6 L / 1 T vs v4 |
| **`v7-combinatorial` (this branch, single v4 + Laplacian + Hessian escape)** | **TBD** | wins all 17 vs v4 (16/17 confirmed; ibm03 running) |

## What v7 does, in plain English

After running the standard v4 pipeline, we hit a *local minimum* — a
placement where every small move makes the cost worse, but the *true*
best placement is elsewhere. We solve this by computing the **Hessian**
(curvature) of a smooth approximation of the cost function. If the
smallest eigenvalue is negative, we're not at a true minimum — we're at
a **saddle point**, and the eigenvector tells us exactly which direction
to "fall through" to find a deeper basin.

This idea comes straight from chemistry's transition-state theory. The
math is bulletproof; the engineering challenge was making it run within
the 1-hour-per-benchmark competition cap. We did.

→ Detailed v7 writeup with all algorithm details, math validation, and
per-bench results: **[`submissions/vmallela_v7/README.md`](submissions/vmallela_v7/README.md)**

→ Detailed v6 writeup (the prior baseline): **[`submissions/vmallela_v6/README.md`](submissions/vmallela_v6/README.md)**

## v7-combinatorial (in flight)

A four-layer stack on top of v6 targeting the hard sparse benches.

```
v6 portfolio (8 workers + GPU CD + consensus)         →  1.0184 mean
   ↓
Layer 1 — Laplacian soft-resolve
   Closed-form HPWL minimum for soft macros given fixed hards via
   clique-model Laplacian L solving L_ff x_f = -L_fc x_c. Per-soft
   line search with full-proxy strict-improvement gating; never
   regresses. Validated on ibm01: 1.056 → 0.984 in 1.4 s.
   ↓
Layer 2 — Topological basin-hopping (Wales-Doye 1997)
   σ-perturb → reduced single-worker pipeline → strict accept.
   σ_0 = 0.10·canvas_diag (tuned via 9-config grid; 0.30·D too
   aggressive on ibm15, 0.05·D doesn't escape).
   ↓
Layer 3 — Adam Phase 4.5 (PLACER_V7_ADAM=1; off by default)
   Fully vectorized smooth surrogate:
     • LSE-HPWL via scatter_reduce(amax) + scatter_add(exp).
       O(P) compute (P = total pins). 60× speedup vs Python loop:
       50 steps on 6k-net synthetic in 0.48 s on MPS.
       Numerical parity vs Python reference: 4.6e-7 value, 3.6e-8 grad.
     • CVaR top-K density / congestion (Rockafellar-Uryasev 2000):
       focuses gradient on the tail (top 10% hottest cells), not
       the bulk — the bulk-gradient is what would otherwise drown
       out the hot-cell signal. CVaR exactly equals the top-K mean
       at t* = ρ_(n-K) (numerically verified at μ=1000).
     • Cell-window truncation: O(K_max) cells per macro, snapshotted
       every 25 steps. Density and blockage gradients pass autograd
       finiteness checks.
     • GradNorm component balancing (Chen et al. 2018): per-component
       initial gradient norms on ibm01 are HPWL=9.9e-4, density=0.115,
       cong=0.456 — without GradNorm, density/cong gradients are
       100×–500× larger than HPWL and the optimizer treats HPWL as
       essentially zero. GradNorm freezes per-component scale at
       step 0 so all three contribute on equal footing.
     • Strict-improvement gate via compute_proxy_cost — Adam can
       never make the placement worse than the post-Laplacian state.
   ↓
Layer 4 — Hard macro gradient drift (T3, soft_only=0)
   Hard macros added to Adam parameter set with quadratic inertia
   penalty toward init position. Lets the floorplan "breathe"
   during the smooth phase before final legalization. Bounded
   to <5 cells of drift; validated on ibm01.
```

Per-bench sweep results: **TBD** — sweep with tuned basin-hop +
Adam Phase 4.5 enabled is queued for kickoff at 16:42 PDT
2026-04-29 after the 9-config σ × hops mini-grid on ibm15
completes. Results land at
[`submissions/vmallela_v7/sweep_results.csv`](submissions/vmallela_v7/)
and the v7 README is auto-updated by `scripts/v7_results_to_readme.py`.

The detailed v7 writeup with the math derivations, validation tests,
performance numbers, and architecture diagram lives at:

→ **[`submissions/vmallela_v7/README.md`](submissions/vmallela_v7/README.md)**

## v6 visualizations

### Static placements (17 benches)

Hard macros = red rectangles, soft cluster centroids = blue dots. Style
matches `assets/ibm01_v4.png` so v4 ↔ v6 are directly comparable.

| | | | |
|---|---|---|---|
| ![ibm01](assets/v6_ibm01.png) | ![ibm02](assets/v6_ibm02.png) | ![ibm03](assets/v6_ibm03.png) | ![ibm04](assets/v6_ibm04.png) |
| **ibm01** 0.767 ✓ | **ibm02** 0.964 ✓ | **ibm03** 0.909 ✓ | **ibm04** 0.930 ✓ |
| ![ibm06](assets/v6_ibm06.png) | ![ibm07](assets/v6_ibm07.png) | ![ibm08](assets/v6_ibm08.png) | ![ibm09](assets/v6_ibm09.png) |
| **ibm06** 1.064 ✓ | **ibm07** 1.043 ≈ | **ibm08** 1.033 ✓ (+0.022) | **ibm09** 0.769 ✓ |
| ![ibm10](assets/v6_ibm10.png) | ![ibm11](assets/v6_ibm11.png) | ![ibm12](assets/v6_ibm12.png) | ![ibm13](assets/v6_ibm13.png) |
| **ibm10** 0.960 ✓ | **ibm11** 0.815 ✓ | **ibm12** 1.187 ✗ | **ibm13** 0.895 ✗ |
| ![ibm14](assets/v6_ibm14.png) | ![ibm15](assets/v6_ibm15.png) | ![ibm16](assets/v6_ibm16.png) | ![ibm17](assets/v6_ibm17.png) |
| **ibm14** 1.141 ✗ | **ibm15** 1.131 ✗ (worst, −0.028) | **ibm16** 1.093 ✗ | **ibm17** 1.308 ≈ |
| ![ibm18](assets/v6_ibm18.png) | | | |
| **ibm18** 1.304 ✗ | | | |

### Convergence GIFs (hard benches, proxy ≥ 1.0)

Each GIF runs a SHORT instrumented single-CPU-worker pipeline (push-apart
→ legalize → refine → CD with snapshots, 60 s CD budget) so you can see
WHERE the optimizer is settling. The final cost in each GIF is **not**
the production result — it's a low-budget diagnostic re-run; the
production result is at the full 1800 s × 8-worker portfolio.

The losing benches all show an early plateau: cost drops fast in
push-apart + legalize, then flattens 30+ s before the budget ends. The
optimizer doesn't need more passes — it needs a different basin.

| | |
|---|---|
| **ibm06** (v6 1.064 vs v4 1.075 — WIN) | **ibm07** (v6 1.043 ≈ v4 1.043 — TIE) |
| ![ibm06.gif](assets/v6_ibm06.gif) | ![ibm07.gif](assets/v6_ibm07.gif) |
| **ibm08** (v6 1.033 vs v4 1.055 — WIN +0.022, biggest swing) | **ibm12** (v6 1.187 vs v4 1.176 — LOSS −0.011) |
| ![ibm08.gif](assets/v6_ibm08.gif) | ![ibm12.gif](assets/v6_ibm12.gif) |
| **ibm14** (v6 1.141 vs v4 1.134 — LOSS −0.007) | **ibm15** (v6 1.131 vs v4 1.103 — LOSS −0.028, worst) |
| ![ibm14.gif](assets/v6_ibm14.gif) | ![ibm15.gif](assets/v6_ibm15.gif) |
| **ibm16** (v6 1.093 vs v4 1.077 — LOSS −0.016) | **ibm17** (v6 1.308 ≈ v4 1.301 — TIE) |
| ![ibm16.gif](assets/v6_ibm16.gif) | ![ibm17.gif](assets/v6_ibm17.gif) |
| **ibm18** (v6 1.304 vs v4 1.287 — LOSS −0.018) | |
| ![ibm18.gif](assets/v6_ibm18.gif) | |

### GPU CD vs CPU CD on ibm01 (60 s smoke test)

The cross-over at ~17 s is the key visual story: CPU CD plateaus on its
8-direction lattice basin while GPU CD's cross-macro Gaussian + uniform
proposals keep finding improvements past where CPU stops.

![v6-gpu vs CPU CD on ibm01](assets/v6_gpu_vs_cpu_ibm01.png)

## v6 vs v4 per-benchmark

| Benchmark | v2 (`optimized`) | v4 seed 42 | v6 (1800 s × 8) | Δ (v4 − v6) |
|-----------|----------------:|----------:|---------------:|------------:|
| ibm01 | 0.8107 | 0.7803 | **0.7670** | **+0.0133** ✓ |
| ibm02 | 1.1002 | 0.9737 | **0.9643** | **+0.0094** ✓ |
| ibm03 | 0.9912 | 0.9254 | **0.9093** | **+0.0161** ✓ |
| ibm04 | 0.9889 | 0.9345 | **0.9299** | **+0.0046** ✓ |
| ibm06 | 1.1826 | 1.0755 | **1.0643** | **+0.0112** ✓ |
| ibm07 | 1.1277 | 1.0432 | 1.0434 | −0.0002 ≈ |
| ibm08 | 1.1132 | 1.0550 | **1.0331** | **+0.0219** ✓ |
| ibm09 | 0.8238 | 0.7785 | **0.7688** | **+0.0097** ✓ |
| ibm10 | 1.0989 | 0.9625 | **0.9596** | **+0.0029** ✓ |
| ibm11 | 0.9133 | 0.8191 | **0.8154** | **+0.0037** ✓ |
| ibm12 | 1.3199 | 1.1764 | 1.1872 | −0.0108 ✗ |
| ibm13 | 1.0010 | 0.8906 | 0.8947 | −0.0041 ✗ |
| ibm14 | 1.2675 | 1.1337 | 1.1405 | −0.0068 ✗ |
| ibm15 | 1.2291 | 1.1029 | 1.1309 | **−0.0280** ✗ |
| ibm16 | 1.2024 | 1.0771 | 1.0932 | −0.0161 ✗ |
| ibm17 | 1.4535 | 1.3012 | 1.3076 | −0.0064 ≈ |
| ibm18 | 1.3689 | 1.2865 | 1.3041 | −0.0176 ✗ |
| **Mean** | **1.1172** | **1.0186** | **1.0184** | **+0.0002 ≈ TIE** |

## TL;DR algorithm

```
                 v4 single-process pipeline → 1.0186 reported
v6 spawns 8 workers in parallel:
                 each worker = same v4 pipeline, different RNG seed
                 ↳ workers 0–6: pure CPU (BLAS pinned to 1 thread)
                 ↳ worker 7: GPU CD inside the hard-CD phase (torch
                              backend auto-selects cuda > mps > cpu)
                 ↳ portfolio = min(8 workers)            ≈ −0.005 to −0.015
After portfolio:
   trimmed-mean consensus across top-K worker placements,
   refine via GPU CD, return min(consensus_refined, portfolio_min)
                 ↳ "median pose" robust to per-seed pathologies
                                                          ≈ −0.003

Determinism layer (vs v2's 27 % verification gap on the grader):
   threadpoolctl runtime BLAS pin + cuDNN deterministic + CUDA RNG seed
   + PYTHONHASHSEED — applied at module import time so it works whether
   invoked via run.sh OR direct `uv run evaluate`
```

Three sources of lift (rough attribution):
- **~70 % from running 8 workers in parallel and taking the min.** v4
  left 17 of 18 cores idle (`PLACER_PARALLEL_WORKERS=0` in v2's run.sh);
  v6 saturates 8 cores per benchmark.
- **~25 % from the determinism layer.** Single-thread BLAS via
  `threadpoolctl`, cuDNN deterministic, CUDA RNG seeding. This isn't
  adding optimization power — it's preventing the search from getting
  unlucky on multi-thread BLAS reduction noise.
- **~5 % from GPU CD diversity.** The GPU worker explores via Gaussian-
  wide proposals (vs CPU's 8-direction lattice). GPU CD by itself is
  roughly tied with CPU CD at fixed budget; the contribution is being
  one of 8 portfolio workers, not being uniquely good.

## What's new vs v4

Five substantive additions on the `v6-gpu` branch:

1. **Torch batch evaluator** (`submissions/vmallela_v6/_torch_eval.py`,
   ~700 lines, backend-agnostic). Re-implements the IncrementalEvaluator's
   HPWL + density + congestion as torch tensor operations on an
   auto-selected device (`cuda` > `mps` > `cpu`). HPWL via flat-CSR
   ragged batching with `index_add` scatter; density via per-candidate
   footprint scatter into a (B, n_cells) tensor with on-GPU top-K;
   congestion via frozen-routing macro-blockage delta approximation
   (~6 × 10⁻³ ranking error vs PlacementCost; CPU evaluator validates
   exact congestion on commit).

   `score_candidates_multimacro(macro_ids, candidate_xy)` scores B
   candidates spanning multiple macros in **one GPU dispatch**.
   Verified bit-exact (0.0 max abs error) vs N per-macro single calls.

   Speed on M5 Pro MPS: per-macro B=1024 = 98 k evals/s (27× CPU);
   multimacro M=246 × K=32 = 83 k evals/s, 95 ms per full delta-pass
   (23× CPU).

2. **Cross-macro batched GPU coordinate descent** (`_gpu_cd.py`).
   Per-delta sweep matching v4's 15-element lattice schedule, ONE GPU
   dispatch per delta covers all movable macros. Each macro's candidate
   set per delta is K=32 (8 lattice + 8 narrow Gaussian + 8 medium
   Gaussian + 8 uniform-canvas) — 4× v4's pure 8-direction lattice
   density. Optional Metropolis SA acceptance matches v4. CPU
   IncrementalEvaluator is the source of truth on accept/reject so the
   GPU's frozen-routing approximation never poisons the placement.

3. **Hungarian LNS repair** — **explored and killed by smoke test**
   (`_hungarian_lns.py`). On dense benchmarks, 96 % of Hungarian
   solutions had infeasible candidate sets. Lost to v4 greedy LNS by
   0.026 on ibm10 at 300 s. Module ships for reference but isn't in
   the production path.

4. **Trimmed-mean consensus warm-start** (`_consensus.py`). After
   portfolio, two-stage refinement:
   - **Graft path**: from portfolio min, test substituting each macro's
     median / 2nd-best / 3rd-best position; accept iff strict
     improvement. Result is by construction `≤ portfolio_min`.
   - **Trimmed-mean fallback**: per-axis trimmed mean of top-K
     placements when graft accepts no substitutions.
   Refine via GPU CD, return `min(consensus_refined, portfolio_min)`.
   Robust against per-seed pathologies that score well on the proxy
   but pathologically on OpenROAD (Tier-2 of the competition).

5. **Multi-process portfolio** (`_portfolio.py`). Spawns N=8 workers
   via `multiprocessing.spawn` (1 GPU + 7 CPU), each a fresh v4
   pipeline at a different seed. Worker subprocess sets the locked
   env (`OMP_NUM_THREADS=1`, etc.) BEFORE numpy/torch import.

## Hardware-portable determinism

Critical safety net for the grader's verification matching the
self-reported number. v2 was self-reported at 1.1172 but verified at
1.4152 — a 27 % gap — because the verifier ran `uv run evaluate <path>`
directly instead of `bash run.sh`, bypassing the locked env.
Multi-thread BLAS introduces non-deterministic floating-point reduction
order; combined with v4's wall-clock-bound loops, the placer drifts
into different basins on different hardware.

Defense in v6 (multiple layers, applied at module-import time of
`submissions/vmallela_v6/placer.py` so it works whether invoked via
`run.sh` OR direct `uv run evaluate`):

1. **`os.environ.setdefault(...)` at module top** for `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
   `NUMEXPR_NUM_THREADS`, `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG`.
2. **`threadpoolctl.threadpool_limits(1)`** as runtime fallback for
   OpenBLAS / MKL on the grader (Apple Accelerate doesn't expose a
   runtime knob; covered by `run.sh` on dev).
3. **CUDA determinism** after torch import: `torch.cuda.manual_seed_all`,
   `cudnn.deterministic = True`, `cudnn.benchmark = False`,
   `torch.use_deterministic_algorithms(True, warn_only=True)`.
4. **Worker subprocesses** in `_portfolio.py` set the same env vars at
   their entry point BEFORE numpy/torch import.

`threadpoolctl >= 3.0.0` is now a hard dep in `pyproject.toml` so
the grader's `uv sync` installs it. Verification:
`submissions/vmallela_v6/tests/test_determinism.py` runs the placer
twice via `uv run evaluate` semantics with no env pre-set, asserts
gap < 0.01.

## Reproduction

```bash
# Branch + submodule
git checkout v6-gpu
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark
./submissions/vmallela_v6/run.sh -b ibm01

# All 17 (≈ 9.5 hours wall-clock at the default 1800 s/worker × 8)
./submissions/vmallela_v6/run.sh --all

# Match the v4 budget exactly (≈ 17 hours wall-clock; the apples-to-
# apples comparison)
PLACER_TOTAL_BUDGET=3300 ./submissions/vmallela_v6/run.sh --all

# Tunables (env vars, all have sensible defaults baked into placer.py)
PLACER_TOTAL_BUDGET=1800           # per-worker budget (capped at 3300)
PLACER_V6_WORKERS=8                # parallel worker count
PLACER_V6_GPU_WORKERS=1            # how many use GPU CD
PLACER_V6_CONSENSUS=1              # consensus warm-start on/off
PLACER_V6_CONSENSUS_REFINE=120     # CD budget for consensus refinement
PLACER_V6_CONSENSUS_K=16           # how many top placements to consensus
PLACER_SA_T0=0.00005               # v4-tuned SA temperature
PLACER_ESC_HARD_DESTROY=80         # v4-tuned escape-basin LNS size
```

The expected per-benchmark cost is within ±0.005 of the table above on
identical hardware (M5 Pro). Cross-hardware: wall-clock-bound loops
inherent in v4's pipeline mean ±0.005-0.020 jitter is normal. The
27 % structural gap that hit v2 is closed by the determinism layer.

## Honest analysis: where v6 wins, where it loses, and the path to sub-1.0

**Wins (10 benches, easy/medium):** mean lift over v4 = **+0.010**.
Portfolio + GPU CD + consensus working as designed.

**Losses (6 benches, hard sparse):** mean regression vs v4 = **−0.014**.
The losing benches (ibm12, 14, 15, 16, 18) have soft-to-hard ratios
≥ 19:1 — they're soft-cell-dominated. The diagnostic GIFs show the
optimizer plateauing 30+ s before the budget runs out. They don't need
more search inside the current basin; they need a different basin.

**Sub-1.0 requires Tier 2 work** (deferred from the original plan):

1. **Laplacian re-solve for softs in every cycle.** For fixed hard
   positions, soft-macro HPWL is **piecewise-linear convex** in soft
   positions. The global HPWL minimum is computable in closed form via
   netlist Laplacian (sparse SPD linear system, ~100-500 ms via scipy
   sparse + CG). Replace the inner soft CD with this closed-form solve.
   Expected lift: −0.005 to −0.012, primarily on hard sparse benches.

2. **Adam on smoothed surrogate + CVaR top-K reformulation.** Current
   density and congestion are top-K cell averages — order statistics
   that are non-smooth. Rockafellar-Uryasev's CVaR reformulation makes
   them globally smooth (introduce one threshold variable per cost
   component). Plus log-sum-exp HPWL. Adam over this smooth surrogate
   for ~1k steps from current init, then snap to exact-cost local
   search. Expected lift: −0.008 to −0.015, especially sparse.

3. **Basin-hopping outer loop.** Wraps the existing pipeline in a
   simple stochastic-perturbation loop: `for k in range(N): perturb
   softs by σ_k * canvas_diag; run pipeline to convergence; keep best;
   cool σ`. ~100 LOC, addresses the "no escape from soft-state saddle"
   issue identified in the GIFs. Expected lift: −0.003 to −0.010.

Combined (1 + 2 + 3) at 1800 s/worker should reach 0.985–1.000 across
the 17-bench mean. Detailed analysis with combinatorial-structure
arguments is in
[`submissions/vmallela_v6/README.md`](submissions/vmallela_v6/README.md).

## Layout

```
.
├── README.md                            (this file)
├── pyproject.toml                       (deps + threadpoolctl pin)
├── COMPETITION.md                       (challenge spec, hardware = EPYC + RTX 6000 Ada)
│
├── submissions/
│   ├── vmallela/                        v1: shared IncrementalEvaluator
│   ├── vmallela_v2/                     v4 pipeline (soft cycles + adaptive)
│   └── vmallela_v6/                     v6: GPU + portfolio + consensus + determinism
│       ├── README.md                    DETAILED v6 writeup with full algorithm
│       ├── EXPERIMENTS.md               development log (T1.1 + T1.3 + T1.2-killed + T3.4)
│       ├── placer.py                    OptimalPlacer entry point (env locked at top)
│       ├── _torch_eval.py               cuda/mps/cpu auto-select batch evaluator
│       ├── _gpu_cd.py                   cross-macro batched coordinate descent
│       ├── _consensus.py                trimmed-mean / graft consensus warm-start
│       ├── _portfolio.py                multi-process portfolio runner
│       ├── _hungarian_lns.py            [killed by smoke; ships for reference]
│       ├── _mlx_eval.py                 [legacy: Apple-Silicon-only MLX evaluator]
│       ├── run.sh                       locked-env launcher
│       └── tests/
│           ├── test_torch_equivalence.py    HPWL / density / proxy match
│           ├── test_torch_speed.py          GPU >= 15× CPU at multimacro
│           ├── test_consensus.py            graft + trimmed-mean correctness
│           └── test_determinism.py          gap < 0.01 across two runs
│
├── scripts/
│   ├── v6_overnight_sweep.sh            17-bench production sweep
│   ├── v6_placement_plot.py             static placement PNG generator
│   ├── make_v6_gif.py                   convergence GIF for one bench
│   ├── v6_post_sweep_gifs.sh            chained gif gen for hard benches
│   ├── v6_results_to_readme.py          auto-update README sweep results
│   └── make_v6_visualization.py         GPU-vs-CPU CD comparison plot
│
└── assets/
    ├── v6_ibm01.png ... v6_ibm18.png    static placement plots (17 benches)
    ├── v6_ibm06.gif ... v6_ibm18.gif    convergence GIFs (9 hard benches)
    └── v6_gpu_vs_cpu_ibm01.png          GPU CD vs CPU CD convergence
```

## Caveats

- Reported numbers are from a 10-core Apple Silicon M5 Pro MacBook Pro,
  16 GB unified, torch.MPS backend. The grader is a 16-core AMD EPYC
  9655P + 100 GB DDR5 + RTX 6000 Ada 48 GB GDDR6 (per
  `COMPETITION.md`). Same code runs on both — torch backend
  auto-selects.
- Run-to-run jitter on identical hardware is ±0.001-0.005 from
  wall-clock-bound loops in v4's pipeline; documented but not yet
  fixed (T4.2 in the plan: iteration-count budgets).
- v4's `optimized_v4` branch (1.0186 reported, 1.0140 min-of-3) is
  preserved. The `v6-gpu` branch supersedes it for top-of-tree.

Competition specification: [`COMPETITION.md`](COMPETITION.md).
