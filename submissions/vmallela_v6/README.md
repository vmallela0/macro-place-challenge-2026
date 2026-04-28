# vmallela_v6 — GPU + Multi-Process Portfolio Macro Placer

Author: vmallela
Branch: `v6-gpu`
Builds on: `submissions/vmallela_v2/` (v4 pipeline) + a new torch-based batch
evaluator (cross-macro batched, backend-agnostic CUDA/MPS/CPU) and a
multi-process portfolio runner.

## What's new vs v4

v4's `submissions/vmallela_v2/run.sh` set `PLACER_PARALLEL_WORKERS=0`,
leaving 17 CPU cores and the entire GPU idle on a 16-core EPYC + RTX 6000 Ada
grader. The v4 HANDOFF.md flagged the multi-worker portfolio as the
highest-EV unspent lever (-0.005 to -0.015 estimated). v6 wires it.

Three substantive additions:

1. **Torch batch evaluator** (`_torch_eval.py`, ~700 lines, backend-agnostic).
   Re-implements the IncrementalEvaluator's HPWL + density + congestion as
   torch tensor operations on an auto-selected device (`cuda` > `mps` > `cpu`).
   This is what runs on the grader's NVIDIA RTX 6000 Ada via torch.cuda; the
   M5 Pro dev box uses torch.MPS. **Same code path on both.**

   Two scoring entry points:
   - `score_candidates(macro_idx, candidate_xy)`: B candidates for one macro.
   - `score_candidates_multimacro(macro_ids, candidate_xy)`: B candidates
     spanning multiple macros in **one GPU dispatch** via flat-CSR ragged
     batching for HPWL and a max-tile padded approach for density/congestion.

   Verified equivalence to TILOS PlacementCost (`tests/test_torch_equivalence.py`):
   - HPWL: machine-precision exact (~1e-7 max abs over 60 random moves).
   - Density: machine-precision exact (~1e-7 max abs over 30 random moves).
   - Total proxy (frozen-routing congestion approximation): max abs error
     **6.0 × 10⁻³** over 20 random moves on ibm01. The CPU evaluator
     validates the exact congestion (with re-routing) on commit, so the GPU
     is never the source of truth — only a ranker.
   - **Multimacro vs per-macro: 0.0 max abs error** (bit-exact).

   Verified speed on M5 Pro MPS:
   - Per-macro B=1024: **98 428 evals/s** (27× CPU).
   - Multimacro M=246 × K=32: **82 622 evals/s**, **95 ms per full delta-pass**
     over all movable macros (23× CPU).
   - On the grader's CUDA RTX 6000 Ada: expect ~5-10× higher throughput.

   `_mlx_eval.py` (the original Apple-Silicon-only MLX evaluator) is kept for
   reference but no longer in the submission path — MLX cannot run on the
   grader, so any GPU contribution under MLX would have silently fallen back
   to CPU at submission time.

2. **Cross-macro batched GPU coordinate descent** (`_gpu_cd.py`). Replaces v4's
   8-direction CPU lattice CD with a per-delta sweep that issues **one GPU
   dispatch per delta level** covering all movable macros. The delta schedule
   mirrors v4's CPU CD lattice (15 levels from 5.0 to 0.02 macro_max_dim), so
   the search covers both long-range escape moves and sub-cell refinement.
   Each macro's candidate set per delta is K=32 (8 lattice + 8 narrow
   Gaussian + 8 medium Gaussian + 8 uniform-canvas) — 4× the proposal
   density of v4's pure 8-direction lattice. Optional Metropolis acceptance
   (PLACER_SA_T0) matches v4's simulated annealing.

   **ibm01 single-seed 60s smoke test**: GPU CD reaches **1.0165**, CPU CD
   reaches **1.0205** → **GPU wins by 0.0040** (target was GPU ≤ 1.019).
   The cross-over at ~17 s is the key visual story: CPU CD plateaus on its
   8-direction lattice basin while GPU CD's Gaussian + uniform proposals
   keep finding improvements past where CPU stops:

   ![v6-gpu vs CPU CD on ibm01](../../assets/v6_gpu_vs_cpu_ibm01.png)

   Cross-macro batching was the structural fix — the previous single-macro
   batched version lost to CPU CD by 0.005; this version wins by 0.004.

   The acceptance is strict against the CPU-exact proxy_cost — the GPU only
   ranks; rejected candidates incur no cost beyond the GPU score. This is
   the critical correctness property: the GPU's frozen-routing congestion
   approximation can never poison the placement.

3. **Hungarian LNS repair** — **explored and killed by smoke test**.
   Implementation lives at `_hungarian_lns.py` for reference. Replaces v4's
   greedy random-best-of-K reinsertion with min-cost-bipartite-matching
   over an `n_destroy x K` GPU-computed cost matrix solved via
   `scipy.optimize.linear_sum_assignment`. On dense benchmarks (ibm10 with
   778 fixed macros / 8 destroyed), 96% of Hungarian solutions are
   infeasible — uniform-random and net-centroid candidates almost always
   overlap a fixed macro, leaving no feasible candidate set for the
   assignment.

   **300s A/B on ibm10 (init 1.336748)**:
   - v4 greedy LNS: **1.272240** (12112 iters, 140 accepts, ~1% infeasible)
   - Hungarian LNS: **1.297937** (13796 iters, 67 accepts, **96% infeasible**)
   - Δ = +0.026 (v4 wins by 0.026)

   The plan's smoke-test bar was Hungarian wins by ≥0.003. Result misses
   the bar by ~1000%. Per the plan's stop condition ("smoke test fails by
   >50% → kill"), T1.2 is killed.

   Why it failed: Hungarian's separable-cost approximation breaks down on
   dense layouts where macros interact strongly, AND the candidate-pool
   construction is structurally hard when free space between fixed macros
   is small. Sparse benchmarks (ibm15-18, where free space is abundant)
   may still benefit — but the v6 submission keeps v4 greedy LNS on every
   worker. Future work: try Hungarian on **soft macros** (no overlap
   constraint → all candidates feasible by definition).

4. **Multi-process portfolio** (`_portfolio.py`, ~150 lines). Spawns N
   worker processes (default 8), each running the full v4 pipeline at a
   different RNG seed. One worker swaps in `gpu_mass_cd` for the hard-CD
   phase via a try/except wrapper — non-Apple-Silicon graders fall back
   transparently to the CPU v4 path. Workers run in parallel via
   `multiprocessing.spawn`; result is the lowest-cost overlap-free
   placement across all workers.

## Pipeline (per worker)

The same pipeline as v4 (`submissions/vmallela_v2/placer.py`):

```
Phase 1: Push-apart (3 dampings)
Phase 2: Legalization tournament (30 orderings × 4 step sizes × 4 starts)
Phase 3: Hard-macro CD + per-net step + LNS destroy-repair
Phase 4: Soft-macro adaptive cycles (FD/surrogate/CD/LNS/hard-polish)
Phase 5: Escape-basin LNS on plateau
```

The only change inside a GPU worker: Phase 3's `_coord_descent` is replaced
with `gpu_mass_cd(t=70%) → _coord_descent(t=30%)`. The 70/30 split runs the
GPU mass-search where its proposal diversity pays off (deltas ≥ macro size),
then hands off to the CPU lattice for fine-tuning at sub-cell deltas where
the structured 8-direction lattice still wins per micro-bench.

## Hardware mapping

| Component        | M5 Pro (this measurement)    | Grader (EPYC 9655P + RTX 6000 Ada) |
|------------------|------------------------------|------------------------------------|
| CPU cores        | 18 (6P + 12E)                | 16 (16P)                           |
| GPU              | 20-core Apple GPU (Metal 4)  | RTX 6000 Ada (48 GB GDDR6)         |
| RAM              | 48 GB unified                | 100 GB DDR5 + 48 GB GDDR6          |
| GPU compute      | ~7.4 TFLOPS via torch.MPS    | ~91 TFLOPS via torch.cuda          |
| Backend          | torch.MPS (auto-selected)    | torch.cuda (auto-selected)         |

The default `PLACER_V6_WORKERS=8` saturates 8 cores per benchmark and leaves
the remaining 8-10 for the OS + grader harness + GPU driver. The MLX worker
runs single-threaded on the CPU side and offloads its inner CD to the GPU.
Throughput target on Apple Silicon: ~250k proxy-score evaluations per second
per worker, vs ~3700/s on CPU.

## Reproduction

```bash
git checkout v6-gpu
git submodule update --init external/MacroPlacement
uv sync
uv pip install mlx scipy   # M5 Pro/M-series only

./submissions/vmallela_v6/run.sh -b ibm01     # single benchmark, 8 workers
./submissions/vmallela_v6/run.sh --all        # 17 IBM benchmarks, serially

# Tunables (env vars)
PLACER_TOTAL_BUDGET=3300        # per-bench wall-clock cap
PLACER_V6_WORKERS=8             # number of parallel workers
PLACER_V6_GPU_WORKERS=1         # how many use the GPU path (rest are pure-CPU v4)
PLACER_SA_T0=0.00005            # SA temperature inside hard CD (matches v4)
PLACER_ESC_HARD_DESTROY=80      # escape-basin LNS aggressiveness (matches v4)
```

## Tests

```bash
uv run python submissions/vmallela_v6/tests/test_mlx_equivalence.py
uv run python submissions/vmallela_v6/tests/test_gpu_speed.py
```

The equivalence test verifies HPWL <1e-7, density <1e-7, total proxy <2e-2
vs PlacementCost. The speed test asserts ≥20× GPU speedup at B=1024 (current
measurement: 62×).

## File layout

```
submissions/vmallela_v6/
├── README.md                       This file
├── placer.py                       OptimalPlacer entry point (portfolio
│                                   driver; spawns workers)
├── run.sh                          Locked-env launcher with v6 tuned defaults
├── _torch_eval.py                  TorchBatchEvaluator — backend-agnostic
│                                   (cuda/mps/cpu auto-select). Cross-macro
│                                   batched HPWL+density+approx-congestion.
├── _gpu_cd.py                      gpu_mass_cd: per-delta sweep over all
│                                   movable macros in one GPU dispatch each
├── _portfolio.py                   Multi-process portfolio runner
├── _mlx_eval.py                    [DEPRECATED] Apple-Silicon-only MLX
│                                   evaluator. Kept for reference.
└── tests/
    ├── test_torch_equivalence.py   HPWL/density/proxy correctness +
    │                               multimacro vs per-macro bit-exactness
    ├── test_torch_speed.py         GPU >= 15× CPU at multimacro
    ├── test_mlx_equivalence.py     [legacy]
    └── test_gpu_speed.py           [legacy]
```

## Caveats

1. **Frozen-routing congestion approximation.** The GPU's congestion proxy
   holds V/H_routing_smooth fixed and only updates the macro-blockage delta
   per candidate. This is exact for HPWL+density and approximate (~6e-3
   absolute on ibm01) for the congestion term. The CPU `IncrementalEvaluator`
   reroutes all affected nets on commit, so the *accepted* placement always
   has exact proxy cost — the approximation only affects ranking.

2. **Cross-platform via torch — same code on grader and dev.** The torch
   evaluator auto-selects `cuda` (grader: RTX 6000 Ada per COMPETITION.md),
   `mps` (M-series Macs), or `cpu` (fallback). torch.MPS gives ~7.4 TFLOPS
   on M5 Pro vs ~69 TFLOPS for native MLX, but the workload is memory-bound
   (sort + index_add dominate, not matmul) so the difference is small.
   torch.cuda on the grader gets the full RTX 6000 Ada.

3. **Per-worker memory.** Each spawn'd worker reloads PlacementCost
   (~200 MB resident on ibm17). 8 workers × 200 MB = 1.6 GB RSS, fine on
   48 GB unified or 100 GB grader RAM. ibm17/ibm18 may push to 2.5 GB total —
   still well within budget.

4. **The portfolio's wall-clock budget is per-worker.** All workers use the
   full PLACER_TOTAL_BUDGET; total real time = budget (since they run in
   parallel). The wall-clock-bound iteration counts in v4 inherit, so
   per-bench jitter ~±0.005 still applies.
