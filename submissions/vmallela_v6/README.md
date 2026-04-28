# vmallela_v6 — GPU + Multi-Process Portfolio Macro Placer

Author: vmallela
Branch: `v6-gpu`
Builds on: `submissions/vmallela_v2/` (v4 pipeline) + a new MLX/Metal batch
evaluator and a multi-process portfolio runner.

## What's new vs v4

v4's `submissions/vmallela_v2/run.sh` set `PLACER_PARALLEL_WORKERS=0`,
leaving 17 CPU cores and the entire GPU idle on a 16-core EPYC + RTX 6000 Ada
grader. The v4 HANDOFF.md flagged the multi-worker portfolio as the
highest-EV unspent lever (-0.005 to -0.015 estimated). v6 wires it.

Three substantive additions:

1. **MLX batch evaluator** (`_mlx_eval.py`, ~700 lines). Re-implements the
   IncrementalEvaluator's HPWL + density + congestion as MLX tensor operations
   that score B candidate single-macro moves in one GPU call. Per-(macro, net)
   "other-pin" extremes are precomputed in CSR form so HPWL deltas vectorize
   trivially. Density and congestion blockage are vectorized via numpy
   broadcasted scatter into per-candidate (B, n_cells) tensors, with MLX
   computing top-K cell sums on GPU.

   Verified equivalence to TILOS PlacementCost (`tests/test_mlx_equivalence.py`):
   - HPWL: max abs error **9.2 × 10⁻⁹** over 60 random moves on ibm01.
   - Density: max abs error **8.3 × 10⁻⁸** over 30 random moves.
   - Total proxy (with frozen-routing congestion approximation): max abs
     error **6.0 × 10⁻³** over 20 random moves. The CPU evaluator validates
     the exact congestion (with re-routing) on commit, so the GPU is never
     the source of truth — only a ranker.

   Verified speed: **226 503 evals/s at B=1024 on Apple M5 Pro 20-core GPU**,
   vs **3 618 evals/s** for the CPU IncrementalEvaluator → **62× speedup**.

2. **GPU mass-coordinate-descent** (`_gpu_cd.py`, ~250 lines). Replaces the
   8-direction CPU lattice CD with a Monte-Carlo proposal sweep: per macro
   per pass, generate K=384 candidates spanning 5 lattice deltas + narrow
   Gaussian (σ = macro size / 2) + medium Gaussian (canvas/8) + wide Gaussian
   (canvas/3) + uniform-canvas, score all on GPU in one batch, validate the
   top-T (default T=4) on the CPU IncrementalEvaluator, accept iff exact
   improves. Optional Metropolis acceptance (PLACER_SA_T0) matches v4's
   simulated annealing.

   The acceptance is strict against the CPU-exact proxy_cost — the GPU only
   ranks; rejected candidates incur no cost beyond the GPU score. This is
   the critical correctness property: the GPU's frozen-routing congestion
   approximation can never poison the placement.

3. **Multi-process portfolio** (`_portfolio.py`, ~150 lines). Spawns N
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
| GPU              | 20-core Apple GPU (Metal 4)  | RTX 6000 Ada (24 GB GDDR6)         |
| Unified memory   | 48 GB                        | 64 GB DDR5 + 24 GB GDDR6           |
| GPU compute      | ~69 TFLOPS sustained matmul  | ~91 TFLOPS sustained FP32          |
| MLX              | Native (Metal)               | Falls back to CPU; CUDA path TODO  |

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
├── README.md                     This file
├── placer.py                     OptimalPlacer entry point (portfolio
│                                 driver; spawns workers)
├── run.sh                        Locked-env launcher with v6 tuned defaults
├── _mlx_eval.py                  MLXBatchEvaluator — exact HPWL+density,
│                                 frozen-routing congestion approx
├── _gpu_cd.py                    gpu_mass_cd: Monte-Carlo proposal CD
│                                 backed by MLX evaluator + CPU validator
├── _portfolio.py                 Multi-process portfolio runner
└── tests/
    ├── test_mlx_equivalence.py   Equivalence vs PlacementCost
    └── test_gpu_speed.py         Speed regression: GPU >= 20× CPU at B=1024
```

## Caveats

1. **GPU CD alone underperforms CPU CD on ibm01 at fixed wall-clock budget**
   (1.024 vs 1.019 at 60s with SA). The GPU's value in v6 is portfolio
   diversity: it explores different basins via Gaussian-wide proposals than
   the CPU lattice CD. The lift comes from min-of-N across workers, not from
   replacing CPU CD on a single seed.

2. **Frozen-routing congestion approximation.** The GPU's congestion proxy
   holds V/H_routing_smooth fixed and only updates the macro-blockage delta
   per candidate. This is exact for HPWL+density and approximate (~6e-3
   absolute on ibm01) for the congestion term. The CPU `IncrementalEvaluator`
   reroutes all affected nets on commit, so the *accepted* placement always
   has exact proxy cost — the approximation only affects ranking.

3. **No CUDA path yet.** MLX is Apple Silicon. The grader is x86 + RTX. The
   GPU worker's `_gpu_cd_wrapper` catches MLX import errors and falls back
   to the CPU v4 path, so the portfolio still runs on x86 — but it loses
   the GPU diversity contribution. A CUDA port (~500 lines mirroring
   `_mlx_eval.py` in cupy or torch.cuda) is the obvious next step.

4. **Per-worker memory.** Each spawn'd worker reloads PlacementCost
   (~200 MB resident on ibm17). 8 workers × 200 MB = 1.6 GB RSS, fine on
   48 GB unified or 64 GB DDR5. ibm17/ibm18 may push to 2.5 GB total —
   still well within budget.

5. **The portfolio's wall-clock budget is per-worker.** All workers use the
   full PLACER_TOTAL_BUDGET; total real time = budget (since they run in
   parallel). The wall-clock-bound iteration counts in v4 inherit, so
   per-bench jitter ~±0.005 still applies.
