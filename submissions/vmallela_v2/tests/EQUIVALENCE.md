# IncrementalEvaluator ↔ PlacementCost equivalence

**Result:** PASS. The incremental evaluator used by vmallela/vmallela_v2
(`IncrementalEvaluator` in `submissions/vmallela/placer.py`) returns proxy
costs that agree with the canonical batch evaluator
(`macro_place.objective.compute_proxy_cost`, which wraps TILOS `PlacementCost`)
to within **2.75 × 10⁻⁷ absolute** across all 303 comparisons performed —
roughly 363× tighter than the 1 × 10⁻⁴ tolerance we care about in practice.

## Methodology

Test: `submissions/vmallela_v2/tests/test_evaluator_equivalence.py`
Date: 2026-04-23

For each of three benchmarks (selected to span size/shape):

| Benchmark | Hard macros | Canvas       | Role          |
|-----------|-------------|--------------|---------------|
| ibm01     | 11          | 16.4 × 16.5  | small         |
| ibm06     | 178         | 32.6 × 32.6  | medium        |
| ibm10     | 786         | 77.0 × 77.1  | large         |

1. Load benchmark and its initial `.plc` via `load_benchmark_from_dir`.
2. Construct `IncrementalEvaluator(plc, benchmark)`; record the initial
   proxy cost (`get_proxy_cost()`).
3. Compute batch cost from the same initial placement via
   `compute_proxy_cost(placement, benchmark, plc)`; record the difference.
4. Apply **100 random hard-macro moves** (seeded `random.Random(42)`):
   - choose a movable hard macro uniformly,
   - pick a canvas-bounded random displacement in ±20% of the canvas per
     axis (clamped to `[0, canvas − macro_size]`),
   - apply the move to the incremental evaluator (`move_macro(idx, nx, ny)`),
     and the same move to a mirrored placement tensor,
   - re-evaluate the batch cost from the mirrored tensor,
   - record `|batch − incr|` (abs) and `|batch − incr| / max(|batch|, 1e-9)`
     (rel).
5. Report the per-benchmark maxima.

Env: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`. Run at `nice -n 10` on the
same MacBook Pro that is running the verification sweep concurrently
(single-threaded BLAS prevents contention; no measurable slowdown in the
sweep).

## Results

| Benchmark | Initial abs Δ | Max abs Δ | Max rel Δ | Wall time |
|-----------|---------------|-----------|-----------|-----------|
| ibm01     | 2.41 × 10⁻⁷   | **2.75 × 10⁻⁷** | 2.60 × 10⁻⁷ |   134.8 s |
| ibm06     | 1.78 × 10⁻⁷   | **2.36 × 10⁻⁷** | 1.36 × 10⁻⁷ |   396.0 s |
| ibm10     | 1.12 × 10⁻⁷   | **2.64 × 10⁻⁷** | 1.96 × 10⁻⁷ |  2660.9 s |

Verdict: **PASS**. No accumulation of drift was observed across the 100-move
sequences — the max diff is already set within the first few moves on each
benchmark and does not grow. Consistent with the observation that the
incremental updates are algebraically equivalent to the full recomputation
(up to ordering of floating-point additions).

## Source of the residual 10⁻⁷ gap

The absolute error of 2-3 × 10⁻⁷ is consistent with IEEE-754 single-precision
(float32) rounding order differences:

- `IncrementalEvaluator` stores macro positions as `float32` (to match
  PlacementCost's internal grid-cell quantisation), but accumulates net HPWL
  and grid density as `float64`.
- `PlacementCost` accumulates in its own order when called full-batch.
- Identical inputs + identical formulas + different summation order give
  bit-different but numerically equivalent float64 results.

This gap does not grow with problem size (ibm01 11 macros → 2.75 × 10⁻⁷;
ibm10 786 macros → 2.64 × 10⁻⁷) and does not grow with move count, which
rules out a logic bug. It is the expected floating-point noise floor.

## Reproducing

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python submissions/vmallela_v2/tests/test_evaluator_equivalence.py
```

Exits 0 on PASS (max abs Δ < 1 × 10⁻⁴ on every benchmark) and nonzero on
divergence. The script prints a per-move progress line every 25 moves and
the final summary.
