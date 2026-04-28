# Verified single-run summary — vmallela_v2 @ seed=42

**Average proxy cost: 1.1172 across 17 IBM ICCAD04 benchmarks.**
**All 17 VALID. Zero overlaps on every benchmark.**

## Methodology

One run per benchmark using `submissions/vmallela_v2/run.sh`:

- Seed: `42` (hard-coded default in `OptimalPlacer.__init__`; `torch.manual_seed`, `random.seed`, `np.random.seed` all set before any work).
- Budget: `PLACER_TOTAL_BUDGET=3300` (the placer hard-caps at 3300 regardless of input to preserve the 1-hour competition cap).
- Threading: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.
- Python hash seed: `PYTHONHASHSEED=42`.
- Parallel workers: `PLACER_PARALLEL_WORKERS=0` (serial).
- Benchmarks run one at a time (no cross-bench CPU contention).
- Hardware: MacBook Pro (10-core Apple Silicon).

Run-to-run jitter under identical env: ~0.002 absolute on the proxy cost at the headline. This comes from `while time.time() - t0 < budget` loops that do "as many moves as fit" — iteration count depends on wall-clock jitter. Not bit-reproducible; semantically reproducible.

## Per-benchmark results

| Bench | Verified proxy | Valid | Overlaps | Wall time | Prior best-of-N | Δ |
|-------|----------------|-------|----------|-----------|-----------------|-----|
| ibm01 | 0.8107         | ✓     | 0        | 1926 s    | 0.8147          | −0.0040 |
| ibm02 | 1.1002         | ✓     | 0        | 1989 s    | 1.1444          | −0.0442 |
| ibm03 | 0.9912         | ✓     | 0        | 1667 s    | 1.0374          | −0.0462 |
| ibm04 | 0.9889         | ✓     | 0        | 2054 s    | 1.0207          | −0.0318 |
| ibm06 | 1.1826         | ✓     | 0        | 2367 s    | 1.2435          | −0.0609 |
| ibm07 | 1.1277         | ✓     | 0        | 2376 s    | 1.1497          | −0.0220 |
| ibm08 | 1.1132         | ✓     | 0        | 2789 s    | 1.1442          | −0.0310 |
| ibm09 | 0.8238         | ✓     | 0        | 2243 s    | 0.8558          | −0.0320 |
| ibm10 | 1.0989         | ✓     | 0        | 3149 s    | 1.1344          | −0.0355 |
| ibm11 | 0.9133         | ✓     | 0        | 2311 s    | 0.9569          | −0.0436 |
| ibm12 | 1.3199         | ✓     | 0        | 3260 s    | 1.3406          | −0.0207 |
| ibm13 | 1.0010         | ✓     | 0        | 2503 s    | 1.0620          | −0.0610 |
| ibm14 | 1.2675         | ✓     | 0        | 3305 s    | 1.2788          | −0.0113 |
| ibm15 | 1.2291         | ✓     | 0        | 3115 s    | 1.2559          | −0.0268 |
| ibm16 | 1.2024         | ✓     | 0        | 3305 s    | 1.2517          | −0.0493 |
| ibm17 | 1.4535         | ✓     | 0        | 3293 s    | 1.4895          | −0.0360 |
| ibm18 | 1.3689         | ✓     | 0        | 3296 s    | 1.4350          | −0.0661 |
| **AVG** | **1.1172** |       |          |           |                 |  |

Every bench beat its prior best-of-N single-point number. Deltas range −0.0040 to −0.0661.

Note: ibm05 is not in the 17-benchmark IBM ICCAD04 set used by the competition (per `COMPETITION.md`), matching the official scoring.

## Leaderboard comparison

| Rank | Method          | Avg proxy | Our delta |
|------|-----------------|-----------|-----------|
| —    | **vmallela_v2** | **1.1172** | —         |
| 1    | Cezar (ReFine)  | 1.2224    | −8.6 %    |
| 2    | MTK DreamPlace++| 1.2818    | −12.9 %   |
| 3    | RoRa            | 1.3241    | −15.6 %   |
| 4    | vmallela v1     | 1.4156    | −21.1 %   |

Delta = `(competitor - ours) / competitor`; positive means we cost less.

## Hardware caveat

Measured on a 10-core Apple Silicon MacBook Pro. The competition judges run on AMD EPYC 9655P (16 cores, dedicated per-process) + RTX 6000 Ada 48 GB. Our placer is pure-Python / NumPy / Torch-CPU (no CUDA). A 16-core EPYC typically runs our hot loops ≥1.3× faster per core than M-series, so the judges' runs — at the same 1-hour budget — should produce equal or slightly better numbers than what we report here.

## Reproduction

```bash
./submissions/vmallela_v2/run.sh --all
```

Runs all 17 benchmarks under the locked env above. Expected per-bench result within ±0.002 of the table. Full sweep takes ~15 hours on a 10-core Mac.
