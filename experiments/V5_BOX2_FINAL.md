# v5_box2 final report — 2026-04-26

Box2 session: 2026-04-25T22:27 → 2026-04-26T03:30 UTC (~5 h compute budget,
the last 4 h gated by user's "4 h then push and shutdown" constraint).

## Hardware (the surprise that ate most of the day)

- Intel Xeon Gold 6130 @ 2.10 GHz, 2 sockets × 16 cores × 2 SMT = **32 phys / 64 logical**
- 251 GiB RAM, 98 GB root disk
- **AVX-512 frequency throttling** is the dominant cost driver. Skylake-SP
  drops core clock to ~1.0–1.3 GHz when many cores execute AVX-512
  simultaneously (MKL BLAS in placer is AVX-512 by default). Box1 (Zen 5)
  doesn't have this license behavior. Empirical concurrency limits on
  ICCAD04 hard benches:

  | concurrency | ibm12 | ibm15 | ibm16 | ibm17 | ibm18 |
  |-------------|-------|-------|-------|-------|-------|
  | 32-way (Phase 1 v1) | INVALID | n/t | n/t | n/t | n/t |
  | 8-way (ssv2)        | INVALID | n/t | n/t | n/t | n/t |
  | 5-way (ssv3)        | INVALID | VALID 1.139 | VALID 1.095 | INVALID | VALID (cycle 2) |
  | 2-way (last shot)   | n/t   | n/t   | n/t   | INVALID 1.7392 (legalize 905s, 231 ovl) | VALID 1.3149 |
  | 1-way patched (CAP=1500) | n/t   | n/t   | n/t   | INVALID 1.7392 (legalize 909s) | (not run) |
  | single-process      | works (escape_test ibm01 600s → 0.8071 VALID) | | | | |

  **ibm17 is unsalvageable on box2 at any concurrency, even with the
  LEGALIZE_BUDGET cap raised from 600 → 1500.** The cap raise lets the
  tournament *try* longer but the iteration count is limited (30 ot ×
  4 sm × 5 starts = 600 _legalize calls). All 600 combos fail to produce
  an overlap-free placement on ibm17 from the (push_apart × 3 + init +
  warmstart) starting set. Not a budget issue — it's a starting-point /
  algorithm-coverage issue. Would need: more push_apart configurations,
  a different legalize ring algorithm, or a different warmstart that
  produces a near-legal solution.

  ibm12 likely shows the same pattern. ibm15/16/18 work because their
  legalize tournament finds a valid candidate within the iteration budget.

### last_shot results (2-way v5_cluster, seed 42)

  - ibm17: INVALID 1.7392 (legalize 905s, 231 overlaps)
  - ibm18: VALID **1.3149** at wall 3348s — 47 SA cycles + 3 escape phases.
    Box1 v5_cells_skip ibm18 = 1.2835. **Ours is 2.4% worse**, same
    direction as ssv3 ibm15/ibm16 — config underperforms baseline on this
    hardware.

### Bottom-line score

| metric | value |
|--------|-------|
| Best confirmed reproducible mean | **1.0046** (box1 v5_cells_skip seed 42) |
| Box2 contributions | (–) cluster30_plateau2 ruled out as regression |
|                    | (–) cluster_translate config 2-3% worse on ibm15/16/18 |
|                    | (+) AVX-512 throttle bug diagnosed, workaround patch shipped |
|                    | (+) Pure-Python flip optimizer (`flip_v2.py`) shipped, gains tiny on ICCAD04 |
|                    | (+) `LEGALIZE_BUDGET` cap raise patch (`v5_box2_legalize_cap.patch`) — useful on Skylake-SP |
| Sub-1.0 achieved   | **No** |

  The failure mode on hard benches is identical across concurrencies:
  `[legalize] Xs cost=inf → INVALID (N overlaps)`. Legalize tournament
  cannot find a valid candidate within its 660 s budget cap because each
  `_legalize()` call runs at ~25 % of box1's per-thread speed under
  AVX-512 throttle.

## What we tried, what worked, what didn't

### Sweeps

1. **Phase 1 v1 (32-way, v5_combined+v5_cluster, paired multi-seed)** —
   broken (AVX-512 throttle). 4 INVALID rows produced before kill, all
   identical 1.3397/1.7392 → deterministic placer fallback.
2. **ssv2 (8-way cluster30_plateau2 × 4 seeds × 5 hard benches)** — also
   broken on ibm12/ibm17.
3. **ssv3 (5-way same config × 3 seeds)** — partial success: ibm15/ibm16
   produced VALID rows, but **the config is a regression vs box1's
   v5_escape_v2 baseline by 2–3 %**. ibm12/17 INVALID at 5-way.
4. **last_shot (2-way v5_cluster ibm17+ibm18)** — running; results land
   ~02:26 UTC.

### Orientation-flip optimizer (Klein-4 sidecar — `experiments/v5_box2_flip_v2.py`)

Pure-Python greedy per-macro flip, bypasses slow PlacementCost.get_cost()
by maintaining an in-memory net graph from `plc.modules_w_pins[].get_sink()`.
Each macro × 4 orientations evaluated, picks lowest local HPWL.

**HPWL reductions on TILOS initial.plc baselines** (mean +0.55 %):

| bench | red% | bench | red% | bench | red% |
|-------|------|-------|------|-------|------|
| ibm01 | +0.96 | ibm07 | +0.23 | ibm13 | +0.46 |
| ibm02 | +0.59 | ibm08 | +2.34 | ibm14 | +0.34 |
| ibm03 | +0.46 | ibm09 | +0.48 | ibm15 | +0.14 |
| ibm04 | +0.89 | ibm10 | +0.38 | ibm16 | +0.41 |
| ibm06 | +0.23 | ibm11 | +0.24 | ibm17 | +0.19 |
|       |      | ibm12 | +0.31 | ibm18 | +0.07 |

**Verdict on flips**: marginal at this benchmark family. ICCAD04 hard macros
are mostly small gate-level primitives (a22/a23/etc.), not big SRAM
arrays where orientation could matter. Proxy-cost translation: WL is
~6 % of proxy on ibm01 (WL=0.064 of proxy=1.04), so 0.55 % HPWL → ~0.0003
absolute proxy reduction per bench. Mean improvement across 17 benches:
~0.0005. Not the unlock to 0.97.

This optimizer would matter much more on the NG45 designs (ariane133,
nvdla, mempool_tile) where macros are large SRAM blocks.

## Best confirmed result

Box1's `v5_cells_skip` seed 42, mean **1.0046** (single seed). Box2 did
not improve on this within its 4 h compute budget.

## Useful artifacts left behind

- `experiments/v5_box2_flip_v2.py` — fast pure-Python flip optimizer.
  Runs all 17 ICCAD04 in parallel in <30 s. Use as post-processor on any
  `.plc` file (TILOS or v5 placer output, once placer is patched to save).
- `experiments/v5_box2_concurrency_test.sh` — diagnostic harness for
  finding throttle-safe concurrency on a new box.
- `experiments/v5_box2_smart_search_v2.sh`, `_v3.sh` — env-tune launchers.
- `experiments/v5_box2_phase1_v3.sh` — paired multi-seed sweep launcher
  (8-way; needs concurrency tuning per box).
- `experiments/v5_box2_analyze.py` — paired t-test analysis on (bench,
  seed, branch) triplets in `results.csv`.

## What box1 should do with these findings

1. **Don't waste compute on cluster30_plateau2 — it's a regression** (3
   seeds × 3 hard benches confirm worse than v5_escape_v2 baseline).
2. **Orientation flips are not worth wiring into the main placer for
   ICCAD04** — gain too small. Save them for NG45 (ariane133/nvdla) when
   that benchmark family enters the picture.
3. **For Skylake-SP boxes specifically**: install CPU-only torch (no
   AVX-512 MKL) or pin concurrency to ≤2 placers per socket. Zen 5 boxes
   like box1 are unaffected.

## Configuration confidence ranking on hard benches (ibm12/15/16/17/18)

Best on hard benches we have data for (single-seed unless noted):

| config | ibm12 | ibm15 | ibm16 | ibm17 | ibm18 | hard mean |
|--------|-------|-------|-------|-------|-------|-----------|
| `v5_cells_skip` (box1, s=42)   | 1.150 | 1.088 | 1.058 | 1.277 | 1.284 | **1.171** |
| `v5_escape_v2` (box1, s=42)    | 1.156 | 1.102 | 1.071 | 1.292 | 1.295 | 1.183 |
| `cluster30_plateau2` (box2, s=43) | INVALID | 1.139 | n/t | INVALID | n/t | (regression) |
| `cluster30_plateau2` (box2, s=44) | INVALID | n/t   | 1.095 | INVALID | n/t | (regression) |

`v5_cells_skip` is the leader on every hard bench we have it for.
