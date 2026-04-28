# vmallela_v6 — Experiments log

Mirrors the format of `submissions/vmallela_v2/EXPERIMENTS.md`. Documents
what was tried, what worked, what didn't, and the data behind each
decision.

---

## Tier 1: Engineering wins

### T1.1 + T1.3 — Cross-macro batched torch evaluator + per-delta sweep — ✅ shipped

**Hypothesis.** v6's previous single-macro batched MLX evaluator did
~50 k evals/s on ibm01 but lost to CPU CD by 0.005 at fixed budget because
the per-macro GPU dispatch overhead dominated. Cross-macro batching (one
GPU call covering all movable macros × K candidates per delta level)
should amortize the dispatch cost and let GPU CD strictly beat CPU CD.

**Implementation.** `_torch_eval.py` adds
`score_candidates_multimacro(macro_ids, candidate_xy)`. HPWL via
flat-CSR ragged batching (one entry per (candidate, net touching that
candidate's macro), `index_add` scatter back); density and congestion
via max-tile padded approach. Bit-exact (0.0 max abs error) vs N
per-macro single calls on ibm01.

**Result.**
- Speed (M5 Pro MPS): per-macro B=1024 = 98 k evals/s; multimacro M=246,
  K=32 = 83 k evals/s, **95 ms per full delta-pass over all 246 macros**.
  vs CPU IncrementalEvaluator 3.6 k evals/s = **23-27× speedup**.
- ibm01 60s SA T0=5e-5 (single-seed):
  - CPU CD: 1.0205
  - GPU CD: 1.0165
  - **GPU wins by 0.0040**, target was GPU ≤ 1.019 ✓
- Cross-over visible at ~17 s in `assets/v6_gpu_vs_cpu_ibm01.png`: CPU
  plateaus on its 8-direction lattice basin, GPU keeps finding
  improvements via cross-macro Gaussian + uniform proposals.

**Decision.** Shipped. Replaces single-macro MLX path; backend is now
torch.cuda (grader) / torch.mps (dev) / torch.cpu (fallback) auto-selected.

### T1.2 — Hungarian LNS repair — ❌ killed by smoke test

**Hypothesis.** v4's `_moves.lns_destroy_repair_phase` reinserts destroyed
macros greedy-best-of-K, committing each macro's position before
considering the others. A min-cost-bipartite-matching over an
`n_destroy x K` cost matrix (GPU-computed via `score_candidates_multimacro`,
solved with `scipy.optimize.linear_sum_assignment`) should beat greedy by
finding the joint optimal under the separable-cost approximation. Plan
expected gain: -0.005 to -0.015.

**Implementation.** `_hungarian_lns.py`. Pre-filter: pre-set C[i, j] = +inf
for (i) out-of-canvas candidates and (ii) candidates that overlap any
non-destroy macro for row i, so Hungarian only considers feasible
assignments. Within-destroy overlap checked post-Hungarian; sync_positions
rollback on infeasible. Two candidate-generation variants tried:
- v1: 6 net-centroid jittered + uniform-random over canvas → 96% infeasible
- v2: clustered around destroy-set current positions (small-jitter +
  net-centroid, no uniform) → still 96% infeasible

**Result on ibm10 (n_hard=786, fixed=778, init 1.336748):**

| Method | 30s cost | 300s cost | Δ vs v4 (300s) | Infeasibility |
|---|---|---|---|---|
| v4 greedy LNS | — | 1.272240 | — | ~1% |
| Hungarian v1 (random) | 1.311621 | 1.297937 | +0.026 (loses) | 96% |
| Hungarian v2 (clustered) | 1.308338 | (not run) | (extrapolates similar) | 96% |

**Why it failed.** On dense benchmarks, Hungarian's separable-cost
approximation breaks down — the marginal cost of moving macro i to
candidate j conditional on the others NOT moving doesn't predict the
joint cost of all destroy-set macros moving simultaneously. AND the free
space between fixed macros is too small for arbitrary candidate generation
to land in. Both v1 and v2 candidate strategies hit 96% infeasibility.

**Decision.** Killed per the plan's stop condition ("smoke test fails its
win bar by more than 50%: kill"). v4 greedy LNS retained on every worker
in the v6 portfolio. The Hungarian module ships for reference;
`_portfolio.py` does NOT monkey-patch the v4 LNS.

**Future revival paths.**
- Hungarian on **soft macros** (no overlap constraint → all candidates
  feasible by definition). Plausibly a strong T2 or T3 follow-on.
- Hungarian on **sparse benchmarks** (ibm15-18, low utilization, abundant
  free space). Untested.
- Iterative Hungarian: solve, identify infeasible assignment edges, add
  them as +inf, re-solve. Bounded iterations.

---

### T3.4 — Trimmed-mean consensus warm-start — ✅ shipped (with caveat)

**Hypothesis.** After running N=8-16 portfolio workers, the per-macro
trimmed-mean of the top-K cheapest placements should give a "consensus"
position that's better than the single best worker (if 75% of workers
agree on macro placement and 25% are pathological outliers, the trim
removes the outliers). When this consensus is then push-apart +
legalize'd and run through a final CD refinement, it should beat the
portfolio min cost.

**Implementation.** `_consensus.py`:
1. Sort N portfolio placements by proxy cost ascending; take top-K.
2. For each macro, compute trimmed-mean of (x, y) over those K
   placements (drop top 20% / bottom 20%).
3. Push-apart + legalize + refine_toward_initial → overlap-free
   starting point.
4. Run gpu_mass_cd (or CPU CD fallback) for `refine_max_time` seconds.
5. Return `min(consensus_refined, portfolio_min)` by cost — strict
   comparison.

Wired into `_portfolio.run_portfolio` as a post-portfolio step (default
on, controlled by `PLACER_V6_CONSENSUS`).

**Synthetic test** (`tests/test_consensus.py`, 8 placements at
target + Gaussian noise + 2 outliers stuck at ±99.0):
- Trimmed-mean recovered target within 1 sigma per macro.
- Outliers (extreme stuck macros) correctly trimmed.
- End-to-end on ibm01 with 8 perturbed legalized placements:
  portfolio min = 1.0721, **consensus refined to 1.0228 (Δ -0.049)**.

**Real-data smoke test results.**

Initial implementation had a soft-position-sync bug: the consensus eval's
`incr_graft = IncrementalEvaluator(_load_plc(name), benchmark)` initializes
soft positions from the FRESH PLC (i.e., initial benchmark soft positions),
and the existing `sync_positions(hard_pos)` only updates hard positions —
soft positions stayed at the initial values. So graft was optimizing
against `(portfolio_min hards, INITIAL softs)`, not the worker's actual
state. Each "improving" move was wrt the wrong state. On ibm01 N=8 240s,
this caused graft to "accept" 37 substitutions but cumulative cost went
from 0.79 → 1.02 (the trial costs lied because softs were wrong).

Fix: added `_sync_full_placement(incr, full)` that updates both hard AND
soft positions and recomputes from scratch. Used at every consensus entry
point (`per_macro_graft`, `_refine_and_return`).

Post-fix smoke results:

| Workers | Per-worker budget | Refine | Portfolio min | Consensus refined | Δ |
|---|---|---|---|---|---|
| N=2 | 30s | 15s | 0.9185 | (consensus skipped: N<3) | — |
| **N=8** | 60s | 60s | **0.9309** | **0.9267** | **-0.0042 ✓** |

N=8 result: graft accepted 30 per-macro substitutions (each strictly
improving wrt the correctly-synced state), driving cost from 0.9309 to
0.9308; the subsequent 60s GPU refine drove it to 0.9267. Portfolio min
was 0.9309. **Consensus WIN by 0.0042 on ibm01 N=8 60s budget.**

Expected lift at full budget (3300s, 8 workers): the per-seed variance
decreases and the cost-floor approaches the per-bench irreducible
minimum, so the consensus advantage will likely shrink to -0.001 to
-0.005 on the easy benchmarks (ibm01) but remain in the -0.005 to -0.015
range on the hard ones (ibm12/14/16/17/18) where worker variance is
high. Validation is the Week 1 snapshot run (task #15).

**OpenROAD Tier-2 robustness.** Consensus is *strictly preferable* to
"raw portfolio min" when both are available. Proxy-pathological
placements (one worker stuck a macro in a corner because the RNG drew
it there) score well on the proxy but tend to underperform on OpenROAD.
The trimmed-mean discards those outliers; the consensus is a "median
pose" that the synthesis tools tend to handle better. Even if consensus
only ties on proxy, it should win on Tier-2 ranking.

**Decision.** Shipped. Default on. Real-data win condition (consensus
beats portfolio min on ibm01 at N=8) pending in-flight smoke test.

---

## Summary so far

- T1.1 + T1.3: GPU CD beats CPU CD on ibm01 by 0.0040 at 60 s (-0.0040
  per-bench gain at the CD slot). **Shipped.**
- T1.2: Hungarian LNS loses on dense benchmarks due to candidate
  infeasibility. **Killed.**
- T3.4: trimmed-mean consensus warm-start. Synthetic test passes;
  real-data N≥8 validation pending. **Shipped (default on).**
- T4.1: superseded by torch backend (one codebase runs on grader CUDA +
  dev MPS).

Next: Tier 2 (T2.3 Adam warm-start, T2.1 per-net full proxy, T2.2 GAT
surrogate). See plan at
`/Users/vmallela/.claude/plans/ultraplan-didn-t-work-please-tingly-bird.md`.
