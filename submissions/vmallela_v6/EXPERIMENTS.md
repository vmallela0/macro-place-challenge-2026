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

## Summary so far

- T1.1 + T1.3: GPU CD beats CPU CD on ibm01 by 0.0040 at 60 s (-0.0040
  per-bench gain at the CD slot). **Shipped.**
- T1.2: Hungarian LNS loses on dense benchmarks due to candidate
  infeasibility. **Killed.**
- T4.1: superseded by torch backend (one codebase runs on grader CUDA +
  dev MPS).

Next: Tier 2 (T2.1 per-net full proxy, T2.3 Adam warm-start, T2.2 GAT
surrogate). See plan at
`/Users/vmallela/.claude/plans/ultraplan-didn-t-work-please-tingly-bird.md`.
