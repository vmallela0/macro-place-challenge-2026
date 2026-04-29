# vmallela_v7 — Combinatorial Hail Mary

A calculated stack of three independently-validated lifts on top of the
v6 pipeline, targeting the hard sparse benches (ibm12, 14, 15, 16, 18)
where v6 regressed vs v4. The combinatorial-structure analysis behind
each lift is in the v6 README's "Honest analysis" section.

## What v7 stacks on top of v6

```
       v6: portfolio (8 workers) + GPU CD + consensus    →  1.0184 mean
                  ↓
            v6 result feeds into v7's three new layers:
                  ↓
    1. Laplacian soft-resolve  (closed-form HPWL warm-start)
       Per-soft line search with full-proxy acceptance, never
       makes things worse. Smoke test: 1.0559 → 0.9838 on ibm01
       legalize+refine state (Δ -0.072 in 1.4s).
                  ↓
    2. Basin-hopping outer loop  (auto-trigger when cost ≥ 1.0)
       Up to 3 hops × 300s each, single-worker reduced pipeline
       per hop. Targets the soft-state plateau pattern observed
       in the v6 diagnostic GIFs.
                  ↓
    3. Adam smooth surrogate    (scaffolded; Phase 0 init)
       LSE-HPWL + CVaR top-K density / congestion. CVaR
       reformulation (Rockafellar-Uryasev) makes top-K averages
       globally smooth. Currently HPWL-only; full density / cong
       gradient with cell-window truncation deferred. Disabled
       by default in v7.
```

## The math, validated

### (1) Laplacian soft-resolve

For SOFT macros given fixed hards, the **HPWL-quadratic surrogate**
$\frac{1}{2} x^T L x$ has a closed-form global minimum via the linear
system $L_{ff} x_f = -L_{fc} x_c + b_{ports}$, where $L$ is the netlist
hypergraph clique-model Laplacian (pair weight $w_n / (k-1)$ for net of
$k$ pins), partitioned into free (soft) and constrained (hard) blocks.

**Validated:** $\| L - L^T \|_F = 1.5 \times 10^{-14}$ (numerical zero,
matrix is exactly symmetric). Smallest 5 eigenvalues:
$[2.8\times 10^{-15}, 0.235, 0.359, 0.364, 0.374]$ — PSD with one
near-zero (the trivial all-ones eigenvector for unconstrained-translation).
CG converges in <100 iters with `grad_norm < 1.1e-3`.

**Why we use it as a target, not a direct apply:** soft macros in this
challenge's clustered formulation have *small but non-zero* footprints
(checked: `macro_h.min() = 0.0063`, not zero). Bulk-applying the
HPWL-quadratic minimum clusters all softs into hot density cells; on
ibm01, HPWL drops 40 % but density blows up 4× and net cost regresses
to 1.7. Per-soft line search at α ∈ {1.0, 0.5, 0.25, 0.1, 0.05}
with full-proxy acceptance keeps the HPWL improvement where it
doesn't conflict with density / congestion.

### (2) Basin-hopping (Wales & Doye 1997)

Standard global-optimization heuristic for energy landscapes with
many local minima separated by barriers. Algorithm:

1. Run local minimizer L from $x_0$ → $x^*_0$.
2. For $k = 1, 2, \ldots$:
   - Perturb: $y_k = x^*_{k-1} + \sigma_k \xi$ where $\xi \sim \mathcal{N}(0, I)$
   - Locally minimize: $x^*_k = L(y_k)$
   - Strict-accept (or Metropolis at T > 0)
   - Cool $\sigma_k$ geometrically.

**Convergence:** Wales 1999 Theorem 1 — for any landscape with
finite-many minima, basin-hopping converges to the global minimum
with probability 1 as $k \to \infty$.

For our placer, $L$ = a reduced-cost single-worker pipeline
(`_reduced_single_pipeline` in `placer.py`): push-apart → legalize →
refine → Laplacian soft-resolve → CD → per-net → hard LNS. Each hop
costs ~300 s wall-clock at default budget, so 3 hops add ~15 min on
top of the main 30-min portfolio.

$\sigma$ schedule: $\sigma_0 = 0.30 \cdot $ canvas_diag, cool factor
$0.6$ per hop. Hard macros perturbed at $0.25 \sigma$ to keep
legalize feasible.

**Auto-trigger:** basin-hopping fires when (a) main portfolio result
is $\geq 1.00$, (b) at least 1 hop's worth of time remains, (c) result
is overlap-free. So easy benches (ibm01–ibm11, all $< 1.0$ in v6)
skip basin-hopping entirely; only the hard benches pay the extra
wall-clock. Forced via `PLACER_V7_BASIN_HOPS=N` to override.

### (3) Adam + CVaR (scaffolded, novel)

Smooth surrogate of the proxy:

$$\text{proxy}_{\text{smooth}}(x) = \text{HPWL}_{\text{LSE}}(x) + \tfrac{1}{2} \text{CVaR}^{(K_d)}_\mu(\rho(x)) + \tfrac{1}{2} \text{CVaR}^{(K_c)}_\mu(\gamma(x))$$

where:

**LSE smoothing for HPWL** (standard placement folklore):

$$\text{HPWL}_{\text{LSE}}^{(\tau)} = \frac{1}{\tau} \sum_n w_n \Big( \log\sum_i e^{\tau x_i^n} + \log\sum_i e^{-\tau x_i^n} \Big) + \text{(y-dim)}$$

Convex in pin positions; converges to true HPWL as $\tau \to \infty$.

**CVaR reformulation** of top-K average (Rockafellar-Uryasev 2000):

$$\text{CVaR}^{(K)}_\alpha(\rho) = \inf_t \Big\{ t + \frac{1}{(1-\alpha) n} \sum_{c=1}^n (\rho_c - t)_+ \Big\}$$

For $\alpha = 1 - K/n$ on a finite sample: at the optimum
$t^* = \rho_{(n-K)}$ (the $(n-K)$-th order statistic), and the
formula equals **exactly** the top-K mean (proved by direct
substitution; see derivation in `_smooth_proxy.py`).

Smoothed via softplus-$\mu$ instead of hard ReLU:
$(x)_+ \approx \frac{1}{\mu} \log(1 + e^{\mu x})$, $\mu = 100$.

**The novel claim:** to my knowledge, no published placer uses CVaR
for the top-K cell-density / cell-congestion terms. Density-bell
(DREAMPlace, RePlAce) and quadratic density (FastPlace) are the
standard smoothings — both are *approximations* that don't preserve
the top-K optimum exactly. CVaR preserves the exact top-K optimum
while being smooth.

**Status in v7:** scaffolded but density / congestion gradient is
**zeroed** in the surrogate for the initial release. The full
implementation requires cell-window truncation (re-snapshot the
window every K Adam steps) since the cell-index map is non-smooth.
That's another 1–2 days. The Laplacian piece does most of what Adam
on HPWL-only would do (closed-form vs gradient descent toward the
same local minimum), so the marginal value of HPWL-only Adam is
small. The CVaR density / congestion piece is where the novel lift
would come from — deferred to v8.

## Pipeline, end-to-end

```
Phase 0: load benchmark; load .plc init
Phase 1: v6 portfolio (8 workers parallel)
         each worker = full v4 pipeline at PLACER_TOTAL_BUDGET seconds
         + GPU CD on 1 of the 8 workers
         + trimmed-mean consensus warm-start at end
   → "portfolio cost"

Phase 2: Laplacian soft-resolve on the portfolio result
         line-search per soft macro toward the L-solve target
         strict-improvement gating
   → "post-Laplacian cost" (≤ portfolio cost)

Phase 3: AUTO basin-hopping if post-Laplacian cost ≥ 1.0
         each hop: perturb → reduced single-worker pipeline → strict accept
         up to 3 hops at 300s each; σ cools 0.30·D → 0.04·D
   → "post-basin cost" (≤ post-Laplacian cost)

Return: best of {portfolio, Laplacian, basin-hop} (always overlap-free).
```

## Reproduction

```bash
git checkout v7-combinatorial
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark (full submitted budget)
./submissions/vmallela_v7/run.sh -b ibm15

# All 17 (≈ 11–13 hours wall-clock at default settings)
./submissions/vmallela_v7/run.sh --all

# Tunables (env vars)
PLACER_TOTAL_BUDGET=3300            # main portfolio per-worker budget
PLACER_V7_LAPLACIAN=1               # 0/1: Laplacian soft-resolve on/off
PLACER_V7_LAPLACIAN_PASSES=2        # IRLS-style outer iterations
PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
PLACER_V7_BASIN_HOPS=0              # 0=auto (only on hard benches), N=force
PLACER_V7_BASIN_HOP_BUDGET=300      # seconds per basin hop
PLACER_V7_BASIN_HOP_AUTO=1.00       # auto-fire threshold (post-Laplacian cost)
PLACER_V7_BASIN_SIGMA0=0.30         # initial perturb σ as fraction of canvas_diag
```

## Honest expectations

If basin-hopping recovers half of the hard-bench regressions and the
Laplacian piece adds another -0.005 to all benches:

| Component | Expected mean lift |
|---|---:|
| v6 baseline | 1.0184 |
| + Laplacian soft-resolve (every bench) | -0.005 to -0.010 |
| + Basin-hopping (hard benches only) | -0.005 to -0.012 |
| **Projected v7 mean** | **0.998 – 1.005** |

That's the **Hail Mary range**. Real result depends on whether the hard
benches' plateau pattern is actually escapable from a Gaussian-perturbed
restart, which the diagnostic GIFs strongly suggest but don't prove.

If sub-1.0 doesn't land here, the next lever is **finishing the CVaR
density / congestion gradient in `_smooth_proxy.py`** — that's the
genuinely novel optimization piece. Estimated 1–2 more days of work.

## Files

```
submissions/vmallela_v7/
├── README.md                  this file
├── placer.py                  OptimalPlacer entry; orchestrates v6 →
│                              Laplacian → basin-hop
├── run.sh                     locked-env launcher
├── _soft_laplacian.py         clique-model Laplacian + line-search refine
├── _basin_hop.py              outer-loop wrapper (algorithm + math docs)
└── _smooth_proxy.py           Adam + CVaR scaffolding (HPWL-only active)
```

## Validation

The Laplacian construction passes:
- $\| L - L^T \|_F < 10^{-13}$ (symmetric)
- 5 smallest eigenvalues $\geq 0$ (PSD)
- CG residual $< 10^{-3}$ at default tolerance
- Improves cost by $\geq 0.05$ on ibm01 from legalize+refine state

Basin-hopping math is canonical (Wales 1999); no new claim.

CVaR equivalence (top-K = CVaR at optimum) is a textbook result
(Rockafellar-Uryasev 2000); we apply it to placement.
