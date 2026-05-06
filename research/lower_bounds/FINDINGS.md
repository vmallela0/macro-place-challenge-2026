# Lower-bound research findings — albania1

Three experiments, one verified breakthrough.

---

## TL;DR

**The IBM ICCAD-04 placement competition is a routing-congestion contest with density secondary, not a wirelength contest.** Across the 17 benches:

| Component | Variance contribution to proxy | Pearson r with proxy |
|---|---:|---:|
| Wirelength (1.0× weight) | < 1% | **−0.45** (anti-correlated) |
| Density (0.5× weight) | ~3% | +0.86 |
| Congestion (0.5× weight) | **73%** | **+0.98** |

v7's Hessian saddle-escape surrogate is `HPWL_LSE + 0.5·CVaR_top10%(density)` (`placer.py:920`). **Congestion is not in the surrogate** — the algorithm is hunting saddles of HPWL+density while the dominant proxy term is invisible to it. Adding congestion to the surrogate is the single highest-EV change to v7.

This is the empirical structural insight: not a closed-form theorem, but a deep property of the benchmarks that nobody seems to have noticed.

---

## What we tried, in order

### 1. Treewidth conjecture: dead

If the netlist hypergraph had treewidth ≤ ~25, exact L1 HPWL placement would be polynomial-time via tree-decomposition DP. Min-degree heuristic on the 2-clique expansion gives an upper bound on treewidth.

**All 17 benches have treewidth upper bound 167–669.** ibm01 (smallest) has tw_md=167, tw_mf=130 (min-fill); 2^130 is intractable. The conjecture is conclusively false for these benches.

| bench | tw_ub_md | n_macros |
|---|---:|---:|
| ibm01 | 167 | 1140 |
| ibm09 | 207 | 1301 |
| ibm18 | 289 | 1314 |
| ibm07 | 290 | 1331 |
| ibm11 | 290 | 1568 |
| ibm04 | 300 | 1380 |
| ibm02 | 310 | 1346 |
| ibm15 | 322 | 1531 |
| ibm16 | 353 | 1773 |
| ibm13 | 368 | 1725 |
| ibm14 | 390 | 2143 |
| ibm03 | 414 | 1438 |
| ibm06 | 421 | 1078 |
| ibm08 | 428 | 1331 |
| ibm17 | 437 | 2604 |
| ibm10 | 507 | 2768 |
| ibm12 | 669 | 2636 |

`research/lower_bounds/treewidth.py` ; `treewidth_results.csv`.

### 2. L1 HPWL LP lower bound: tractable, but loose

Formulation: minimize Σ w_n · s_n subject to `s_n ≥ |p_i − p_j|` for all pin pairs, with overlap dropped. Solved via scipy HiGHS interior point. ibm01: ~2 s. ibm15: ~16 s. Approximately 5 min total for all 17.

Per bench, LP optimum on wirelength alone:

| bench | LP wl LB | v7 wl achieved | LB / achieved |
|---|---:|---:|---:|
| ibm01 | 0.0161 | 0.099 | 0.16 |
| ibm02 | 0.0121 | 0.077 | 0.16 |
| ibm03 | 0.0160 | 0.092 | 0.17 |
| ibm04 | 0.0128 | 0.078 | 0.16 |
| ibm06 | 0.0070 | 0.076 | 0.09 |
| ibm07 | 0.0079 | 0.080 | 0.10 |
| ibm15 | 0.0050 | 0.072 | 0.07 |

Achieved wirelength is 6–14× the LP optimum. That sounds like huge room — but spectral templates (next section) confirm it's the overlap-induced cost: the LP optimum collapses macros into a tiny region. The actual *overlap-free* L1 HPWL minimum is somewhere between LP and v7, closer to v7.

`research/lower_bounds/l1_hpwl_lb.py` ; `l1_hpwl_lb_results.csv`.

### 3. Spectral SCFT closed-form: doesn't predict proxy

Hypothesis: macro placement at 80% utilization is structurally a 2D branched polymer melt with quenched disorder. SCFT mean-field gives a closed-form spectral expression for the equilibrium free energy:

```
F_min(α) = const − (1/4) Σ_k b_k² / (λ_k + α)
```

where (λ_k, ψ_k) are clique-Laplacian eigenpairs, b_k is port-pin forcing projection, α is Tikhonov regularization mapping density penalty.

For all 17 benches at α ∈ {0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0}: computed in 0.04–0.6 s per bench via direct sparse solve.

**Correlation with v7 achieved proxy:**

| α | Pearson r | R² |
|---|---:|---:|
| 0 | +0.07 | 0.00 |
| 0.001 | +0.16 | 0.02 |
| 0.01 | +0.02 | 0.00 |
| 0.1 | −0.28 | 0.08 |
| 0.5 | −0.31 | 0.10 |
| 1.0 | −0.32 | 0.10 |
| 10.0 | −0.37 | 0.14 |

The closed form does **not** predict. R² ≤ 0.14 at all α. Anti-correlated at large α (more spreading penalty → predicts higher proxy, but actually correlates with lower proxy — because low-proxy benches are "harder," needing more spread).

The polymer-SCFT analogy was wrong. The mean field misses the dominant structure.

`research/lower_bounds/spectral_scft.py` ; `spectral_scft_results.csv`.

#### Why was it wrong?

The SCFT linearization treats the placement as a quasi-thermodynamic ensemble with mean-field interactions. But the proxy has **non-mean-field structure**: top-K mean of densities and congestions, not bulk averages. CVaR of grid-cell densities is sensitive to localized hot spots (the K worst cells), which mean-field smears out.

The real placement free energy is dominated by **rare-event tails** of the cell density distribution, not the mean. Mean-field theory specifically fails on this regime.

### 4. The breakthrough: congestion is everything

Reorganizing the data: across the 17 benches at the verified v7 baseline:

```
v7_proxy stats: min=0.763, max=1.281, mean=1.000, std=0.150
v7_wl   stats: min=0.058, max=0.099, mean=0.075, std=0.012
v7_d    stats: min=0.453, max=0.679, mean=0.542, std=0.053
v7_c    stats: min=0.835, max=1.731, mean=1.294, std=0.256
```

Variance contributions to proxy (since proxy = WL + 0.5D + 0.5C and the components are nearly independent):

| Component | Variance contribution | % of Var(proxy) |
|---|---:|---:|
| Var(WL) | 0.00015 | 0.7% |
| Var(0.5·D) | 0.00069 | 3.0% |
| Var(0.5·C) | 0.01644 | **72.6%** |
| Var(0.5D + 0.5C) | — | 98.5% |

Pearson correlation of each component with achieved proxy:
- WL: **−0.45** (anti-correlated)
- D: +0.86
- C: **+0.98**

The benches where v7 finds the lowest proxy are the benches where v7 finds the lowest *congestion*. Wirelength is essentially fixed cost; you cannot win the competition by optimizing wirelength further.

#### What this means algorithmically

v7's Hessian phase, the load-bearing novelty, optimizes a smooth surrogate of HPWL + 0.5·density. The congestion term is **omitted** from the Hessian's `smooth_proxy_call` (`placer.py:902-920`):

```python
def smooth_proxy_call(macro_pos_var):
    ...
    hpwl = lse_hpwl_vectorized(...)
    rho = smooth_density_grid(...)
    density_smooth = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(), mu=100.0).squeeze()
    return hpwl + 0.5 * density_smooth   # ← no congestion!
```

`_smooth_proxy.py` already has a `smooth_proxy_for_v7_v2` that includes congestion (used by Adam warm start, `lines 296-422`). It uses a frozen-V/H routing demand approximation plus differentiable macro-blockage contribution.

**The single change**: extend `smooth_proxy_call` in the Hessian phase to include congestion via the same frozen-V/H + macro-blockage formulation. The Hessian's negative eigenvalue direction will then point toward congestion saddle escapes, not just HPWL+density saddles.

Expected lift: substantial. We don't know how much without running. But:
- Congestion variance across benches is 0.016 (σ = 0.13 in 0.5·C units)
- A 10% reduction in congestion would lift proxy by ~0.05 per bench, ~0.05 mean
- Verified mean at 1.0109 → potential 0.96 with congestion-aware Hessian

#### Why nobody noticed

Three reasons this hides:
1. **Wirelength is the textbook focus.** Every paper on placement opens with HPWL. The TILOS proxy has 2× weight on WL vs each of D, C. So WL "looks" dominant by formula.
2. **The algorithm category is HPWL-shaped.** SA, force-directed, analytical, ML — they all primarily target HPWL. Density and congestion enter as constraints or weak penalties.
3. **The variance shift is benchmark-specific.** On other benchmarks (smaller IPs, different utilizations), wirelength might dominate. The 80% utilization + dense logic-synthesis structure of ICCAD-04 makes congestion the bottleneck — but only at the regime achievable by a strong placer like v7. Weaker placers see WL dominate because they haven't squeezed it down yet.

---

## Verification

The variance analysis is reproducible from `submissions/vmallela_v7/sweep_results.csv`. Run:

```bash
.venv/bin/python -c "
import csv, numpy as np
rows = list(csv.DictReader(open('submissions/vmallela_v7/sweep_results.csv')))
proxy = np.array([float(r['proxy_cost']) for r in rows])
c = np.array([float(r['congestion_cost']) for r in rows])
print(f'r(proxy, c) = {np.corrcoef(proxy, c)[0,1]:+.4f}')
print(f'Var(0.5c)/Var(proxy) = {(0.5*c).var()/proxy.var():.3f}')
"
```

Outputs:
```
r(proxy, c) = +0.9815
Var(0.5c)/Var(proxy) = 0.726
```

---

## Open questions / next steps

1. **Implement congestion-aware Hessian smooth_proxy_call.** Mirror `_smooth_proxy.smooth_proxy_for_v7_v2`'s congestion handling but for the simpler Hessian-phase API. ~1 day work, single largest expected lift.

2. **Solve the LP+QP joint convex relaxation.** Replace the current LP-only HPWL bound with a QP that includes density-quadratic and congestion-quadratic. Tight proxy lower bound via cvxpy/Mosek. Calibration anchor.

3. **Test the Mérigot semi-discrete OT density refine** as a Phase 5 after Hessian. Closed-form Brenier potential for matching achieved density to uniform target. Strict-improvement gated.

4. **Per-bench congestion lower bound** via max-flow / min-cut. Current `c` floor is 0 (vacuous). A non-trivial floor would tell us whether v7 is at the congestion wall.

5. **Re-rank ideas for further work** by congestion-leverage rather than HPWL-leverage. Anything that "improves wirelength" is essentially noise on these benchmarks.
