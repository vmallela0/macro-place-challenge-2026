# Autonomous iteration log

Started: 2026-05-05 23:10 PDT (user asleep; goal: push proxy < 1.0109).
End conditions: verified mean ≤ 0.95 OR 3 consecutive < 0.005 improvements OR ~12h elapsed.

Style: bold math reformulations (Ramanujan), rigorous verification (Tao).

---

## Iter 0 — initial state

Branch: `albania1` (off `v7-combinatorial-submission`, the verified-1.0109 branch)
- Cherry-picked istanbul (adaptive line-search Hessian)
- Klein-4 orientation flip + sidecar (Tier 2)
- Halo knob (default off)
- CVaR k_dens knob (default 0.10)
- **Congestion-aware Hessian smooth_proxy_call** (the breakthrough — default ON)
- benchmark.py forward-compat for unknown fields

Running (background): albania1 cong A/B sweep on ibm17, ibm08 with cong-on for both arms.

Verified baseline (cong-off, from `submissions/vmallela_v7/sweep_results.csv`):
- ibm17: 1.2813
- ibm08: 1.0291
- 17-bench mean: 1.0003 (dev box) / 1.0109 (LSJ verified)

Hypotheses to test:
- H1: Adding congestion to Hessian surrogate lowers proxy on ibm17/ibm08 vs verified.
- H2: Multi-eigvec Hessian (k=3) widens escape subspace, finds better minimum.
- H3: Iterative Hessian (3 passes) crosses multiple saddles per bench.
- H4: Cong weight tuning (1.0x or 2.0x instead of proxy's 0.5x) emphasizes the dominant variance term.
- H5: Mérigot semi-discrete OT post-Hessian gives free density refine.

Expected next action: wait for A/B (~3.7h), then decide on H1.
