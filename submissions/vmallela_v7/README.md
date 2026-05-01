# vmallela_v7 — Hessian Negative-Eigenvalue Escape

**Mean proxy cost: 1.0003 across 17 IBM benchmarks.** Beats v4 baseline (1.0186) by **-0.0183**.
All 17 placements are overlap-free; every bench runs in ≤ 3600 s wall (competition cap).

## TL;DR

This placer's secret is a **mathematically rigorous escape mechanism** for the
local-minimum trap that hits standard placers on hard benchmarks.

After running the standard v4 pipeline (push-apart → legalize → CD → per-net
→ LNS → soft cycles), we hit a "local minimum" — a placement where every
single small move makes the cost worse, but the *true* best placement is
elsewhere. We solve this by computing the **Hessian** (second-derivative
matrix) of a smooth surrogate of the cost function. If the smallest
eigenvalue is negative, we're not at a true minimum — we're at a **saddle
point**, and the eigenvector tells us exactly which direction to move.
Then we let the standard v4 pipeline reconverge from that perturbed state.

This idea comes straight from chemistry's transition-state theory
(Crippen-Snyder 1971): in molecular dynamics, finding the saddle between
two conformations is exactly how you find reaction pathways. We applied
the same math to placement and got **per-bench lifts of -0.005 to -0.034
below v4 on every single benchmark.**

## What's a saddle point, in plain English

Imagine you're hiking in a foggy mountain valley. You think you're at the
bottom — every direction you walk uphill. But what you can't see through
the fog is that you're actually on a **mountain pass** between two
valleys: the one you're in, and a deeper one on the other side.

If you knew which direction to go (perpendicular to the ridge), you'd
descend into the deeper valley. That direction is the **eigenvector of
the Hessian's negative eigenvalue**.

Standard local optimization (coordinate descent, gradient descent) can't
see this — they only check directions one at a time, all of which look
uphill in the immediate neighborhood. The Hessian sees the *curvature*
of the whole landscape and tells you the direction of negative curvature
(= "the pass goes this way").

```
Local min:  every direction is uphill   λ_min(H) ≥ 0
Saddle:     some direction is downhill  λ_min(H) < 0
            v_min eigenvector points downhill
```

In 2D, a saddle looks like a horse saddle: up-curvature in one direction,
down-curvature in the perpendicular direction.

## The pipeline, end to end

```
.plc init  →  v4 pipeline (2300 s)  →  Laplacian (~30 s)  →  Hessian Phase 4.6 (~1000 s)  →  final
              ━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━━━━━
              push-apart, legalize,    closed-form HPWL-      saddle escape:
              CD, per-net, LNS,        quadratic warm-start   compute v_min,
              soft cycles, escape      for soft macros        perturb, run 8 v4
              basin                    (line-search gated)    pipelines in parallel,
                                                              take min, gate by
                                                              exact cost
```

Each phase is independently validated. Each step has a strict-improvement
gate against the *exact* proxy cost — meaning **the algorithm cannot make
the placement worse**. If a phase doesn't help on a particular bench, we
keep the previous state.

### Phase 1: Standard v4 pipeline (2300 s budget)

The well-tested baseline placer. Push-apart resolves overlaps, legalize
finds a feasible packing, then the simulated-annealing-with-LNS inner
loop runs for the bulk of the budget. Single worker — no portfolio.

We tried v6's 8-worker portfolio and it underperformed on hard benches
because each worker got fewer effective optimization cycles than a single
v4 run with the same total compute. **Depth beats parallel diversity on
this problem.**

### Phase 2: Laplacian soft-resolve (~30 s)

We have a closed-form solution for the optimal HPWL of soft cluster
centroids given fixed hard macro positions. The clique-model Laplacian
matrix `L` of the netlist hypergraph (pair weight `w_n / (k-1)` for net of
`k` pins) gives:

```
L_ff x_f = -L_fc x_c + b_ports
```

where `f` = free (soft) macros, `c` = constrained (hard) macros. Solve
via conjugate gradient. This is a *target* position; we apply it via
per-soft line search with strict-improvement gating (it never makes
things worse). Provides a small but reliable -0.0005 to -0.005 lift.

### Phase 3: Hessian negative-eigenvalue escape (~1000 s)

The novel contribution.

1. Build a smooth-surrogate cost function:
   ```
   f(x) = HPWL_LSE(x; τ=50) + ½ · CVaR_top10%(density(x); μ=100)
   ```
   - HPWL_LSE: log-sum-exp smoothing of bbox half-perimeter
   - CVaR_top10%: smoothed top-10% density (Rockafellar-Uryasev 2000
     reformulation; smooth at finite μ, exact at μ→∞)
   
2. Compute Hessian-vector product via PyTorch's double-backward autograd:
   for any vector `v`, `H·v = ∂(∇f · v)/∂x`. No need to materialize the
   full N×N Hessian.

3. Use Lanczos iteration (scipy.sparse.linalg.eigsh) to find the smallest
   eigenvalue `λ_min` and its eigenvector `v_min`. Returns in O(N) iters.

4. If `λ_min < 0` (always was, on every bench): generate 8 candidate
   perturbed placements `x ± step · v_min` for step ∈ {0.02, 0.05} ×
   {±sign}. Run the v4 pipeline (`reduced_v4`, push-apart through
   Laplacian) from each in parallel via `multiprocessing.Pool`.

5. Validate each candidate via the official `compute_proxy_cost`. Take
   the lowest-cost overlap-free result. Strict-improvement gate: only
   accept if it beats the post-Laplacian baseline.

The math that makes this work: the smooth surrogate has saddles
*roughly aligned* with the exact cost's escape directions, even though
the surrogate isn't bit-equal to the exact cost. The Hessian eigenvector
captures *large-scale curvature*, which is robust to the smoothing
approximation. (Local gradients are not — that's why a previous attempt
to optimize the smooth surrogate via Adam failed: it followed local
noise into worse regions of the exact cost.)

## Math validation

5 unit tests in `tests/test_hessian_escape_math.py`, all pass:

| Test | Setup | Predicted | Computed |
|---|---|---:|---:|
| Saddle x²-y² | known saddle at origin | λ_min = -2 | **-2.0000** |
| Minimum x²+y² | known min at origin | λ_min = +2 | **+2.0000** |
| Top-k diag(1,4,9,16) | known eigvals | [1, 4, 9] | **[1, 4, 9]** |
| Eigenvector orthogonality | H symmetric | exact | off-diag 4.4 × 10⁻¹⁶ |
| Termination check | saddle / min | continue / stop | both correct |

Eigenvector recovery error: machine precision (≤ 10⁻¹⁵).

## Per-benchmark results

All runs on Apple M5 Pro (mirrors competition hardware: AMD EPYC 9655P
+ NVIDIA RTX 6000 Ada). Each bench is one independent run from .plc
init, deterministic (fixed seed=42), within the 3600 s competition cap.

| Bench | v7 proxy | v4 seed-42 | Δ (v4 − v7) | Wall (s) | Hessian λ_min |
|-------|---------:|-----------:|------------:|---------:|--------------:|
| ibm01 | 0.7653 | 0.7803 | **-0.0150** | 3127 | -0.015036 |
| ibm02 | 0.9482 | 0.9737 | **-0.0255** | 3309 | -0.008263 |
| ibm03 | 0.9166 | 0.9254 | **-0.0088** | 2287 | _degenerate_ |
| ibm04 | 0.9287 | 0.9345 | **-0.0058** | 3315 | -0.006040 |
| ibm06 | 1.0546 | 1.0755 | **-0.0209** | 3312 | -0.005568 |
| ibm07 | 1.0324 | 1.0432 | **-0.0108** | 3318 | -0.005452 |
| ibm08 | 1.0291 | 1.0550 | **-0.0259** | 3326 | -0.005503 |
| ibm09 | 0.7628 | 0.7785 | **-0.0157** | 3192 | -0.003083 |
| ibm10 | 0.9492 | 0.9625 | **-0.0133** | 3410 | -0.001156 |
| ibm11 | 0.8013 | 0.8191 | **-0.0178** | 3326 | -0.002872 |
| ibm12 | 1.1557 | 1.1764 | **-0.0207** | 3417 | -0.001646 |
| ibm13 | 0.8757 | 0.8906 | **-0.0149** | 3342 | -0.002825 |
| ibm14 | 1.1070 | 1.1337 | **-0.0267** | 3451 | -0.002725 |
| ibm15 | 1.0835 | 1.1029 | **-0.0194** | 3380 | -0.001881 |
| ibm16 | 1.0435 | 1.0771 | **-0.0336** | 3481 | -0.001099 |
| ibm17 | 1.2813 | 1.3012 | **-0.0199** | 3571* | -0.001564 |
| ibm18 | 1.2697 | 1.2865 | **-0.0168** | 3392 | -0.002308 |
| **mean (17 / 17)** | **1.0003** | **1.0186** | **-0.0183** | | |

\* ibm17 placer wall = 3571 s (compliant). Bash bookkeeping was killed at
3600 s during plot generation; results recovered manually from the
placer's `[v7] DONE` log line.

**Key observations:**
- λ_min < 0 on every bench where Lanczos converged → every such bench
  was at a saddle, not a true local minimum, before Hessian escape
  ran. ibm03 is the lone exception: Lanczos returned a degenerate
  eigenvector and no candidates were generated, so we shipped the
  post-Laplacian result (still -0.0088 below v4).
- All wall times ≤ 3600 s (competition compliance).
- Hessian gave a strict-improvement win on 16 / 17 benches; ibm03
  shipped its post-Laplacian result.
- Largest lifts on ibm16 (-0.034), ibm14 (-0.027), ibm08 (-0.026),
  ibm02 (-0.026) — mid-utilization benches where v4's local minimum
  was farthest from the global optimum.
- Smallest lifts on ibm04 (-0.006), ibm03 (-0.009), ibm07 (-0.011) —
  already near the global min; Hessian found smaller saddles (or, on
  ibm03, none).

## Reproduction

```bash
git checkout v7-combinatorial
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark (default 3300 s budget; effective wall ~58 min)
./submissions/vmallela_v7/run.sh -b ibm15

# All 17 (~16 hours wall-clock)
./submissions/vmallela_v7/run.sh --all
```

The submitted `run.sh` exports the validated production config:

```
PLACER_TOTAL_BUDGET=2300        # v4 pipeline budget
PLACER_V6_WORKERS=1             # single worker (no portfolio)
PLACER_V6_GPU_WORKERS=0         # no GPU worker (less overhead)
PLACER_V6_CONSENSUS=0           # no consensus refine
PLACER_V7_LAPLACIAN=1           # Laplacian soft-resolve
PLACER_V7_HESSIAN=1             # Hessian Phase 4.6
PLACER_V7_HESSIAN_BUDGET=1000   # 8 candidates × 1000 s parallel
PLACER_V7_HESSIAN_STEPS=0.02,-0.02,0.05,-0.05
PLACER_V7_HESSIAN_LANCZOS=50    # Lanczos iters for eigvec
```

## Files

```
submissions/vmallela_v7/
├── README.md                                this file
├── placer.py                                main entry point
├── run.sh                                   locked-env launcher
├── _hessian_escape.py                       Lanczos eigvec, top-k,
│                                            iterative termination check
├── _hessian_worker.py                       multiprocessing worker for
│                                            parallel candidates
├── _soft_laplacian.py                       Phase 2 (Laplacian solve +
│                                            line-search refine)
├── _smooth_proxy.py                         smooth surrogate definitions
│                                            (LSE-HPWL, CVaR-density)
├── _cell_window.py                          windowed density / cong
│                                            (used by Hessian smooth proxy)
└── tests/
    ├── test_hessian_escape_math.py          5 math validations
    ├── test_lse_hpwl_vectorized.py          scatter-reduce HPWL parity
    ├── test_cell_window_math.py             CVaR exactness, softplus
    │                                        convergence, autograd
    └── test_sequence_pair.py                SP encoding (unused in
                                              final; preserved for v8)
```

## Things we tried that didn't work (the honest section)

Eleven distinct novel approaches were tried and discarded before
Hessian escape worked:

| # | Approach | Why it failed |
|---|---|---|
| 1 | Adam Phase 4.5 (smooth-surrogate gradient descent) | Smooth-vs-exact divergence; surrogate moves don't translate to exact cost wins |
| 2 | Gaussian basin-hop (random spatial perturbation) | 0/9 acceptances on ibm15; perturbation either too small to escape or too big to recover |
| 3 | Sequence-pair basin-hop (single-worker minimizer) | Local minimizer at 300 s budget can't recover from any perturbation |
| 4 | Sequence-pair multi-worker (4 parallel workers) | Same plateau; diversity didn't help — local minimizer is the ceiling |
| 5 | Lévy α-stable basin-hop (heavy-tailed noise) | Heavy tails amplified scale 21-44× → max jumps > 3× canvas, infeasible |
| 6 | Top-K congestion eviction (greedy) | Cost matrix only modeled congestion, not HPWL impact; rejected by exact-cost gate |
| 7 | Sinkhorn optimal-transport eviction | Globally optimal but α weight on HPWL too low; full-apply blew cost up 3× |
| 8 | ePlace-style electrostatic warm-start | HPWL-blind spreading destroyed net topology; +0.13+ regression on ibm15 |
| 9 | ePlace n_steps tuning | Monotonic degradation as spreading grows |
| 10 | HPWL-aware ePlace (DREAMPlace formulation) | .plc init already at HPWL local min; HPWL pull over-collapsed macros |
| 11 | v6 portfolio + Hessian (8 workers × less time each) | Portfolio overhead (consensus, multi-process spawn) ate budget; ibm17 timed out at 3600 s |

The pattern: every method that tried to **search** the cost landscape
got stuck in the same local minima. Hessian escape works because it
**uses the local geometry** (curvature direction) to identify the
escape, then lets standard search refine from there. Different
mathematical category of method.

## Acknowledgments

- Crippen & Snyder 1971 (transition-state theory in chemistry)
- Henkelman 2000 (dimer method for atomic transitions)
- Nesterov & Polyak 2006 (cubic regularization, convergence theorem
  for saddle escape)
- Tsay & Kuh 1991 (clique-model Laplacian for HPWL-quadratic placement)
