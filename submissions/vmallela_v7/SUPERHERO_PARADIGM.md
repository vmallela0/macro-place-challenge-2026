# Superhero — the Phase-1 paradigm shift

## TL;DR

`superhero_stretch.py` solves placement initialization in **closed form**
via one sparse LU on the netlist Laplacian augmented with a Bayesian prior.

### Full 17-bench result (competition-compliant — single global config)

`stretch=1.10, λ=2000` applied to every bench, no per-bench tuning. CD-polished:

```
                default→CD    super→CD     delta     %
  mean over 17:    1.4155      1.3702     -0.0452   -3.20%
  super wins:                                       15/17
```

Per-bench breakdown (largest wins on top):

| bench | default→CD | super→CD | Δ      |
|-------|-----------:|---------:|-------:|
| ibm06 |     1.6352 |   1.5362 |  -6.06%|
| ibm07 |     1.4575 |   1.3703 |  -5.98%|
| ibm03 |     1.3121 |   1.2371 |  -5.72%|
| ibm13 |     1.3547 |   1.2912 |  -4.68%|
| ibm15 |     1.5583 |   1.4859 |  -4.65%|
| ibm12 |     1.5939 |   1.5211 |  -4.57%|
| ibm08 |     1.4490 |   1.3833 |  -4.53%|
| ibm10 |     1.2216 |   1.1699 |  -4.24%|
| ibm16 |     1.4467 |   1.3871 |  -4.12%|
| ibm11 |     1.1904 |   1.1440 |  -3.89%|
| ibm09 |     1.0702 |   1.0299 |  -3.76%|
| ibm14 |     1.5266 |   1.4810 |  -2.99%|
| ibm17 |     1.7214 |   1.6759 |  -2.64%|
| ibm04 |     1.2392 |   1.2186 |  -1.66%|
| ibm01 |     1.0233 |   1.0181 |  -0.51%|
| ibm02 |     1.4833 |   1.5195 |  +2.44%|  ← loss
| ibm18 |     1.7795 |   1.8249 |  +2.55%|  ← loss

### Per-bench tuned upper bound (research only — NOT competition compliant)

If we allow per-bench (stretch, λ) tuning, the same algorithm hits about
−4.95% mean (all 17 wins at raw init). The TUNED_RESEARCH_ONLY dict in
the source documents the per-bench best (stretch, λ) we observed.

## The math

Movable set M = hard macros ∪ soft macros. Anchored set P = ports
(positions fixed by the benchmark on the canvas edge).

Build adjacency `W` on M ∪ P via **clique-edge expansion** of nets:
for a k-pin net with weight w (clipped), every pair of pins gets
`w / (k − 1)` of edge weight. Laplacian `L = diag(rowsum W) − W`.

Build the **stretched-default prior**:

    g_i = stretch · (default_i − canvas_center) + canvas_center

clipped to canvas. `stretch ∈ {1.02 … 1.15}` per bench. This is the
crucial novel piece — pulled outward enough to leave **headroom** for
the Laplacian to perturb without macros overlapping neighbors.

Solve the regularized normal equation (closed form, sparse LU):

    (L_MM + (α + λ) I) x_M = λ g_M − L_MP x_P                       (★)

| symbol | meaning |
|--------|---------|
| L_MM   | (movable, movable) block of L |
| L_MP   | (movable, anchored) block of L |
| x_P    | port positions (fixed) |
| g_M    | stretched-default prior for movables |
| α      | tiny Tikhonov (e.g. 1e-3), regularizes disconnected components |
| λ      | Bayesian prior weight (typical 100 … 5000) |

This is **THE LITERAL CLOSED-FORM** MAP estimator for placement under the
joint posterior:

    p(x | net structure, default prior) ∝
        exp(−½ x^T L x)  ·  exp(−½ λ ‖x − g‖²)

Each factor justified by a theorem:

| Factor                       | Reference                  |
|------------------------------|----------------------------|
| Anchored quadratic placement | Tutte 1963 / Hall 1970     |
| Tikhonov regularization      | Tikhonov 1943              |
| Stretched prior              | (this work — empirically)   |

## Why it works (and not the earlier variants)

We worked through 6 prior approaches that **all** lost to default after CD:

1. `gravity_drop`  (n-D simplex + harmonic gravity)            +0.10 mean
2. Spectral on hard-only graph                                  failed (disconnected)
3. Anchored quadratic on full graph, no prior                  +0.10 mean (clustering)
4. Picard repulsion (Coulomb hinge)                            +0.08 mean (steady-state stays clustered)
5. CDF-uniform OT prior                                         +0.06 mean (destroys CONG)
6. Pure default as prior                                       +0.04 mean (high λ ⇒ recovers default)
7. **Stretched default as prior**                              **−0.043 mean (3/4 wins)** ✅

The diagnosis for (1)–(6) is identical: WL-min flows toward clustering;
DEN/CONG penalties create overlaps. (7) breaks the symmetry — by stretching
the prior **outward**, the Laplacian's centripetal pull lands the macros at
a **slightly-spread version of default**, simultaneously decreasing WL
(net topology pulled toward equilibrium) and respecting DEN (macros
spaced like default ± small perturbation).

## Per-bench tuned hyperparameters

| bench | stretch | λ   |
|-------|---------|-----|
| ibm06 | 1.15    | 500 |
| ibm01 | 1.05    | 500 |
| ibm02 | 1.02    | 500 |
| ibm09 | 1.08    | 500 |

Smaller canvases need smaller stretch (less headroom to pull outward
before hitting boundary). For unseen benches, fallback is
`stretch=1.05, λ=500`.

## Reproduce

```bash
.venv/bin/python submissions/vmallela_v7/superhero_stretch.py \
  --benchmark ibm06 --output /tmp/super_final/ibm06.json

# Then polish with v1's _coord_descent (or v7's full pipeline)
.venv/bin/python submissions/vmallela_v7/grav_polish.py \
  --benchmark ibm06 --grav-init /tmp/super_final/ibm06.json \
  --output /tmp/super_final/polish/ibm06.json \
  --cd-time 240 --legalize-iters 0 --arms grav,default
```

## What's next

- Sweep `stretch` and `λ` on the remaining 13 IBM benches; build a complete
  tuning table.
- Test the WIN under v7's full polish pipeline (not just plain CD).
  CD-only on Mac is a weak polish; v7's electrostatic / Hessian-escape
  stages on competition HW should extract more value from the better init.
- Investigate why ibm02 alone doesn't win — probably needs `stretch < 1.0`
  (contract!) because its hand-tuned default is already more spread than
  the proxy prefers. Look into per-bench AUTO-TUNE via 1D bisection on
  stretch.
