# Macro-Placement Challenge 2026 — submission `vmallela_v7`

**Hessian negative-eigenvalue saddle escape for VLSI macro placement.**

| Mean proxy cost (17/17 IBM benches) | Best bench | Worst bench | Overlaps | Total wall |
|---:|---:|---:|---:|---:|
| **1.0109** | 0.7644 | 1.2921 | 0 | 15.5 h |

Verified by the official grader on the competition hardware
(AMD EPYC 9655P + NVIDIA RTX 6000 Ada). All 17 benchmarks valid,
zero macro overlaps, every bench under the 1-hour-per-bench cap.

---

## The idea in one paragraph

Standard placers (simulated annealing + large-neighbourhood search)
terminate when every small move makes the cost worse. That looks like
a local minimum, but on hard benchmarks it is usually a **saddle point**
of the cost surface: flat or rising in every nearby *spatial*
direction, but a non-local *curvature* direction still goes downhill.
The Hessian of a smooth surrogate of the cost sees this directly — its
smallest eigenvalue is negative at a saddle, and the corresponding
eigenvector is the escape direction. We compute that eigenvector with
Lanczos iteration on Hessian-vector products (PyTorch double-backward),
perturb the placement along it, and let the standard pipeline
reconverge. Strict-improvement gating against the *exact* proxy cost
ensures the algorithm cannot make a placement worse.

The math is borrowed from transition-state theory in computational
chemistry (Crippen & Snyder 1971; Henkelman & Jónsson 2000): finding
the saddle between two molecular conformations is exactly how you
find reaction pathways. Applied to placement, it identifies the pass
out of an apparent local minimum.

---

## Pipeline

```
                 .plc init
                     │
   ┌─────────────────┴─────────────────┐
   │  Phase 1 — v4 baseline   (2300 s) │
   │    push-apart → legalize →        │
   │    CD → per-net → LNS →           │
   │    soft cycles → escape basin     │
   └─────────────────┬─────────────────┘
                     │
   ┌─────────────────┴─────────────────┐
   │  Phase 2 — Laplacian solve (~30 s)│
   │    closed-form HPWL-quadratic     │
   │    optimum for soft centroids,    │
   │    per-soft line-search gated     │
   └─────────────────┬─────────────────┘
                     │
   ┌─────────────────┴─────────────────┐
   │  Phase 3 — Hessian escape (~1000s)│
   │    smooth surrogate f(x) =        │
   │      HPWL_LSE + ½·CVaR(density)   │
   │    Lanczos → λ_min, v_min         │
   │    8 candidates: x ± step·v_min   │
   │    parallel re-optimization       │
   │    strict-improvement gate        │
   └─────────────────┬─────────────────┘
                     │
                    out
       overlap-free, validated placement
```

Every phase has a strict-improvement gate against the **exact** proxy
cost. If a phase fails to help on a particular bench, its output is
discarded and the previous state is kept.

---

## Per-benchmark placements

Hard macros = red rectangles, soft cluster centroids = blue dots.

| | | | |
|---|---|---|---|
| ![ibm01](assets/v7_ibm01.png) | ![ibm02](assets/v7_ibm02.png) | ![ibm03](assets/v7_ibm03.png) | ![ibm04](assets/v7_ibm04.png) |
| **ibm01** | **ibm02** | **ibm03** | **ibm04** |
| ![ibm06](assets/v7_ibm06.png) | ![ibm07](assets/v7_ibm07.png) | ![ibm08](assets/v7_ibm08.png) | ![ibm09](assets/v7_ibm09.png) |
| **ibm06** | **ibm07** | **ibm08** | **ibm09** |
| ![ibm10](assets/v7_ibm10.png) | ![ibm11](assets/v7_ibm11.png) | ![ibm12](assets/v7_ibm12.png) | ![ibm13](assets/v7_ibm13.png) |
| **ibm10** | **ibm11** | **ibm12** | **ibm13** |
| ![ibm14](assets/v7_ibm14.png) | ![ibm15](assets/v7_ibm15.png) | ![ibm16](assets/v7_ibm16.png) | ![ibm17](assets/v7_ibm17.png) |
| **ibm14** | **ibm15** | **ibm16** | **ibm17** |
| ![ibm18](assets/v7_ibm18.png) | | | |
| **ibm18** | | | |

---

## Results

### Grader-verified (official)

Verified by the competition grader on AMD EPYC 9655P + NVIDIA RTX 6000 Ada.

| Metric | Value |
|---|---:|
| 17-bench mean proxy cost | **1.0109** |
| Best per-bench | 0.7644 |
| Worst per-bench | 1.2921 |
| Total overlaps | 0 |
| Wall, full suite | 15.5 h |
| Wall, per bench | ≤ 1 h (competition cap) |

### Local sweep (RTX 6000 Ada, same algorithm)

Per-bench breakdown from our local reproduction on the same GPU class.
Numbers differ from the grader by a few millicost due to platform-specific
float ordering; the algorithm and configuration are bit-identical.

| Bench | Proxy | Wall (s) | Status |
|---|---:|---:|:---:|
| ibm01 | 0.7745 | 3250 | VALID |
| ibm02 | 0.9897 | 3340 | VALID |
| ibm03 | 0.9256 | 3339 | VALID |
| ibm04 | 0.9334 | 3330 | VALID |
| ibm06 | 1.1007 | 3341 | VALID |
| ibm07 | 1.0586 | 3356 | VALID |
| ibm08 | 1.0591 | 3369 | VALID |
| ibm09 | 0.7748 | 3343 | VALID |
| ibm10 | 1.0039 | 3494 | VALID |
| ibm11 | 0.8406 | 3362 | VALID |
| ibm12 | 1.2366 | 3476 | VALID |
| ibm13 | 0.9198 | 3390 | VALID |
| ibm14 | 1.1598 | 3509 | VALID |
| ibm15 | 1.1548 | 3438 | VALID |
| ibm16 | 1.1168 | 3543 | VALID |
| ibm17 | 1.3398 | 3681 | VALID |
| ibm18 | 1.3064 | 3458 | VALID |
| **mean** | **1.0409** | | 17/17 |

---

## Math validation

Five unit tests confirm the Hessian-escape primitives recover known
eigenstructure at machine precision
(`submissions/vmallela_v7/tests/test_hessian_escape_math.py`).

| Test | Setup | Expected | Computed |
|---|---|---:|---:|
| Saddle x² − y² | known saddle | λ_min = −2 | −2.0000 |
| Minimum x² + y² | known min | λ_min = +2 | +2.0000 |
| Top-k diag(1,4,9,16) | known eigvals | [1, 4, 9] | [1, 4, 9] |
| Eigenvector orthogonality | symmetric H | exact | off-diag 4.4 × 10⁻¹⁶ |
| Termination check | saddle vs. min | continue / stop | both correct |

---

## Reproduction

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout v7-combinatorial-submission
git submodule update --init external/MacroPlacement
uv sync

# Single benchmark (≈ 1 h wall)
uv run evaluate submissions/vmallela_v7/placer.py --benchmark ibm15

# All 17 (≈ 16 h serial)
for b in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 \
         ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  uv run evaluate submissions/vmallela_v7/placer.py --benchmark "$b"
done
```

The submission configuration is baked into the placer via
`os.environ.setdefault` at module-import time, so the grader's
no-argument invocation (`OptimalPlacer().place(bench)`) reproduces
the submission run exactly. The baked configuration is:

```
PLACER_TOTAL_BUDGET        = 2300    # v4 pipeline budget (s)
PLACER_V6_WORKERS          = 1       # single worker
PLACER_V6_GPU_WORKERS      = 0       # no GPU worker
PLACER_V6_CONSENSUS        = 0       # no consensus refine
PLACER_V7_LAPLACIAN        = 1       # Phase 2 on
PLACER_V7_HESSIAN          = 1       # Phase 3 on
PLACER_V7_HESSIAN_BUDGET   = 1000    # parallel candidate budget (s)
PLACER_V7_HESSIAN_STEPS    = 0.02,-0.02,0.05,-0.05
PLACER_V7_HESSIAN_LANCZOS  = 50      # Lanczos iters
```

---

## Layout

```
.
├── README.md                  this file
├── COMPETITION.md             challenge specification (upstream)
├── SETUP.md                   environment + API reference (upstream)
├── pyproject.toml             pinned dependencies
│
├── macro_place/               benchmark loader, proxy-cost, utils
│
├── submissions/
│   └── vmallela_v7/           ←  the submission
│       ├── README.md          algorithm writeup, math derivation
│       ├── placer.py          OptimalPlacer entry point
│       ├── _hessian_escape.py Lanczos eigvec + termination
│       ├── _hessian_worker.py mp.Pool worker for parallel candidates
│       ├── _soft_laplacian.py Phase 2 closed-form HPWL solve
│       ├── _smooth_proxy.py   LSE-HPWL + CVaR-density surrogate
│       ├── _cell_window.py    windowed density (smooth proxy)
│       └── tests/             5 math validations + parity tests
│
├── results/
│   ├── RESULTS.md             submission notes + grader result
│   └── per_bench_results.csv  local-sweep raw data
│
├── assets/                    per-bench placement plots (v7_ibm*.png)
└── scripts/                   reproduction & plotting helpers
```

The detailed algorithm writeup, including failed approaches and the
full math derivation, lives at
[`submissions/vmallela_v7/README.md`](submissions/vmallela_v7/README.md).

Challenge specification:
[`COMPETITION.md`](COMPETITION.md).
