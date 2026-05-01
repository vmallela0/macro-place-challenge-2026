# lsj — c4d reproduction of vmallela_v7

Reproduction of the `vmallela_v7` macro-placement submission on a fresh
**GCP `c4d-standard-16`** instance (16 vCPU AMD EPYC Turin, 62 GB RAM,
Linux 6.1, Python 3.11). Same code, same env vars, no GPU.

**Target.** Match the dev-box 17-bench mean proxy cost of **1.0003** within
±0.005 cross-architecture variance, with `overlaps=0` on every bench.

This branch (`lsj`) is the live record: each bench's row, plot, and the
running mean get pushed here as soon as it finishes.

## The algorithm in one paragraph

A standard placer (simulated annealing + LNS) gets stuck not at a local
minimum but at a **saddle** — every spatial move looks uphill, but a
non-local *curvature* direction still goes downhill. We compute the
**Hessian** of a smooth surrogate of the proxy cost (LSE-smoothed HPWL +
top-10% CVaR density), find its smallest eigenvalue via Lanczos with
PyTorch double-backward Hessian-vector products (no full N×N matrix),
and if `λ_min < 0` we step ±`v_min` to escape. The pipeline:

```
  Phase 1 (2300 s)   single v4 worker — SA + LNS + soft cycles + escape basin
                     → overlap-free placement at a saddle of the exact proxy
  Phase 2 (~30 s)    Laplacian soft-resolve (closed-form HPWL-quadratic min,
                     strict-improvement gated against exact proxy)
  Phase 3 (1000 s)   Hessian saddle escape — Lanczos on H, 8 candidates
                     ±{0.02, 0.05}·v_min, run v4 from each in parallel,
                     keep the lowest overlap-free result
```

Every phase has a **strict-improvement gate** against the exact (non-smooth)
proxy cost — the algorithm never makes a placement worse.

Wall budget per bench: **3600 s** hard. ~16 h for the full 17-bench sweep.

## Live results

<!--LSJ:START-->
_Live results — c4d-standard-16, 16 vCPU AMD EPYC Turin. 0 / 17 complete._

| Bench | Proxy cost | Dev-box ref | Δ (this − dev) | Overlaps | Wall (s) | PNG |
|---|---:|---:|---:|---:|---:|:---:|
| ibm01 | _pending_ | 0.7653 | — | — | — | — |
| ibm02 | _pending_ | 0.9482 | — | — | — | — |
| ibm03 | _pending_ | 0.9166 | — | — | — | — |
| ibm04 | _pending_ | 0.9287 | — | — | — | — |
| ibm06 | _pending_ | 1.0546 | — | — | — | — |
| ibm07 | _pending_ | 1.0324 | — | — | — | — |
| ibm08 | _pending_ | 1.0291 | — | — | — | — |
| ibm09 | _pending_ | 0.7628 | — | — | — | — |
| ibm10 | _pending_ | 0.9492 | — | — | — | — |
| ibm11 | _pending_ | 0.8013 | — | — | — | — |
| ibm12 | _pending_ | 1.1557 | — | — | — | — |
| ibm13 | _pending_ | 0.8757 | — | — | — | — |
| ibm14 | _pending_ | 1.1070 | — | — | — | — |
| ibm15 | _pending_ | 1.0835 | — | — | — | — |
| ibm16 | _pending_ | 1.0435 | — | — | — | — |
| ibm17 | _pending_ | 1.2813 | — | — | — | — |
| ibm18 | _pending_ | 1.2697 | — | — | — | — |

_Sweep has not started — table will populate as each bench finishes._
<!--LSJ:END-->

## Files in this branch

- `lsj/results.csv` — appended one row per bench (proxy, density, wirelength, congestion, overlaps, wall, exit, UTC timestamp)
- `lsj/png/<bench>.png` — placement plot for each completed bench
- `lsj/update_readme.py` — regenerates the table above from `results.csv`
- `lsj/watcher.sh` — polls `/tmp/v7_singlev4_sweep_*/results.csv`, copies new rows + PNGs into the branch, commits, pushes
- `lsj/run_pipeline.sh` — top-level entry that launches the sweep and the watcher as detached daemons
- `submissions/vmallela_v7/` — unchanged submission artifact (placer + Hessian module)
- `scripts/v7_singlev4_full_sweep.sh` — unchanged sweep driver (validates env, runs all 17 benches)

## Reproducing locally

```bash
git clone https://github.com/vmallela0/macro-place-challenge-2026
cd macro-place-challenge-2026
git checkout lsj
git submodule update --init external/MacroPlacement
uv venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy 'matplotlib>=3.5' tqdm 'absl-py>=1.0' 'scipy>=1.10' 'threadpoolctl>=3.0'
.venv/bin/pip install --no-deps -e .

# smoke (~57 min) — expect proxy ≈ 1.0835 ± 0.005 on ibm15
.venv/bin/python -m macro_place.evaluate submissions/vmallela_v7/placer.py --benchmark ibm15

# full sweep + auto-push (detached, ~16 h)
bash lsj/run_pipeline.sh
```
