# vmallela_v8 — ARC + Replica Exchange + Riemannian descent

Branch `v8`. Cumulative additions over `slj2`:

1. **Phase A — Adaptive Regularization with Cubics (ARC)**: replaces v7's
   grid search `{±0.02, ±0.05}` step pattern with one principled cubic-
   regularized step (Cartis-Gould-Toint 2011). Uses the existing Lanczos
   basis from v7's Hessian path; the cubic subproblem is solved on the
   k×k Krylov projection, costing roughly zero extra HVPs.
2. **Phase B — Replica Exchange (Parallel Tempering)**: M=8 chains at
   geometric temperature ladder, swap every τ steps. Lowest-T chain's
   running best (exact-cost gated) is the output. Replaces the
   8-candidate portfolio.
3. **Phase C — Riemannian descent**: ~200 gradient-preserving steps on
   the no-overlap manifold after PT. Tangent-projection + windowed
   retraction. Local refinement, not exploration.

All phases additive on top of v7's pipeline. v7/slj2 code unchanged.

## Live results

<!--V8:START-->
_v8 results — 0 / 17 complete. ibm15-first ordering._

| Bench | v8 proxy | v7 dev-box | Δ vs v7 | Overlaps | Wall (s) | Status | PNG |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| ibm15 | — | 1.0835 | — | — | — | — | — |
| ibm17 | — | 1.2813 | — | — | — | — | — |
| ibm18 | — | 1.2697 | — | — | — | — | — |
| ibm12 | — | 1.1557 | — | — | — | — | — |
| ibm14 | — | 1.1070 | — | — | — | — | — |
| ibm16 | — | 1.0435 | — | — | — | — | — |
| ibm13 | — | 0.8757 | — | — | — | — | — |
| ibm04 | — | 0.9287 | — | — | — | — | — |
| ibm06 | — | 1.0546 | — | — | — | — | — |
| ibm07 | — | 1.0324 | — | — | — | — | — |
| ibm08 | — | 1.0291 | — | — | — | — | — |
| ibm09 | — | 0.7628 | — | — | — | — | — |
| ibm10 | — | 0.9492 | — | — | — | — | — |
| ibm11 | — | 0.8013 | — | — | — | — | — |
| ibm01 | — | 0.7653 | — | — | — | — | — |
| ibm02 | — | 0.9482 | — | — | — | — | — |
| ibm03 | — | 0.9166 | — | — | — | — | — |

<!--V8:END-->

## Files

- `placer.py` — entry point. Subclasses v7 OptimalPlacer; overrides Phase
  4.6 (Hessian) and adds Phase 5 (Riemannian).
- `_arc.py`, `_arc_subproblem.py` — Phase A.
- `_replica_exchange.py`, `_pt_worker.py` — Phase B.
- `_riemannian.py`, `_short_pushapart.py` — Phase C.
- `_checkpoint.py`, `_resource_guard.py`, `_runlog.py` — operational scaffolding.
- `run_v8.sh` — top-level launcher (env, phase loop, sweep).
- `update_readme.py` — regenerates the live-results table from results.csv.
- `RUNLOG.md` — append-only operational log.
- `tests/` — phase math tests + cross-platform parity test.

## Env vars

- `PLACER_V8_ARC=1` — enable Phase A (ARC). Default 0.
- `PLACER_V8_REPLICA=1` — enable Phase B (PT). Default 0.
- `PLACER_V8_RIEMANNIAN=1` — enable Phase C (Riemannian). Default 0.
- `PLACER_V8_PT_CHAINS` — number of PT chains. Default 8.
- `PLACER_V8_PT_STEPS` — total PT steps (split across chains). Default 8000.
- `PLACER_V8_PT_TMIN`, `PLACER_V8_PT_TMAX` — temperature ladder. Defaults 0.01, 1.0.
- `PLACER_V8_RIEM_STEPS` — Riemannian steps. Default 200.
- `PLACER_V8_ARC_M_INIT` — ARC M_init. Default 1.0.

All other env vars (PLACER_V7_*, PLACER_SLJ2_*) carry over from v7/slj2 unchanged.

## Cross-platform contract

- Device: auto (CUDA > MPS > CPU).
- All tensors: explicit `dtype=torch.float32`, explicit `device=device`.
- `multiprocessing.get_context("spawn")` for any pool — `fork` breaks
  CUDA on the smoke pod and breaks Mac entirely.
- Seeds set on every device (`torch.cuda.manual_seed_all` if CUDA).
- Deterministic mode `warn_only=True`.

## Smoke / sweep

```bash
# Smoke (ibm15 only)
PLACER_V8_ARC=1 PLACER_V8_REPLICA=1 PLACER_V8_RIEMANNIAN=1 \
  PLACER_SLJ2_POOL=8 \
  ./submissions/vmallela_v8/run_v8.sh --smoke

# Full sweep, ibm15-first, auto-push per bench
PLACER_V8_ARC=1 PLACER_V8_REPLICA=1 PLACER_V8_RIEMANNIAN=1 \
  PLACER_SLJ2_POOL=8 \
  ./submissions/vmallela_v8/run_v8.sh --full-sweep
```
