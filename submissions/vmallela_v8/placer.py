"""vmallela_v8 — ARC + Replica Exchange + Riemannian descent.

Subclasses v7's OptimalPlacer. Overrides Phase 4.6 (Hessian escape) to run
the v8 phases (ARC → PT → Riemannian) when their env gates are set, with
strict-improvement fallback to whatever v7 produced.

Phase 4.5 (Adam) and Phase 5 (basin-hop) are inherited from v7 unchanged.

Env gates (off by default; gracefully falls back to v7):
    PLACER_V8_ARC=1         — Phase A: cubic regularization step
    PLACER_V8_REPLICA=1     — Phase B: parallel tempering
    PLACER_V8_RIEMANNIAN=1  — Phase C: Riemannian polish
"""
from __future__ import annotations
import os
import sys
import time
import math
from pathlib import Path
import random
import numpy as np
import torch

# Self-applying locked env (mirrors v7).
for _k, _v in [
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "42"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
]:
    os.environ.setdefault(_k, _v)

_HERE = Path(__file__).resolve().parent
# v8 first so its modules win on name collisions; then v7 for shared deps.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vmallela"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v2"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v6"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v7"))

try:
    import threadpoolctl as _tp
    _tp.threadpool_limits(1)
except ImportError:
    pass

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# Load v7 OptimalPlacer with a non-colliding module name so our class
# below remains the one evaluate.py picks up (it filters on
# __module__ == path.stem == "placer").
import importlib.util as _ilu

_v7_path = _HERE.parent / "vmallela_v7" / "placer.py"
_v7_spec = _ilu.spec_from_file_location("_v7_for_v8", str(_v7_path))
_v7_mod = _ilu.module_from_spec(_v7_spec)
sys.modules["_v7_for_v8"] = _v7_mod
_v7_spec.loader.exec_module(_v7_mod)
_V7Placer = _v7_mod.OptimalPlacer

from _runlog import log as runlog


class OptimalPlacer(_V7Placer):
    """v8 = v7 + (ARC | PT | Riemannian) overrides on Phase 4.6."""

    def _hessian_escape_phase(self, current_pos, current_cost, bench_path,
                                step_sizes, hop_budget, n_lanczos_iters):
        """v8 dispatch: ARC → PT → Riemannian, each env-gated.

        If none enabled, defers to v7's behaviour (grid search / topk /
        mirror as configured by PLACER_SLJ2_*).
        """
        v8_arc = os.environ.get("PLACER_V8_ARC", "0") == "1"
        v8_pt = os.environ.get("PLACER_V8_REPLICA", "0") == "1"
        v8_riem = os.environ.get("PLACER_V8_RIEMANNIAN", "0") == "1"

        if not (v8_arc or v8_pt or v8_riem):
            return super()._hessian_escape_phase(
                current_pos, current_cost, bench_path,
                step_sizes, hop_budget, n_lanczos_iters)

        runlog("placer", "v8_phase_enter",
               f"arc={int(v8_arc)} pt={int(v8_pt)} riem={int(v8_riem)} "
               f"current_cost={float(current_cost):.6f} budget={hop_budget}s")

        try:
            from _placer_phases import run_v8_phases
            new_pos, new_cost = run_v8_phases(
                self, current_pos, current_cost, bench_path,
                hop_budget=hop_budget,
                n_lanczos_iters=n_lanczos_iters,
                v8_arc=v8_arc, v8_pt=v8_pt, v8_riem=v8_riem)
        except Exception as e:
            runlog("placer", "v8_phase_error", f"{type(e).__name__}: {e}")
            print(f"  [v8] phase error ({type(e).__name__}: {e}); "
                  f"falling back to v7 path", flush=True)
            return super()._hessian_escape_phase(
                current_pos, current_cost, bench_path,
                step_sizes, hop_budget, n_lanczos_iters)

        # Strict-improvement gate vs the input.
        if isinstance(new_pos, np.ndarray):
            new_pos = torch.tensor(new_pos, dtype=torch.float32)
        if new_cost < float(current_cost) - 1e-7:
            runlog("placer", "v8_phase_win",
                   f"{float(current_cost):.6f} -> {new_cost:.6f}")
            return new_pos, new_cost
        runlog("placer", "v8_phase_keep",
               f"no improvement; keep {float(current_cost):.6f}")
        return current_pos, current_cost
