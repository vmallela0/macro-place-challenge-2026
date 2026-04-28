"""exp_v53: Grover-like amplitude amplification on softs + adaptive CD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _quantum_amp import quantum_amp_soft
from _softmacro import soft_macro_cd as orig

def combined(pos_np, benchmark, incr_eval, max_time, verbose=False):
    # 50% qamp, 50% regular soft CD
    _, c1 = quantum_amp_soft(pos_np, benchmark, incr_eval, max_time * 0.5,
                              n_candidates=16, n_iters=50, verbose=verbose)
    _, c2 = orig(pos_np, benchmark, incr_eval, max_time * 0.5, verbose=verbose)
    return pos_np, min(c1, c2)

import _softmacro
_softmacro.soft_macro_cd = combined
_m.soft_macro_cd = combined


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
