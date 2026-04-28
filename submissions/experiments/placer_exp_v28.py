"""exp_v28: (1+1)-ES on softs interleaved with normal soft CD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _oneplusone import one_plus_one_soft
from _softmacro import soft_macro_cd as orig

def combined(pos_np, benchmark, incr_eval, max_time, verbose=False):
    # 75% CD, 25% ES
    _, c1 = orig(pos_np, benchmark, incr_eval, max_time * 0.75, verbose=verbose)
    _, c2 = one_plus_one_soft(pos_np, benchmark, incr_eval, max_time * 0.25,
                              sigma_init=0.8, k_mutate=30, verbose=verbose)
    return pos_np, min(c1, c2)

import _softmacro
_softmacro.soft_macro_cd = combined
_m.soft_macro_cd = combined


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
