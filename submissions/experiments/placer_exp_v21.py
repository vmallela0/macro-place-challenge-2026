"""exp_v21: informed MH proposals replacing uniform SA."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _informed_mh import mh_informed_soft
from _softmacro import soft_macro_cd as orig_soft_cd
import _softmacro

def combined(pos_np, benchmark, incr_eval, max_time, verbose=False):
    # 70% classic CD, 30% informed MH
    _, c1 = orig_soft_cd(pos_np, benchmark, incr_eval, max_time * 0.7, verbose=verbose)
    _, c2 = mh_informed_soft(pos_np, benchmark, incr_eval, max_time * 0.3, verbose=verbose)
    return pos_np, min(c1, c2)

_softmacro.soft_macro_cd = combined
_m.soft_macro_cd = combined


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
