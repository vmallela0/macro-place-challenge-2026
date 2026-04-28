"""exp_v60: worst-net-first HPWL optimization + soft CD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _worst_net import worst_net_soft
from _softmacro import soft_macro_cd as orig

def combined(pos, bench, incr, t, verbose=False):
    _, c1 = worst_net_soft(pos, bench, incr, t * 0.4, verbose=verbose)
    _, c2 = orig(pos, bench, incr, t * 0.6, verbose=verbose)
    return pos, min(c1, c2)

import _softmacro
_softmacro.soft_macro_cd = combined
_m.soft_macro_cd = combined


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
