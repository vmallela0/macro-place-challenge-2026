"""exp_v26: Nesterov-momentum soft CD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _nesterov_soft import nesterov_soft_cd
import _softmacro
_softmacro.soft_macro_cd = lambda pos_np, benchmark, incr_eval, max_time, verbose=False: \
    nesterov_soft_cd(pos_np, benchmark, incr_eval, max_time, momentum=0.6, verbose=verbose)
_m.soft_macro_cd = _softmacro.soft_macro_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
