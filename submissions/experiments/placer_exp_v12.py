"""exp_v12: grid-based soft CD — per-macro 7x7 grid probe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _softmacro_fast import soft_cd_per_macro_grid
import _softmacro
_softmacro.soft_macro_cd = lambda pos_np, benchmark, incr_eval, max_time, verbose=False: \
    soft_cd_per_macro_grid(pos_np, benchmark, incr_eval, max_time,
                           grid_step=0.4, grid_radius=2, verbose=verbose)
_m.soft_macro_cd = _softmacro.soft_macro_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 180
