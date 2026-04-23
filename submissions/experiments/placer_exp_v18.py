"""exp_v18: batch FD for soft macros (all softs move simultaneously per iteration)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _batch_fd import batch_fd_soft
import _softmacro
_softmacro.soft_macro_cd = lambda pos_np, benchmark, incr_eval, max_time, verbose=False: \
    batch_fd_soft(pos_np, benchmark, incr_eval, max_time, damping=0.3, verbose=verbose)
_m.soft_macro_cd = _softmacro.soft_macro_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
    # With batch FD we can afford more soft CD time since each iter is fast
    SOFT_CD_PER_CYCLE = 20
