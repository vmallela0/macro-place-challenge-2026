"""exp_v37: v10 + permutation swap on hard-macro subsets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _perm_swap import permutation_swap_hard
from _softmacro import soft_macro_cd

def with_perm(pos_np, benchmark, incr_eval, max_time, verbose=False):
    # 30% perm swap on hards, 70% regular soft CD
    _, c1 = permutation_swap_hard(pos_np, benchmark, incr_eval, max_time * 0.3, k=4, verbose=verbose)
    _, c2 = soft_macro_cd(pos_np, benchmark, incr_eval, max_time * 0.7, verbose=verbose)
    return pos_np, min(c1, c2)

import _softmacro
_softmacro.soft_macro_cd = with_perm
_m.soft_macro_cd = with_perm


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
