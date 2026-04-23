"""exp_v27: Nesterov soft + stateful surrogate. Alternates between them per cycle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _nesterov_soft import nesterov_soft_cd
from _soft_surrogate_v2 import soft_cd_surrogate_v2, reset_surrogate_state

_call_count = [0]

def combined(pos_np, benchmark, incr_eval, max_time, verbose=False):
    _call_count[0] += 1
    # Half time Nesterov, half stateful surrogate
    _, c1 = nesterov_soft_cd(pos_np, benchmark, incr_eval, max_time * 0.5, momentum=0.6)
    _, c2 = soft_cd_surrogate_v2(pos_np, benchmark, incr_eval, max_time * 0.5)
    return pos_np, min(c1, c2)

import _softmacro
_softmacro.soft_macro_cd = combined
_m.soft_macro_cd = combined


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75

    def place(self, benchmark):
        _call_count[0] = 0
        reset_surrogate_state()
        return super().place(benchmark)
