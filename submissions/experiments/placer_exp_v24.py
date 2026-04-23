"""exp_v24: soft CD + periodic perturb-restart when plateau."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _soft_restart import soft_perturb_and_reopt
from _softmacro import soft_macro_cd

# Wrap: every 3rd call, do a perturb-restart
_call_count = [0]
_orig_soft = soft_macro_cd

def wrapped(pos_np, benchmark, incr_eval, max_time, verbose=False):
    _call_count[0] += 1
    if _call_count[0] % 3 == 0:
        # Perturb restart
        _, c = soft_perturb_and_reopt(pos_np, benchmark, incr_eval, _orig_soft,
                                       perturb_frac=0.2, perturb_sigma=1.5,
                                       reopt_time=max_time * 0.6,
                                       verbose=verbose)
        # And a normal CD after
        _, c2 = _orig_soft(pos_np, benchmark, incr_eval, max_time * 0.4, verbose=verbose)
        return pos_np, min(c, c2)
    return _orig_soft(pos_np, benchmark, incr_eval, max_time, verbose=verbose)

import _softmacro
_softmacro.soft_macro_cd = wrapped
_m.soft_macro_cd = wrapped


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75

    def place(self, benchmark):
        _call_count[0] = 0
        return super().place(benchmark)
