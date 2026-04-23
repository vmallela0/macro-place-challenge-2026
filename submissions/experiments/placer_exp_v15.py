"""exp_v15: surrogate-guided soft CD (MLP-ranked probes, idea #1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _soft_surrogate import soft_cd_surrogate
import _softmacro
_softmacro.soft_macro_cd = lambda pos_np, benchmark, incr_eval, max_time, verbose=False: \
    soft_cd_surrogate(pos_np, benchmark, incr_eval, max_time,
                      warmup_frac=0.25, cand_per_macro=20, verify_top_k=2,
                      verbose=verbose)
_m.soft_macro_cd = _softmacro.soft_macro_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 180
    # Bigger soft cycles to let surrogate pay off
    SOFT_CD_PER_CYCLE = 15
    LEGALIZE_BUDGET = 70  # extra time for CPU-contention safety
