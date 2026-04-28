"""exp_v19: surrogate v2 — shared model across cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _soft_surrogate_v2 import soft_cd_surrogate_v2, reset_surrogate_state
import _softmacro
_softmacro.soft_macro_cd = soft_cd_surrogate_v2
_m.soft_macro_cd = soft_cd_surrogate_v2


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75

    def place(self, benchmark):
        reset_surrogate_state()  # fresh state per benchmark
        return super().place(benchmark)
