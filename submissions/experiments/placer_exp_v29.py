"""exp_v29: annealed delta schedule across soft cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _annealed_soft import soft_cd_annealed, reset_phase
import _softmacro
_softmacro.soft_macro_cd = soft_cd_annealed
_m.soft_macro_cd = soft_cd_annealed


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75

    def place(self, benchmark):
        reset_phase()
        return super().place(benchmark)
