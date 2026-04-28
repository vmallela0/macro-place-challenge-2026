"""exp_v32: density-aware soft CD (candidates bias toward low-density cells)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _density_aware_soft import density_aware_soft_cd
import _softmacro
_softmacro.soft_macro_cd = density_aware_soft_cd
_m.soft_macro_cd = density_aware_soft_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
