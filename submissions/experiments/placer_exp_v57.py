"""exp_v57: lazy (active-set) soft CD — faster convergence on sparse updates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _lazy_soft import lazy_soft_cd
import _softmacro
_softmacro.soft_macro_cd = lazy_soft_cd
_m.soft_macro_cd = lazy_soft_cd


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
