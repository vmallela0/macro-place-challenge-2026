"""exp_v56: adaptive hard+soft alternating cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v36", str(Path(__file__).resolve().parent / "placer_exp_v36.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
