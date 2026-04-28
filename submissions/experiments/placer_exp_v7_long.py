"""exp_v7 with extended budget to see where soft-CD converges."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v7", str(Path(__file__).resolve().parent / "placer_exp_v7.py"))
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

class OptimalPlacer(_m.OptimalPlacer):
    LEGALIZE_BUDGET = 55
    CD1_BUDGET = 60
    LNS_A = 20
    CD_A = 10
    SOFT_PER_CYCLE = 30
    HARD_PER_CYCLE = 8
    TOTAL_BUDGET = 600
