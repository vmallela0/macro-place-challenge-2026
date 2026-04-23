"""exp_v31: MINIMIZE hard phases, maximize soft cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
    CD1_BUDGET = 15     # less hard CD
    LNS_BUDGET = 5
    CD_LNS = 3
    FD_PER_CYCLE = 1
    SOFT_CD_PER_CYCLE = 15   # more soft
    SOFT_LNS_PER_CYCLE = 8
    HARD_PER_CYCLE = 3
