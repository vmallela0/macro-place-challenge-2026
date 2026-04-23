"""exp_v30: more hard-CD + LNS before starting soft cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
    CD1_BUDGET = 45     # doubled from 25
    LNS_BUDGET = 20     # doubled
    CD_LNS = 10
    # Less time for soft cycles
    FD_PER_CYCLE = 2
    SOFT_CD_PER_CYCLE = 6
    SOFT_LNS_PER_CYCLE = 8
    HARD_PER_CYCLE = 4
