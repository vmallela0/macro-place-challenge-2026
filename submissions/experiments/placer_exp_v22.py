"""exp_v22: fine-grain hard/soft interleaving. Many short 2s+2s cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
    CD1_BUDGET = 25
    LNS_BUDGET = 10
    CD_LNS = 5
    # Tiny cycles
    FD_PER_CYCLE = 1
    SOFT_CD_PER_CYCLE = 3
    SOFT_LNS_PER_CYCLE = 2
    HARD_PER_CYCLE = 2
