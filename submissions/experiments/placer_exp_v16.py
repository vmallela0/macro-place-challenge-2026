"""exp_v16: v10 + targeted soft-LNS on congestion hotspots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _targeted_soft_lns import targeted_soft_lns_phase
import _soft_lns
_soft_lns.soft_lns_phase = targeted_soft_lns_phase
_m.soft_lns_phase = targeted_soft_lns_phase


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
