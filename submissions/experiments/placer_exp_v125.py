"""exp_v125 (idea A19): LNS seed-selection weighted by incident-net size."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _soft_lns_congbias import soft_lns_congbias_phase
_m.soft_lns_phase = soft_lns_congbias_phase

class OptimalPlacer(_m.OptimalPlacer):
    pass
