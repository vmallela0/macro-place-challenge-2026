"""exp_v88: sub_v4 at 1500s on ibm01 — super-long budget convergence test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
class OptimalPlacer(_m.OptimalPlacer):
    pass
