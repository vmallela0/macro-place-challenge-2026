"""exp_v108: sub_v4 @ 1200s seed=1729 (Ramanujan) — test budget-depth on ibm03."""
import os
os.environ.setdefault("PLACER_TOTAL_BUDGET", "1200")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

class OptimalPlacer(_m.OptimalPlacer):
    def __init__(self, seed=1729):
        super().__init__(seed=seed)
