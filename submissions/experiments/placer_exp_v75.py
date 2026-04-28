"""exp_v75: run submission_v4 at 400s — comparison to v73 (v51 400s)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

# Override via env at launch time (see placer_submission_v4 behavior)
class OptimalPlacer(_m.OptimalPlacer):
    pass
