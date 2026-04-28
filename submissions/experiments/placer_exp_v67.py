"""exp_v67: repeated v51 (per-net + adaptive, winner) with different seed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v51", str(Path(__file__).resolve().parent / "placer_exp_v51.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

class OptimalPlacer(_m.OptimalPlacer):
    def __init__(self, seed=20241225):
        super().__init__(seed=seed)
