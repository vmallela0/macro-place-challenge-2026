"""exp_v69: v51 variant — bigger per-net budget (30s)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v51", str(Path(__file__).resolve().parent / "placer_exp_v51.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    def __init__(self, seed=42):
        super().__init__(seed=seed)
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 60  # trim to give per-net more time

    def place(self, benchmark):
        return super().place(benchmark)
