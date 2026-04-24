"""exp_v127 (idea IP7): center-collapsed init (all macros at canvas center + jitter)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
from _init_variants import center_init


class OptimalPlacer(_m.OptimalPlacer):
    def place(self, benchmark):
        n_hard = benchmark.num_hard_macros
        alt = center_init(benchmark.macro_positions[:n_hard].numpy().astype('float64'), benchmark)
        benchmark.macro_positions[:n_hard] = torch.from_numpy(alt).float()
        return super().place(benchmark)
