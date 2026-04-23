"""v10 with TOTAL_BUDGET read from env TOTAL_BUDGET_OVERRIDE.

Also scales legalize budget by n_hard (for dense benchmarks that need
more time to clear initial overlaps).
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10",
    str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)


class OptimalPlacer(_m.OptimalPlacer):
    def __init__(self, seed=42):
        super().__init__(seed)
        # Env override
        budget = os.environ.get("TOTAL_BUDGET_OVERRIDE")
        if budget:
            self.TOTAL_BUDGET = int(budget)

    def place(self, benchmark):
        # Scale legalize budget by n_hard
        n_hard = benchmark.num_hard_macros
        # Base is 45s for ~250 macros; scale linearly with generous factor
        self.LEGALIZE_BUDGET = max(45, int(60 * (n_hard / 250.0)))
        # Cap legalize at 20% of total budget
        self.LEGALIZE_BUDGET = min(self.LEGALIZE_BUDGET, max(60, int(self.TOTAL_BUDGET * 0.15)))
        # Scale CD1 to ~25% of remainder
        remain = self.TOTAL_BUDGET - self.LEGALIZE_BUDGET
        self.CD1_BUDGET = max(25, int(remain * 0.10))
        self.LNS_BUDGET = max(10, int(remain * 0.05))
        self.CD_LNS = max(5, int(remain * 0.025))
        # Each cycle uses ~10% of remaining; cycles loop until budget gone
        cycle_rem = remain - self.CD1_BUDGET - self.LNS_BUDGET - self.CD_LNS
        # We want ~10-20 cycles → budget per cycle
        est_cycles = 10
        per_cycle = max(20, int(cycle_rem / est_cycles))
        # Allocate within cycle: 10% FD, 40% soft CD, 40% soft LNS, 10% hard
        self.FD_PER_CYCLE = max(2, int(per_cycle * 0.05))
        self.SOFT_CD_PER_CYCLE = max(8, int(per_cycle * 0.40))
        self.SOFT_LNS_PER_CYCLE = max(8, int(per_cycle * 0.40))
        self.HARD_PER_CYCLE = max(5, int(per_cycle * 0.15))

        print(f"  [tune] n_hard={n_hard} total={self.TOTAL_BUDGET} "
              f"legalize={self.LEGALIZE_BUDGET} cd1={self.CD1_BUDGET} "
              f"cycle={per_cycle} (fd={self.FD_PER_CYCLE} soft={self.SOFT_CD_PER_CYCLE} "
              f"slns={self.SOFT_LNS_PER_CYCLE} hard={self.HARD_PER_CYCLE})")
        return super().place(benchmark)
