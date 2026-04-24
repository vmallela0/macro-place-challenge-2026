"""exp_v162: v155 repeat (different seed 2024) — for jitter sanity check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.10, 0.20, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
    def __init__(self, seed=2024):
        super().__init__(seed=seed)
