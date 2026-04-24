"""exp_v176: v164 jitter (7/48/15/15/15) - more fd, less sur"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.07, 0.48, 0.15, 0.15, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
