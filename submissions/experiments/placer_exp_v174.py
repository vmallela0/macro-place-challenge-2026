"""exp_v174: v164 jitter (5/50/15/17/13) - more lns, less hard"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.15, 0.17, 0.13)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
