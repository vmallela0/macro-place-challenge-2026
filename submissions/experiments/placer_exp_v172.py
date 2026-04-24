"""exp_v172: v164 jitter (5/50/13/17/15) - less regCD, more lns"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.13, 0.17, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
