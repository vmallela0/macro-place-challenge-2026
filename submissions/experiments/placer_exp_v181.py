"""exp_v181: v164 + INITIAL_CYCLE_DIVISOR=12"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.15, 0.15, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 12
