"""exp_v179: v164 + GROW_FACTOR=1.4"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.15, 0.15, 0.15)
    GROW_FACTOR = 1.4
    INITIAL_CYCLE_DIVISOR = 10
