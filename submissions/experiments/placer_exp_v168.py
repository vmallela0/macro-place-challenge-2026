"""exp_v168: 10/45/15/20/10 (more FD)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.10, 0.45, 0.15, 0.20, 0.10)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
