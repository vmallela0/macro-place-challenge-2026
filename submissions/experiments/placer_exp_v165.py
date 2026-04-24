"""exp_v165: 5/50/20/15/10 (more regCD, less hard)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.20, 0.15, 0.10)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
