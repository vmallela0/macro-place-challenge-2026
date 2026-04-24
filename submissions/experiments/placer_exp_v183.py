"""exp_v183: v164 + PLATEAU_THRESHOLD=1e-7, PLATEAU_COUNT=6"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.15, 0.15, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
    PLATEAU_THRESHOLD = 1e-7
    PLATEAU_COUNT = 6
