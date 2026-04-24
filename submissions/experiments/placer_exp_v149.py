"""exp_v149: Initial divisor 10 (bigger cycle)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    INITIAL_CYCLE_DIVISOR = 10
