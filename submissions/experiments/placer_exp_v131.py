"""exp_v131: RATIOS double-FD (10/30/15/25/20)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.10, 0.30, 0.15, 0.25, 0.20)
