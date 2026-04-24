"""exp_v136: RATIOS more-regCD (5/25/30/25/15)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.25, 0.30, 0.25, 0.15)
