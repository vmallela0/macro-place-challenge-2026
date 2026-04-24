"""exp_v134: RATIOS more-LNS (5/25/10/40/20)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.25, 0.10, 0.40, 0.20)
