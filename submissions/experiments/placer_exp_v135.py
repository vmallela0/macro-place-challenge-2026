"""exp_v135: RATIOS more-hard (5/25/10/20/40)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.25, 0.10, 0.20, 0.40)
