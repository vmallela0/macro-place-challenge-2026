"""exp_v132: RATIOS no-FD (0/40/15/25/20)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.00, 0.40, 0.15, 0.25, 0.20)
