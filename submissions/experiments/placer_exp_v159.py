"""exp_v159: surrogate-dominant, no FD (0/60/10/20/10)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.00, 0.60, 0.10, 0.20, 0.10)
