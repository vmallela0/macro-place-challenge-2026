"""exp_v157: surrogate-heavy with some LNS preserved (5/45/10/30/10)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.45, 0.10, 0.30, 0.10)
