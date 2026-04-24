"""exp_v143: Shrink 0.85 (conservative)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    SHRINK_FACTOR = 0.85
