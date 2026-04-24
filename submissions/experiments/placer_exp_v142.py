"""exp_v142: Shrink 0.5 (aggressive)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    SHRINK_FACTOR = 0.5
