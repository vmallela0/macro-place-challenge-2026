"""exp_v145: Plateau threshold 1e-4 (loose)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    PLATEAU_THRESHOLD = 1e-4
