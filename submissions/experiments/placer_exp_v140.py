"""exp_v140: LNS n_candidates=60"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    LNS_N_CANDIDATES = 60
