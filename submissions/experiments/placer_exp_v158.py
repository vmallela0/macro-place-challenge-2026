"""exp_v158: v133 ratios but zero regular CD (surrogate subsumes it)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.55, 0.00, 0.25, 0.15)
