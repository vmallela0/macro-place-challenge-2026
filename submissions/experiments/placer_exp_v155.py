"""exp_v155: stacked winners — v133 ratios + v144 grow + v149 divisor + v136 regCD boost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    RATIOS = (0.05, 0.50, 0.10, 0.20, 0.15)  # v133 winner
    GROW_FACTOR = 1.3                        # v144 marginal winner
    INITIAL_CYCLE_DIVISOR = 10               # v149 marginal
