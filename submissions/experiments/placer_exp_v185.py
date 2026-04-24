"""exp_v185: v164 config + quadratic-placement (Kahng-Kennings) tournament seed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qp_init_placer import QPInitPlacer

class OptimalPlacer(QPInitPlacer):
    RATIOS = (0.05, 0.50, 0.15, 0.15, 0.15)
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
