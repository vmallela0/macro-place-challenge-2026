"""exp_v187: QP init + UCB bandit scheduler (combined)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qp_bandit_placer import QPBanditPlacer

class OptimalPlacer(QPBanditPlacer):
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
