"""exp_v186: v164 config, but cycle ratios replaced with UCB1 bandit scheduler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bandit_placer import BanditPlacer

class OptimalPlacer(BanditPlacer):
    # RATIOS are ignored by BanditPlacer — UCB allocates instead.
    # Keep adaptive scheduler params from v164.
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
