"""exp_v189: bandit scheduler with prior favoring the surrogate arm (higher init pulls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bandit_placer import BanditPlacer

class OptimalPlacer(BanditPlacer):
    # Surrogate (idx 1) gets both higher initial pulls AND a positive prior reward
    # to bias early cycles toward it. This mimics the v164 finding that sur=0.50 works well.
    INIT_PULLS = (1, 4, 1, 1, 1)
    INIT_REWARDS = (0.0, 0.5, 0.0, 0.0, 0.0)  # seeded belief sur gives ~0.5/s early
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
