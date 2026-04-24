"""exp_v190: v164 config + bandit scheduler with c=0.01 (UCB bonus scaled down).

Observed issue in v186: reward magnitudes are ~1e-3/sec but default c=sqrt(2)
gives UCB bonus ~1.5, so exploration dominates and ratios stay ~uniform forever.
Fix: scale c down by ~100x.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bandit_placer import BanditPlacer

class OptimalPlacer(BanditPlacer):
    UCB_C = 0.005  # tuned for observed reward scale ~1e-3 per sec
    GROW_FACTOR = 1.3
    INITIAL_CYCLE_DIVISOR = 10
