"""exp_v188: QP init with v2 defaults (to isolate QP's effect, no v164 config change)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qp_init_placer import QPInitPlacer

class OptimalPlacer(QPInitPlacer):
    # Use ParameterizedPlacer defaults (matches submission_v4 behavior)
    RATIOS = (0.05, 0.30, 0.15, 0.30, 0.20)
    GROW_FACTOR = 1.1
    INITIAL_CYCLE_DIVISOR = 15
