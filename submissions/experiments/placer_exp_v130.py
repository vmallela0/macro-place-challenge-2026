"""exp_v130: harness smoketest — defaults (baseline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('submissions/experiments').resolve()))
from _param_placer import ParameterizedPlacer

class OptimalPlacer(ParameterizedPlacer):
    pass
