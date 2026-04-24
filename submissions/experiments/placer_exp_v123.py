"""exp_v123 (idea D12): replaces soft_macro_cd with UCB1-ordered probe variant.

Hypothesis: probing directions in UCB-learned order (early-accept first
improving direction) speeds soft CD enough to fit more cycles in the same
budget, yielding better ibm01 cost at 220 s.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

# Monkey-patch the soft_macro_cd used by submission_v4
from _softmacro_ucb import soft_macro_cd_ucb
_m.soft_macro_cd = soft_macro_cd_ucb

class OptimalPlacer(_m.OptimalPlacer):
    pass
