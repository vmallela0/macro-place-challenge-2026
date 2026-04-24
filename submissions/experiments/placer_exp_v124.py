"""exp_v124 (control for A1): soft CD with ASCENDING net_count order.

Tests whether the default 'most-connected first' ordering matters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from _softmacro_reverse import soft_macro_cd_reverse
_m.soft_macro_cd = soft_macro_cd_reverse

class OptimalPlacer(_m.OptimalPlacer):
    pass
