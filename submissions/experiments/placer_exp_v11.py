"""exp_v11: 4-direction soft CD (halved probes, 2x macros visited)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("_v10", str(Path(__file__).resolve().parent / "placer_exp_v10.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

# Monkey-patch soft_macro_cd to use 4-direction version
from _softmacro_fast import fast_soft_cd_4dir
import _softmacro
_softmacro.soft_macro_cd = fast_soft_cd_4dir
# Also repatch inside the imported _v10 (its closure already captured soft_macro_cd)
_m.soft_macro_cd = fast_soft_cd_4dir


class OptimalPlacer(_m.OptimalPlacer):
    TOTAL_BUDGET = 220
    LEGALIZE_BUDGET = 75
