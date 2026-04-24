"""Fast equivalence check: 30 random moves on ibm01 only vs batch PlacementCost.
For gating small evaluator refactors — the full test_evaluator_equivalence.py
takes ~30min on this laptop because it walks ibm10. Use the full test pre-commit
whenever you're not sure."""
import os
import sys
import random
import importlib.util
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

_vmallela_path = ROOT / "submissions" / "vmallela" / "placer.py"
_spec = importlib.util.spec_from_file_location("_vmallela_placer", str(_vmallela_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
IncrementalEvaluator = _mod.IncrementalEvaluator

TESTCASE_ROOT = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

BENCH = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
N_MOVES = int(sys.argv[2]) if len(sys.argv) > 2 else 30

print(f"[quick-eq] {BENCH}: {N_MOVES} moves")
bench_dir = TESTCASE_ROOT / BENCH
benchmark, plc = load_benchmark_from_dir(str(bench_dir))
n_hard = benchmark.num_hard_macros
cw = float(benchmark.canvas_width)
ch = float(benchmark.canvas_height)

incr = IncrementalEvaluator(plc, benchmark)
incr_initial = incr.get_proxy_cost()
placement = benchmark.macro_positions.clone()
batch_initial = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
init_diff = abs(batch_initial - incr_initial)
print(f"[quick-eq] initial: batch={batch_initial:.8f} incr={incr_initial:.8f} diff={init_diff:.2e}")

movable = [i for i in range(n_hard) if not bool(benchmark.macro_fixed[i])]
rng = random.Random(42)
sizes = benchmark.macro_sizes[:n_hard].numpy()
max_abs = init_diff

current = placement.clone()
for i in range(N_MOVES):
    m = rng.choice(movable)
    w, h = sizes[m]
    x = rng.uniform(w / 2, cw - w / 2)
    y = rng.uniform(h / 2, ch - h / 2)
    incr.move_macro(m, x, y)
    current[m, 0] = float(x)
    current[m, 1] = float(y)
    incr_c = incr.get_proxy_cost()
    batch_c = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
    d = abs(incr_c - batch_c)
    if d > max_abs:
        max_abs = d
    if i % 5 == 0 or d > 1e-4:
        print(f"[quick-eq] move {i}: incr={incr_c:.8f} batch={batch_c:.8f} diff={d:.2e} "
              f"(max={max_abs:.2e})")

print(f"\n[quick-eq] FINAL max_abs_diff={max_abs:.2e} ({'PASS' if max_abs < 1e-4 else 'FAIL'})")
sys.exit(0 if max_abs < 1e-4 else 1)
