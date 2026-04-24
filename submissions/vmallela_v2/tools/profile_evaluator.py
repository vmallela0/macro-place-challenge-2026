"""Profile the hot path of IncrementalEvaluator to find Cython/GPU port targets.

Runs ~300 random macro moves on ibm01, collects per-function stats via cProfile.
Expected dominant functions:
  - __smooth_routing_cong   (called every move, O(G^2) work)
  - __two_pin_net_routing   (Python for-loop per net cell)
  - _recompute_congestion_cost (sort + concat)
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "submissions" / "vmallela"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_vmallela_placer", str(_ROOT / "submissions" / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from macro_place.benchmark import Benchmark

IncrementalEvaluator = _mod.IncrementalEvaluator
_load_plc = _mod._load_plc

BENCH = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
N_MOVES = int(sys.argv[2]) if len(sys.argv) > 2 else 300

benchmark = Benchmark.load(str(_ROOT / "benchmarks" / "processed" / "public" / f"{BENCH}.pt"))
plc = _load_plc(BENCH)
incr = IncrementalEvaluator(plc, benchmark)
print(f"[profile] {BENCH}: n_hard={benchmark.num_hard_macros} n_nets={incr.n_nets} "
      f"grid={incr.grid_col}x{incr.grid_row}")

sizes_np = benchmark.macro_sizes[:benchmark.num_hard_macros].numpy()
cw = float(benchmark.canvas_width)
ch = float(benchmark.canvas_height)

rng = np.random.RandomState(0)

def one_move():
    m = rng.randint(benchmark.num_hard_macros)
    w, h = sizes_np[m]
    x = rng.uniform(w / 2, cw - w / 2)
    y = rng.uniform(h / 2, ch - h / 2)
    incr.move_macro(m, x, y)
    incr.undo_move()

# warm up
for _ in range(5):
    one_move()

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
for _ in range(N_MOVES):
    one_move()
pr.disable()
elapsed = time.time() - t0

ps = pstats.Stats(pr).sort_stats("cumulative")
print(f"\n[profile] {N_MOVES} moves in {elapsed:.2f}s → {elapsed / N_MOVES * 1000:.2f} ms/move\n")
ps.print_stats(25)
