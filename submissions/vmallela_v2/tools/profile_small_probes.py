"""Profile with CD-like small-delta probes to measure v5's skip-path benefit.

Each probe picks a random macro and a random CD-schedule delta, moves the
macro by (delta*dir_x, delta*dir_y) where dir is a unit direction. Most of
these probes keep the macro's pins in the same grid cells → v5 should skip
routing updates.
"""
import cProfile
import pstats
import sys
import time
import random
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_vmallela_placer", str(_ROOT / "submissions" / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

IncrementalEvaluator = _mod.IncrementalEvaluator
_load_plc = _mod._load_plc

BENCH = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
N_MOVES = int(sys.argv[2]) if len(sys.argv) > 2 else 500

benchmark = Benchmark.load(str(_ROOT / "benchmarks" / "processed" / "public" / f"{BENCH}.pt"))
plc = _load_plc(BENCH)
incr = IncrementalEvaluator(plc, benchmark)
print(f"[profile-sp] {BENCH}: n_hard={benchmark.num_hard_macros} grid={incr.grid_col}x{incr.grid_row} "
      f"cell={incr.grid_width:.3f}x{incr.grid_height:.3f}")

sizes_np = benchmark.macro_sizes[:benchmark.num_hard_macros].numpy()
cw = float(benchmark.canvas_width)
ch = float(benchmark.canvas_height)

# CD-like deltas (matches _coord_descent schedule).
DELTAS = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]

rng = random.Random(42)
n_skip = 0
n_full = 0

def one_move():
    global n_skip, n_full
    m = rng.randrange(benchmark.num_hard_macros)
    d = rng.choice(DELTAS)
    dx, dy = rng.choice(DIRS)
    w, h = sizes_np[m]
    ox = float(incr.macro_pos[m, 0])
    oy = float(incr.macro_pos[m, 1])
    nx = min(max(ox + d * dx, w / 2), cw - w / 2)
    ny = min(max(oy + d * dy, h / 2), ch - h / 2)
    incr.move_macro(m, nx, ny)
    # Inspect undo state to count skips
    if incr._undo.get('cells_changed', True):
        n_full += 1
    else:
        n_skip += 1
    incr.undo_move()

# warm-up
for _ in range(5):
    one_move()
n_skip = 0; n_full = 0

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
for _ in range(N_MOVES):
    one_move()
pr.disable()
elapsed = time.time() - t0

print(f"\n[profile-sp] {N_MOVES} small-delta probes in {elapsed:.2f}s "
      f"→ {elapsed / N_MOVES * 1000:.3f} ms/probe")
print(f"[profile-sp] skip path hits: {n_skip}/{N_MOVES} ({100*n_skip/N_MOVES:.1f}%)")
print(f"[profile-sp] full path hits: {n_full}/{N_MOVES} ({100*n_full/N_MOVES:.1f}%)")

ps = pstats.Stats(pr).sort_stats("cumulative")
ps.print_stats(10)
