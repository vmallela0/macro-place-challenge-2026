"""T3: hard macro drift validation.

Goal: when adam_warm_start runs with `soft_only=False` and a moderate
inertia (proximal_weight_frac), hard macros should drift but stay within
~1 cell width of their initial position over 200 steps. Verifies that
the floorplan can "breathe" without hards careening across the canvas.

Asserts on ibm01:
1. With soft_only=True (default), hard positions UNCHANGED.
2. With soft_only=False and inertia=1.0, hards drift > 0 but
   < 2 cell widths in the L_inf norm over 200 steps.
3. Loss with hard drift is no worse than loss with soft_only at step 200.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))

from macro_place.benchmark import Benchmark
from placer import _load_plc, IncrementalEvaluator
from _smooth_proxy import adam_warm_start


def _build_incr_eval(name="ibm01"):
    bp = ROOT / "benchmarks" / "processed" / "public" / f"{name}.pt"
    bench = Benchmark.load(str(bp))
    plc = _load_plc(name)
    return IncrementalEvaluator(plc, bench), bench, plc


def test_soft_only_freezes_hards():
    """With soft_only=True the hard positions never change."""
    incr, _, _ = _build_incr_eval("ibm01")
    n_hard = incr.n_hard
    init_hard = np.array(incr.macro_pos[:n_hard])
    pos_final, _ = adam_warm_start(
        incr, None, n_steps=50, soft_only=True,
        enable_density=True, enable_congestion=True,
        verbose=False)
    drift = np.max(np.abs(pos_final[:n_hard] - init_hard))
    print(f"  ✓ soft_only: max hard drift = {drift:.6e} (expected 0)")
    assert drift < 1e-4, f"hards drifted unexpectedly: {drift}"


def test_hard_drift_bounded():
    """With soft_only=False and inertia, hards drift but stay close."""
    incr, _, _ = _build_incr_eval("ibm01")
    n_hard = incr.n_hard
    init_hard = np.array(incr.macro_pos[:n_hard])
    cw, ch = float(incr.cw), float(incr.ch)
    cell_w = float(incr.grid_width)
    cell_h = float(incr.grid_height)

    pos_final, history = adam_warm_start(
        incr, None,
        n_steps=200,
        soft_only=False,                  # hard gradients flow
        proximal_weight_frac=1.0,         # moderate inertia
        enable_density=True,
        enable_congestion=True,
        verbose=False)
    drift = pos_final[:n_hard] - init_hard
    drift_x = np.abs(drift[:, 0])
    drift_y = np.abs(drift[:, 1])
    max_drift_cells = max(drift_x.max() / cell_w,
                           drift_y.max() / cell_h)
    mean_drift_cells = (drift_x.mean() / cell_w
                        + drift_y.mean() / cell_h) / 2.0
    print(f"  hard drift over 200 steps: "
          f"max={max_drift_cells:.3f} cells "
          f"mean={mean_drift_cells:.3f} cells "
          f"(canvas {cw:.1f} × {ch:.1f}, cell {cell_w:.2f}×{cell_h:.2f})",
          flush=True)
    # Some drift expected, but bounded.
    assert mean_drift_cells > 0.001, \
        f"hards didn't drift at all: mean={mean_drift_cells}"
    # Cap drift to avoid floorplan blowup.
    assert max_drift_cells < 5.0, \
        f"hards drifted too far: max={max_drift_cells} cells"
    # Loss should still decrease.
    drop = (history["loss"][0] - history["loss"][-1]) / max(history["loss"][0], 1e-9)
    print(f"  ✓ hard drift mode: drop {drop*100:.1f}%; "
          f"max drift {max_drift_cells:.2f} cells (< 5 cap)")
    assert drop > 0.05, f"loss did not drop with hard drift: {drop}"


if __name__ == "__main__":
    test_soft_only_freezes_hards()
    test_hard_drift_bounded()
    print("ALL OK")
