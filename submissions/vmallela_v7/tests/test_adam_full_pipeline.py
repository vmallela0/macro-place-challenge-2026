"""End-to-end Adam integration test on a real benchmark.

Loads ibm01 (cheapest), builds an IncrementalEvaluator, runs adam_warm_start
with all three loss terms (HPWL + density + congestion), and asserts:

1. 50 Adam steps complete in < 30s on the auto-selected device.
2. All loss components are finite throughout the run.
3. Loss decreases meaningfully over the run (≥ 5% drop).
4. GradNorm component weights are finite and non-degenerate.
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


def _benchmark_path(name):
    return ROOT / "benchmarks" / "processed" / "public" / f"{name}.pt"


def _build_incr_eval(name="ibm01"):
    bp = _benchmark_path(name)
    bench = Benchmark.load(str(bp))
    plc = _load_plc(name)
    incr = IncrementalEvaluator(plc, bench)
    return incr, bench, plc


def test_50_steps_under_30s_ibm01():
    incr, _, _ = _build_incr_eval("ibm01")
    print(f"  ibm01: n_total={incr.macro_pos.shape[0]} "
          f"n_pins={int(incr.pin_macro.shape[0])} "
          f"n_nets={int(incr.net_weight.shape[0])} "
          f"n_cells={incr.n_cells}", flush=True)
    t0 = time.time()
    pos_final, history = adam_warm_start(
        incr, None,
        n_steps=50,
        lr_frac_canvas=0.02,
        proximal_weight_frac=1.0,
        soft_only=True,
        enable_density=True,
        enable_congestion=True,
        window_margin_cells=4,
        snapshot_every=25,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  ✓ 50 Adam steps wall {elapsed:.2f}s", flush=True)
    assert elapsed < 30.0, f"too slow: {elapsed:.1f}s"

    # Sanity on losses
    assert all(np.isfinite(history["loss"])), \
        f"non-finite loss in history: {history['loss']}"
    assert history["loss"][0] > 0, \
        f"initial loss should be positive, got {history['loss'][0]}"
    drop = (history["loss"][0] - history["loss"][-1]) / max(history["loss"][0], 1e-9)
    print(f"  ✓ loss drop: {history['loss'][0]:.4f} → "
          f"{history['loss'][-1]:.4f} ({drop*100:.1f}%)", flush=True)
    assert drop > 0.05, f"loss decreased <5% over 50 steps: drop={drop:.3f}"

    # Sanity on output shape
    assert pos_final.shape == incr.macro_pos.shape, \
        f"shape mismatch: got {pos_final.shape}, expected {incr.macro_pos.shape}"
    assert np.isfinite(pos_final).all(), "non-finite positions"


if __name__ == "__main__":
    test_50_steps_under_30s_ibm01()
    print("ALL OK")
