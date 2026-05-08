"""Standalone smoke test: does Phase 0 homotopy produce a BETTER warm-start
than .plc init?

For ibm06: load bench + .plc init, build IncrementalEvaluator, capture
its proxy cost (the actual leaderboard formula). Then run Phase 0
homotopy spreader, write back the positions to incr, recompute. Compare.

If Phase 0's output proxy < .plc init's proxy: Phase 0 is producing a
better warm-start, worth integrating.
If ≥: Phase 0 spreader is broken or harmful, abandon.

Pure read-only of .plc + cost compute. Fast (no SA, no v4, no Hessian).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
sys.path.insert(0, str(ROOT))

import importlib.util as _ilu
import numpy as np
import torch  # noqa: F401

from macro_place.benchmark import Benchmark

v1_spec = _ilu.spec_from_file_location(
    "v1_test", str(ROOT / "submissions" / "vmallela" / "placer.py"))
v1 = _ilu.module_from_spec(v1_spec); v1_spec.loader.exec_module(v1)

from _phase0_electrostatic import electrostatic_spread_homotopy


def cost_summary(incr) -> tuple[float, float, float, float]:
    """Returns (proxy, wl, dens, cong). Proxy = wl + dens + cong."""
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()
    wl = float(incr.wirelength_cost)
    dens = float(incr.density_cost)
    cong = float(incr.congestion_cost)
    return (wl + dens + cong, wl, dens, cong)


def run(bench_name: str, *,
        n_stages: int, n_iters: int,
        lambda_0: float, lambda_f: float,
        lr_frac: float, init_from_plc: bool):
    bench_path = ROOT / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"
    bench = Benchmark.load(str(bench_path))

    plc = v1._load_plc(bench_name)
    incr = v1.IncrementalEvaluator(plc, bench)

    t0 = time.time()
    p0_proxy, p0_wl, p0_dens, p0_cong = cost_summary(incr)
    print(f"\n=== {bench_name} ===")
    print(f"baseline (.plc init):     "
          f"proxy={p0_proxy:.4f} wl={p0_wl:.4f} "
          f"dens={p0_dens:.4f} cong={p0_cong:.4f}")
    print(f"  measure took {time.time()-t0:.1f}s")

    t1 = time.time()
    new_pos = electrostatic_spread_homotopy(
        bench, incr,
        n_iters=n_iters,
        n_stages=n_stages,
        lr_frac_canvas=lr_frac,
        lambda_0=lambda_0,
        lambda_f=lambda_f,
        init_from_plc=init_from_plc,
        soft_only=False,
        verbose=True)
    print(f"  phase0 took {time.time()-t1:.1f}s; "
          f"applying & re-measuring...")

    incr.macro_pos[:] = new_pos
    incr._recompute_pin_positions()
    p1_proxy, p1_wl, p1_dens, p1_cong = cost_summary(incr)
    print(f"phase0 homotopy out:      "
          f"proxy={p1_proxy:.4f} wl={p1_wl:.4f} "
          f"dens={p1_dens:.4f} cong={p1_cong:.4f}")
    delta = p1_proxy - p0_proxy
    print(f"  Δproxy = {delta:+.4f}  "
          f"({'improvement!' if delta < 0 else 'REGRESSION'})")


if __name__ == "__main__":
    run("ibm06",
        n_stages=20, n_iters=500,
        lambda_0=0.05, lambda_f=2.0,    # CVaR-scale ~1; keep λ near unity
        lr_frac=0.001,                   # 5x smaller — avoid overshoot
        init_from_plc=True)
