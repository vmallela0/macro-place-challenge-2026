"""End-to-end smoke for orientation sidecar pipeline.

Loads ibm01 benchmark, runs Klein-4 greedy on the .plc-init placement
(default) directly (no v4 pipeline — just verifies the integration).
Writes orientations.pt sidecar. Then verifies:
  1. Sidecar exists and round-trips cleanly via torch.load
  2. All entries are valid OpenROAD orientation strings
  3. HPWL strictly does not increase
  4. The TCL writer's _load_orientation_sidecar picks it up

Run:  .venv/bin/python submissions/vmallela_v7/tests/smoke_orientation_e2e.py
"""
import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))

from macro_place.benchmark import Benchmark
from _orientation_flip import (
    klein4_orient, save_orientation_sidecar, ORIENTATIONS,
)


def main():
    bench_path = ROOT / "benchmarks" / "processed" / "public" / "ibm01.pt"
    print(f"loading benchmark from {bench_path}")
    bench = Benchmark.load(str(bench_path))
    print(f"  {bench}")
    print(f"  num_hard_macros={bench.num_hard_macros}")
    print(f"  num_nets={bench.num_nets}")
    print(f"  pin_offset_lists={len(bench.macro_pin_offsets)}")

    # Load via placer.IncrementalEvaluator to get the full pin/net flat
    # arrays the orientation flip needs. Mirrors what placer.py does.
    import importlib.util as ilu
    v1_spec = ilu.spec_from_file_location(
        "_v1_smoke", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    print(f"  IncrementalEvaluator: pin_macro shape={incr.pin_macro.shape}, "
          f"net_starts shape={incr.net_starts.shape}")

    pin_macro = np.asarray(incr.pin_macro)
    pin_xoff = np.asarray(incr.pin_xoff)
    pin_yoff = np.asarray(incr.pin_yoff)
    net_starts = np.asarray(incr.net_starts)
    net_weight = np.asarray(incr.net_weight)
    macro_pos = np.asarray(incr.macro_pos)
    n_hard = bench.num_hard_macros
    n_nets = int(net_starts.shape[0] - 1)

    pin_to_net = np.zeros(int(pin_macro.shape[0]), dtype=np.int64)
    for nid in range(n_nets):
        pin_to_net[net_starts[nid]:net_starts[nid + 1]] = nid

    print("running klein4_orient...")
    t0 = time.time()
    orientations, info = klein4_orient(
        macro_pos=macro_pos,
        macro_w=np.asarray(incr.macro_w),
        macro_h=np.asarray(incr.macro_h),
        pin_macro=pin_macro,
        pin_xoff=pin_xoff,
        pin_yoff=pin_yoff,
        pin_to_net=pin_to_net,
        net_weight=net_weight,
        n_hard=n_hard,
        n_nets=n_nets,
        n_passes=2,
        verbose=True,
    )
    wall = time.time() - t0
    print(f"  done in {wall:.2f}s")
    print(f"  initial HPWL: {info['initial_hpwl']:.1f}")
    print(f"  final HPWL:   {info['final_hpwl']:.1f}")
    print(f"  Δ HPWL:       {info['delta_hpwl']:+.1f}")
    print(f"  flipped:      {info['n_flipped']}/{info['n_hard']}")
    print(f"  counts:       {info['counts']}")

    assert len(orientations) == n_hard
    assert all(o in ORIENTATIONS for o in orientations), \
        "all entries must be valid OpenROAD orientations"
    assert info["final_hpwl"] <= info["initial_hpwl"] + 1e-6, \
        "greedy must not increase HPWL"

    # Save sidecar in a temp location and verify the TCL writer's loader picks it up.
    with tempfile.TemporaryDirectory() as td:
        placement_path = str(Path(td) / "ibm01.npy")
        sidecar_path = placement_path + ".orientations.pt"
        np.save(placement_path, macro_pos)
        save_orientation_sidecar(orientations, sidecar_path)
        assert Path(sidecar_path).exists()
        roundtrip = torch.load(sidecar_path, weights_only=False)
        assert roundtrip["hard_macro_orientations"] == orientations
        assert roundtrip["version"] == 1

        # Now test the TCL writer's loader.
        sys.path.insert(0, str(ROOT / "scripts"))
        import generate_macro_placement_tcl as gmt
        loaded = gmt._load_orientation_sidecar(placement_path)
        assert loaded == orientations, \
            "TCL writer must round-trip the same list"
    print("✓ sidecar round-trip + TCL loader OK")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
