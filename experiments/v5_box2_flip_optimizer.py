#!/usr/bin/env python3
"""Greedy per-macro Klein-4 orientation optimizer.

Loads a benchmark + placement file, then for each hard macro tries the 4
Klein-4 orientations (N, FN, FS, S) and picks the one that minimizes proxy
cost. Uses the same PlacementCost API the placer uses, so cost values are
directly comparable to placer output.

Usage:
  python experiments/v5_box2_flip_optimizer.py <bench> <plc_file> [<output_plc>]

If <output_plc> is omitted, only prints the cost delta without writing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from macro_place._plc import PlacementCost

KLEIN_4 = ("N", "FN", "FS", "S")


def proxy_cost(plc, w_wl=1.0, w_den=0.5, w_cong=0.5):
    return (
        w_wl * plc.get_cost()
        + w_den * plc.get_density_cost()
        + w_cong * plc.get_congestion_cost()
    )


def hpwl_only_cost(plc):
    """Inner-loop cost: just wirelength. Orientation flips don't change density
    (rectangles are same when flipped about center) and only marginally change
    congestion, so HPWL is a tight upper-bound proxy. Much faster than full
    proxy_cost — single get_cost() call vs three."""
    return plc.get_cost()


def optimize_orientations(bench_name: str, plc_file: str, max_passes: int = 2):
    repo = Path(__file__).resolve().parent.parent
    netlist = repo / "external/MacroPlacement/Testcases/ICCAD04" / bench_name / "netlist.pb.txt"
    plc = PlacementCost(str(netlist))
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    # Enable incremental cost — should make get_cost() ~10-100x faster after
    # local changes like orientation flips.
    try:
        plc.set_use_incremental_cost(True)
    except Exception:
        pass

    hard_indices = plc.hard_macro_indices
    n_hard = len(hard_indices)

    initial = proxy_cost(plc)
    initial_wl = plc.get_cost()
    initial_den = plc.get_density_cost()
    initial_cong = plc.get_congestion_cost()
    print(f"[{bench_name}] n_hard={n_hard}")
    print(f"[{bench_name}] initial proxy={initial:.4f}  wl={initial_wl:.4f} den={initial_den:.4f} cong={initial_cong:.4f}")

    t0 = time.time()
    total_changes = 0
    for pass_num in range(max_passes):
        pass_changes = 0
        for idx in hard_indices:
            cur_orient = plc.get_macro_orientation(idx)
            best_orient = cur_orient
            best_cost = hpwl_only_cost(plc)
            for o in KLEIN_4:
                if o == cur_orient:
                    continue
                plc.update_macro_orientation(idx, o)
                c = hpwl_only_cost(plc)
                if c < best_cost - 1e-6:
                    best_cost = c
                    best_orient = o
            if best_orient != cur_orient:
                plc.update_macro_orientation(idx, best_orient)
                pass_changes += 1
            else:
                plc.update_macro_orientation(idx, cur_orient)
        total_changes += pass_changes
        cur = proxy_cost(plc)
        print(f"[{bench_name}] pass {pass_num+1}: changes={pass_changes}  proxy={cur:.4f}  Δ={initial-cur:+.4f}")
        if pass_changes == 0:
            break

    final = proxy_cost(plc)
    final_wl = plc.get_cost()
    final_den = plc.get_density_cost()
    final_cong = plc.get_congestion_cost()
    elapsed = time.time() - t0
    print(f"[{bench_name}] FINAL proxy={final:.4f}  wl={final_wl:.4f} den={final_den:.4f} cong={final_cong:.4f}")
    print(f"[{bench_name}] reduction: {initial-final:+.4f} ({(initial-final)/initial*100:+.2f}%)  passes_changes={total_changes}  wall={elapsed:.1f}s")
    return plc, initial, final


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: flip_optimizer.py <bench> <plc_file> [<output_plc>]")
        sys.exit(2)
    bench, plc_file = sys.argv[1], sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    plc, initial, final = optimize_orientations(bench, plc_file)
    if output:
        plc.save_placement(output)
        print(f"saved optimized placement to {output}")
