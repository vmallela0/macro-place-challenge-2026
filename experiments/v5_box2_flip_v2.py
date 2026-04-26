#!/usr/bin/env python3
"""Pure-Python orientation flip optimizer.

Bypasses PlacementCost's slow get_cost() entirely. Builds an in-memory net
graph from plc.modules_w_pins[].get_sink() and computes HPWL directly.
Each macro flip only re-evaluates nets touching that macro → orders of
magnitude faster than going through PlacementCost.

Usage:
  python experiments/v5_box2_flip_v2.py <bench> <plc_file> [<output_plc>]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from macro_place._plc import PlacementCost

KLEIN_4 = ("N", "FN", "FS", "S")


def rotate_offset(off_x, off_y, orient):
    if orient == "N":
        return off_x, off_y
    if orient == "FN":
        return -off_x, off_y
    if orient == "FS":
        return off_x, -off_y
    if orient == "S":
        return -off_x, -off_y
    return off_x, off_y  # default


def build_net_index(plc):
    """Return:
       - pin_macro_idx[pin_idx]: macro node_index that owns this pin (or -1 for ports)
       - pin_offset[pin_idx]: (off_x, off_y) of pin within its macro
       - nets: list of (source_pin_idx, [sink_pin_indices])
       - macro_to_nets[macro_idx]: list of net_idx where this macro has any pin
    """
    n_pins = len(plc.modules_w_pins)
    pin_macro_idx = [-1] * n_pins
    pin_offset = [(0.0, 0.0)] * n_pins
    pin_macro_name = [None] * n_pins

    name_to_macro_idx = {}
    for idx in plc.get_macro_indices():
        name_to_macro_idx[plc.get_node_name(idx)] = idx

    for pin_idx in range(n_pins):
        p = plc.modules_w_pins[pin_idx]
        if hasattr(p, "x_offset"):
            pin_offset[pin_idx] = (float(p.x_offset), float(p.y_offset))
        macro_name = None
        if hasattr(p, "get_macro_name"):
            try:
                macro_name = p.get_macro_name()
            except Exception:
                macro_name = None
        if macro_name and macro_name in name_to_macro_idx:
            pin_macro_idx[pin_idx] = name_to_macro_idx[macro_name]
            pin_macro_name[pin_idx] = macro_name

    # Build a pin-name → pin-index map ONCE so the net build is O(pins) not O(pins²).
    name_to_pin_idx = {}
    fullname_to_pin_idx = {}
    for pi in range(n_pins):
        pp = plc.modules_w_pins[pi]
        if hasattr(pp, "get_name"):
            try:
                n = pp.get_name()
            except Exception:
                n = None
            if n:
                name_to_pin_idx[n] = pi
                if pin_macro_name[pi]:
                    fullname_to_pin_idx[f"{pin_macro_name[pi]}/{n}"] = pi

    # Build nets from source-pin sink dicts.
    nets = []
    for src in range(n_pins):
        p = plc.modules_w_pins[src]
        if not hasattr(p, "get_sink"):
            continue
        try:
            sinks = p.get_sink()
        except Exception:
            sinks = {}
        if not sinks:
            continue
        sink_indices = []
        for k, v in sinks.items():
            if isinstance(v, (list, tuple)):
                # k = macro name, v = list of pin names within that macro
                for pname in v:
                    full = f"{k}/{pname}" if k else pname
                    pi = fullname_to_pin_idx.get(full) or name_to_pin_idx.get(pname) or name_to_pin_idx.get(full)
                    if pi is not None:
                        sink_indices.append(pi)
            elif isinstance(k, int):
                sink_indices.append(k)
        if sink_indices:
            nets.append((src, sink_indices))

    # macro_to_nets
    macro_to_nets = {}
    for net_idx, (src, sinks) in enumerate(nets):
        macros = set()
        if pin_macro_idx[src] >= 0:
            macros.add(pin_macro_idx[src])
        for s in sinks:
            if pin_macro_idx[s] >= 0:
                macros.add(pin_macro_idx[s])
        for m in macros:
            macro_to_nets.setdefault(m, []).append(net_idx)
    return pin_macro_idx, pin_offset, nets, macro_to_nets


def compute_pin_pos(pin_idx, plc, pin_macro_idx, pin_offset, orient_map):
    macro_idx = pin_macro_idx[pin_idx]
    if macro_idx < 0:
        # port — fixed location at the pin's own coordinates
        p = plc.modules_w_pins[pin_idx]
        if hasattr(p, "get_pos"):
            try:
                pos = p.get_pos()
                return pos[0], pos[1]
            except Exception:
                pass
        # fallback
        return 0.0, 0.0
    cx, cy = plc.get_node_location(macro_idx)
    ox, oy = pin_offset[pin_idx]
    rx, ry = rotate_offset(ox, oy, orient_map.get(macro_idx, "N"))
    return cx + rx, cy + ry


def compute_total_hpwl(plc, pin_macro_idx, pin_offset, nets, orient_map):
    total = 0.0
    for src, sinks in nets:
        sx, sy = compute_pin_pos(src, plc, pin_macro_idx, pin_offset, orient_map)
        min_x = max_x = sx
        min_y = max_y = sy
        for s in sinks:
            x, y = compute_pin_pos(s, plc, pin_macro_idx, pin_offset, orient_map)
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        total += (max_x - min_x) + (max_y - min_y)
    return total


def compute_macro_hpwl(plc, pin_macro_idx, pin_offset, nets, macro_nets, orient_map):
    """HPWL summed over only the nets touching macro_idx."""
    total = 0.0
    for net_idx in macro_nets:
        src, sinks = nets[net_idx]
        sx, sy = compute_pin_pos(src, plc, pin_macro_idx, pin_offset, orient_map)
        min_x = max_x = sx
        min_y = max_y = sy
        for s in sinks:
            x, y = compute_pin_pos(s, plc, pin_macro_idx, pin_offset, orient_map)
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        total += (max_x - min_x) + (max_y - min_y)
    return total


def proxy_cost(plc):
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


def optimize(bench_name: str, plc_file: str, max_passes: int = 3):
    repo = Path(__file__).resolve().parent.parent
    netlist = repo / "external/MacroPlacement/Testcases/ICCAD04" / bench_name / "netlist.pb.txt"
    plc = PlacementCost(str(netlist))
    plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    print(f"[{bench_name}] building net index...")
    t0 = time.time()
    pin_macro_idx, pin_offset, nets, macro_to_nets = build_net_index(plc)
    print(f"[{bench_name}] index built: {len(nets)} nets, {sum(1 for m in pin_macro_idx if m>=0)}/{len(pin_macro_idx)} pins on macros, t={time.time()-t0:.1f}s")

    print(f"[{bench_name}] caching macro positions...")
    t1 = time.time()
    macro_positions = {}
    for idx in plc.get_macro_indices():
        macro_positions[idx] = plc.get_node_location(idx)
    print(f"[{bench_name}] cached {len(macro_positions)} positions, t={time.time()-t1:.1f}s")

    # Cache port positions too — they have fixed positions stored on the pin object.
    port_positions = {}
    for pi in range(len(plc.modules_w_pins)):
        if pin_macro_idx[pi] < 0:
            p = plc.modules_w_pins[pi]
            try:
                port_positions[pi] = p.get_pos() if hasattr(p, "get_pos") else (0.0, 0.0)
            except Exception:
                port_positions[pi] = (0.0, 0.0)

    hard_indices = list(plc.hard_macro_indices)
    orient_map = {idx: plc.get_macro_orientation(idx) for idx in hard_indices}

    # Pure-python HPWL using cached positions.
    def pin_pos_cached(pi):
        m = pin_macro_idx[pi]
        if m < 0:
            return port_positions.get(pi, (0.0, 0.0))
        cx, cy = macro_positions[m]
        ox, oy = pin_offset[pi]
        rx, ry = rotate_offset(ox, oy, orient_map.get(m, "N"))
        return cx + rx, cy + ry

    def hpwl_subset(net_idx_list):
        total = 0.0
        for ni in net_idx_list:
            src, sinks = nets[ni]
            sx, sy = pin_pos_cached(src)
            min_x = max_x = sx; min_y = max_y = sy
            for s in sinks:
                x, y = pin_pos_cached(s)
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
            total += (max_x - min_x) + (max_y - min_y)
        return total

    all_net_indices = list(range(len(nets)))
    print(f"[{bench_name}] computing initial HPWL...")
    t2 = time.time()
    initial_hpwl = hpwl_subset(all_net_indices)
    print(f"[{bench_name}] initial_hpwl={initial_hpwl:.4f}  t={time.time()-t2:.1f}s")

    total_changes = 0
    for pass_num in range(max_passes):
        pass_changes = 0
        t3 = time.time()
        for idx in hard_indices:
            macro_nets = macro_to_nets.get(idx, [])
            if not macro_nets:
                continue
            cur = orient_map[idx]
            best_orient = cur
            best_hpwl = hpwl_subset(macro_nets)
            for o in KLEIN_4:
                if o == cur:
                    continue
                orient_map[idx] = o
                h = hpwl_subset(macro_nets)
                if h < best_hpwl - 1e-9:
                    best_hpwl = h
                    best_orient = o
            orient_map[idx] = best_orient
            if best_orient != cur:
                pass_changes += 1
        total_changes += pass_changes
        cur_total = hpwl_subset(all_net_indices)
        print(f"[{bench_name}] pass {pass_num+1}: changes={pass_changes}  hpwl={cur_total:.4f}  Δ={(initial_hpwl-cur_total)/initial_hpwl*100:+.2f}%  t={time.time()-t3:.1f}s")
        if pass_changes == 0:
            break

    final_hpwl = hpwl_subset(all_net_indices)
    print(f"[{bench_name}] hpwl: {initial_hpwl:.4f} → {final_hpwl:.4f} ({(initial_hpwl-final_hpwl)/initial_hpwl*100:+.2f}% reduction)")

    # Apply final orientations to plc only if output requested. Skip proxy_cost
    # recompute by default — it's O(N²) slow on big benches and HPWL reduction
    # is the meaningful metric for orientation flips anyway.
    print(f"[{bench_name}] FINAL: hpwl_pct_reduction={(initial_hpwl-final_hpwl)/initial_hpwl*100:+.2f}% changes={total_changes}")
    return plc, orient_map, initial_hpwl, final_hpwl, total_changes


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: flip_v2.py <bench> <plc_file> [<output_plc>]")
        sys.exit(2)
    bench, plc_file = sys.argv[1], sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    plc, orient_map, initial_hpwl, final_hpwl, changes = optimize(bench, plc_file)
    if output:
        # Apply orientations + save (the slow plc.update + save part).
        import time as _t
        t = _t.time()
        for idx, o in orient_map.items():
            plc.update_macro_orientation(idx, o)
        plc.save_placement(output)
        print(f"saved to {output}  (apply+save took {_t.time()-t:.1f}s)")
