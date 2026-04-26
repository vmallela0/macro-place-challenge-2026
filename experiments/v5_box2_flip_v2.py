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
        # sinks may be dict {macro_name: [pin_names]} or {pin_idx: weight}.
        sink_indices = []
        for k, v in sinks.items():
            # heuristic: if v is a list of pin names, look up pin indices.
            if isinstance(v, (list, tuple)):
                # k is macro name, v is list of pin names
                for pname in v:
                    full = f"{k}/{pname}" if k else pname
                    # search for matching pin name
                    for pi, pp in enumerate(plc.modules_w_pins):
                        if hasattr(pp, "get_name"):
                            n = pp.get_name()
                            if n == full or n == pname:
                                sink_indices.append(pi)
                                break
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

    initial_proxy = proxy_cost(plc)
    initial_wl = plc.get_cost()
    print(f"[{bench_name}] initial: proxy={initial_proxy:.4f}  wl(plc)={initial_wl:.4f}")

    hard_indices = list(plc.hard_macro_indices)

    # Initial orientation map.
    orient_map = {idx: plc.get_macro_orientation(idx) for idx in hard_indices}

    # Verify our HPWL matches plc's initial wl roughly (sanity check).
    init_hpwl = compute_total_hpwl(plc, pin_macro_idx, pin_offset, nets, orient_map)
    print(f"[{bench_name}] our_hpwl={init_hpwl:.4f}  (plc_wl={initial_wl:.4f}, ratio={init_hpwl/initial_wl if initial_wl>0 else 0:.3f})")

    # Greedy flip per macro.
    total_changes = 0
    for pass_num in range(max_passes):
        pass_changes = 0
        for idx in hard_indices:
            macro_nets = macro_to_nets.get(idx, [])
            if not macro_nets:
                continue
            cur = orient_map[idx]
            best_orient = cur
            best_hpwl = compute_macro_hpwl(plc, pin_macro_idx, pin_offset, nets, macro_nets, orient_map)
            for o in KLEIN_4:
                if o == cur:
                    continue
                orient_map[idx] = o
                h = compute_macro_hpwl(plc, pin_macro_idx, pin_offset, nets, macro_nets, orient_map)
                if h < best_hpwl - 1e-9:
                    best_hpwl = h
                    best_orient = o
            orient_map[idx] = best_orient
            if best_orient != cur:
                pass_changes += 1
        total_changes += pass_changes
        cur_total = compute_total_hpwl(plc, pin_macro_idx, pin_offset, nets, orient_map)
        print(f"[{bench_name}] pass {pass_num+1}: changes={pass_changes}  our_hpwl={cur_total:.4f}")
        if pass_changes == 0:
            break

    # Apply final orientations to plc.
    for idx, o in orient_map.items():
        plc.update_macro_orientation(idx, o)

    final_proxy = proxy_cost(plc)
    final_wl = plc.get_cost()
    print(f"[{bench_name}] FINAL: proxy={final_proxy:.4f}  wl(plc)={final_wl:.4f}  Δproxy={initial_proxy-final_proxy:+.4f} ({(initial_proxy-final_proxy)/initial_proxy*100:+.2f}%)")
    return plc, initial_proxy, final_proxy, total_changes


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: flip_v2.py <bench> <plc_file> [<output_plc>]")
        sys.exit(2)
    bench, plc_file = sys.argv[1], sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    plc, initial, final, changes = optimize(bench, plc_file)
    if output:
        plc.save_placement(output)
        print(f"saved to {output}")
