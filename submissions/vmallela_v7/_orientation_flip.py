"""Klein-4 orientation flip — Tier 2 sidecar.

For each hard macro, pick one of the 4 axis-aligned orientations
(R0, MY, R180, MX in OpenROAD naming = N, FN, S, FS in DEF naming)
that minimizes the macro's contribution to the HPWL of nets it
participates in. Klein-4 keeps the macro's bounding box invariant
(no R90/R270 rotation), so the placement is unchanged from the
perspective of the Tier 1 proxy / TILOS evaluator, which uses the
default orientation regardless.

The output is a sidecar `*.orientations.pt` next to the `.npy`
placement, consumed by the TCL generator at Tier 2 (OpenROAD WNS/
TNS/Area). Sidecar strings use OpenROAD `place_macro -orientation`
syntax (R0/MY/R180/MX) so the TCL writer can pass them through.

Math
----
For a pin with macro-local offset (px, py):
    R0:   ( px,  py)
    MY:   (-px,  py)   # mirror about Y axis (DEF: FN)
    R180: (-px, -py)
    MX:   ( px, -py)   # mirror about X axis (DEF: FS)

For each net, HPWL = (max_x - min_x) + (max_y - min_y) over the pin
absolute positions. Per-macro greedy: for each hard macro h and each
candidate orientation, recompute the HPWL of nets touching h using h's
transformed pin offsets and pick the min.

Order matters slightly (Gauss-Seidel). A single pass over macros
captures the bulk of the lift; we offer a `n_passes` knob.
"""
from __future__ import annotations
from collections import defaultdict
import time
import numpy as np


ORIENTATIONS = ("R0", "MY", "R180", "MX")
# Sign multipliers for (px, py) under each Klein-4 element. Names use
# OpenROAD `place_macro -orientation` syntax.
_FLIPS = {
    "R0":   (+1.0, +1.0),
    "MY":   (-1.0, +1.0),   # DEF: FN
    "R180": (-1.0, -1.0),
    "MX":   (+1.0, -1.0),   # DEF: FS
}


def _build_pin_groups(pin_macro: np.ndarray, n_hard: int):
    """Return list[ndarray]: pin indices owned by each hard macro."""
    groups = [[] for _ in range(n_hard)]
    for pin_idx, m in enumerate(pin_macro):
        if 0 <= m < n_hard:
            groups[int(m)].append(pin_idx)
    return [np.asarray(g, dtype=np.int64) for g in groups]


def _build_macro_to_nets(pin_groups, pin_to_net: np.ndarray, n_hard: int):
    """For each hard macro: dict net_id -> list of pin indices in that net
    owned by this macro. Used to recompute incident-net HPWL quickly."""
    macro_nets = []
    for h in range(n_hard):
        d = defaultdict(list)
        for p in pin_groups[h]:
            d[int(pin_to_net[p])].append(int(p))
        macro_nets.append(dict(d))
    return macro_nets


def _net_pin_table(pin_to_net: np.ndarray, n_nets: int):
    """net_id -> list of pin indices."""
    table = [[] for _ in range(n_nets)]
    for p, n in enumerate(pin_to_net):
        table[int(n)].append(int(p))
    return [np.asarray(t, dtype=np.int64) for t in table]


def klein4_orient(
    macro_pos: np.ndarray,         # (n_total, 2)
    macro_w: np.ndarray,           # (n_total,) unused but documents intent
    macro_h: np.ndarray,
    pin_macro: np.ndarray,         # (n_pins,) int, -1 for ports
    pin_xoff: np.ndarray,          # (n_pins,) macro-local x offset
    pin_yoff: np.ndarray,
    pin_to_net: np.ndarray,        # (n_pins,) net id per pin
    net_weight: np.ndarray,        # (n_nets,)
    n_hard: int,
    n_nets: int,
    *,
    n_passes: int = 2,
    verbose: bool = False,
) -> tuple[list[str], dict]:
    """Return per-hard-macro orientation choice and diagnostics.

    Per-macro greedy over Klein-4. `n_passes` Gauss-Seidel passes.
    Returns (orientations, info).
    """
    t0 = time.time()
    pin_groups = _build_pin_groups(pin_macro, n_hard)
    macro_nets = _build_macro_to_nets(pin_groups, pin_to_net, n_hard)
    net_pins = _net_pin_table(pin_to_net, n_nets)

    # Working copies of pin absolute positions; we mutate per macro.
    is_port = (pin_macro < 0)
    abs_x = np.where(is_port, pin_xoff,
                     macro_pos[np.maximum(pin_macro, 0), 0] + pin_xoff)
    abs_y = np.where(is_port, pin_yoff,
                     macro_pos[np.maximum(pin_macro, 0), 1] + pin_yoff)

    orient = ["R0"] * n_hard
    # Cache per-net per-component (min, max) so we can re-pick orientation
    # for one macro by re-evaluating only that macro's net set.
    initial_hpwl = _hpwl_total(abs_x, abs_y, net_pins, net_weight)

    for ip in range(n_passes):
        moved = 0
        for h in range(n_hard):
            inc_nets = macro_nets[h]
            if not inc_nets:
                continue
            # Save current pin positions for this macro's pins
            grp = pin_groups[h]
            base_x = pin_xoff[grp].copy()
            base_y = pin_yoff[grp].copy()
            best_lab = orient[h]
            best_cost = _net_subset_hpwl(
                abs_x, abs_y, inc_nets, net_pins, net_weight)
            best_x = abs_x[grp].copy()
            best_y = abs_y[grp].copy()

            for lab in ORIENTATIONS:
                if lab == best_lab:
                    continue
                sx, sy = _FLIPS[lab]
                cx = macro_pos[h, 0]
                cy = macro_pos[h, 1]
                abs_x[grp] = cx + sx * base_x
                abs_y[grp] = cy + sy * base_y
                cost = _net_subset_hpwl(
                    abs_x, abs_y, inc_nets, net_pins, net_weight)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_lab = lab
                    best_x = abs_x[grp].copy()
                    best_y = abs_y[grp].copy()
            # Commit best
            abs_x[grp] = best_x
            abs_y[grp] = best_y
            if best_lab != orient[h]:
                orient[h] = best_lab
                moved += 1
        if verbose:
            print(f"    [orient] pass {ip}: {moved}/{n_hard} macros flipped",
                  flush=True)
        if moved == 0:
            break

    final_hpwl = _hpwl_total(abs_x, abs_y, net_pins, net_weight)
    info = {
        "initial_hpwl": float(initial_hpwl),
        "final_hpwl": float(final_hpwl),
        "delta_hpwl": float(initial_hpwl - final_hpwl),
        "n_flipped": sum(1 for o in orient if o != "R0"),
        "n_hard": n_hard,
        "wall_s": time.time() - t0,
        "counts": {lab: orient.count(lab) for lab in ORIENTATIONS},
    }
    return orient, info


def _net_subset_hpwl(abs_x, abs_y, inc_nets, net_pins, net_weight):
    total = 0.0
    for nid, _pins_owned in inc_nets.items():
        pins = net_pins[nid]
        if pins.size == 0:
            continue
        xs = abs_x[pins]; ys = abs_y[pins]
        w = float(net_weight[nid])
        total += w * ((xs.max() - xs.min()) + (ys.max() - ys.min()))
    return total


def _hpwl_total(abs_x, abs_y, net_pins, net_weight):
    total = 0.0
    for nid, pins in enumerate(net_pins):
        if pins.size == 0:
            continue
        xs = abs_x[pins]; ys = abs_y[pins]
        total += float(net_weight[nid]) * (
            (xs.max() - xs.min()) + (ys.max() - ys.min()))
    return total


def save_orientation_sidecar(orientations: list[str], path: str):
    """Save sidecar at path. Format: torch.save'd dict.

    Schema:
        {"hard_macro_orientations": ["N", "FN", ...],
         "version": 1,
         "ordering": "hard_macro_index_in_benchmark"}
    """
    import torch as _torch
    _torch.save({
        "hard_macro_orientations": list(orientations),
        "version": 1,
        "ordering": "hard_macro_index_in_benchmark",
    }, path)
