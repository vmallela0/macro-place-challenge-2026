"""Klein-4 orientation flip: greedy per-macro pin-aware HPWL minimization.

Tier 2 (OpenROAD) only — has no effect on Tier 1 proxy because the TILOS
evaluator reads pin offsets from netlist.pb.txt at fixed N orientation and
the rules forbid evaluator changes. The orientation choices land in
`orientations.pt` next to the placer output; the Tier 2 TCL generator
applies them to the plc via `update_macro_orientation` before emitting
OpenROAD `place_macro -orientation`.

Klein-4 mapping (column-0 of macro_pin_offsets is x, column-1 is y):

    o=0  N   ( px,  py)
    o=1  FN  (-px,  py)   x-mirror
    o=2  FS  ( px, -py)   y-mirror
    o=3  S   (-px, -py)   180° rotation

90° rotations (E/FE/W/FW) are explicitly disallowed by the competition rules
(fakeram45 SRAMs aren't designed for it).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch


ORIENT_NAMES = ("N", "FN", "FS", "S")


def _build_macro_to_nets(
    net_pin_nodes: List[torch.Tensor], num_hard: int
) -> List[List[Tuple[int, int]]]:
    """For each hard macro, list (net_idx, pin_slot) it appears on."""
    macro_to_nets: List[List[Tuple[int, int]]] = [[] for _ in range(num_hard)]
    for net_idx, pins in enumerate(net_pin_nodes):
        if pins.numel() == 0:
            continue
        owners = pins[:, 0].tolist()
        slots = pins[:, 1].tolist()
        for owner, slot in zip(owners, slots):
            if 0 <= owner < num_hard:
                macro_to_nets[owner].append((net_idx, slot))
    return macro_to_nets


def _net_pin_xy(
    placement: torch.Tensor,
    benchmark,
    pins: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pin (x,y) for one net under current orientation-N pin offsets.

    pins is the [P, 2] (owner, slot) tensor for this net.
    """
    num_macros = benchmark.num_macros
    num_hard = benchmark.num_hard_macros
    owners = pins[:, 0]
    slots = pins[:, 1]

    P = pins.shape[0]
    xs = placement.new_zeros(P)
    ys = placement.new_zeros(P)
    for i in range(P):
        owner = int(owners[i])
        slot = int(slots[i])
        if owner < num_hard:
            ox, oy = benchmark.macro_pin_offsets[owner][slot].tolist()
            xs[i] = placement[owner, 0] + ox
            ys[i] = placement[owner, 1] + oy
        elif owner < num_macros:
            xs[i] = placement[owner, 0]
            ys[i] = placement[owner, 1]
        else:
            port_idx = owner - num_macros
            xs[i] = benchmark.port_positions[port_idx, 0]
            ys[i] = benchmark.port_positions[port_idx, 1]
    return xs, ys


def _hpwl(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    if xs.numel() <= 1:
        return xs.new_zeros(())
    return (xs.max() - xs.min()) + (ys.max() - ys.min())


def compute_optimal_orientations(
    placement: torch.Tensor,
    benchmark,
) -> torch.Tensor:
    """Greedy single-pass orientation pick (Klein-4) per hard macro.

    For each (movable) hard macro:
        for each o in {N, FN, FS, S}:
            recompute HPWL of every net incident to m, with m's pin
            offsets reflected per o (other macros' offsets unchanged).
        pick o minimizing the *delta* in summed incident-net HPWL.

    Returns: Long tensor of shape [num_hard_macros], values in {0,1,2,3}.
    Fixed macros stay at orientation N (0).
    """
    if not benchmark.net_pin_nodes:
        return torch.zeros(benchmark.num_hard_macros, dtype=torch.long)

    num_hard = benchmark.num_hard_macros
    placement_cpu = placement.detach().cpu().float()
    macro_to_nets = _build_macro_to_nets(benchmark.net_pin_nodes, num_hard)

    orientations = torch.zeros(num_hard, dtype=torch.long)

    for m in range(num_hard):
        if benchmark.macro_fixed[m]:
            continue
        nets_for_m = macro_to_nets[m]
        if not nets_for_m:
            continue
        pin_offs = benchmark.macro_pin_offsets[m]  # [num_pins, 2]
        if pin_offs.numel() == 0:
            continue

        # Per-orientation sign multipliers on (px, py)
        signs = torch.tensor([
            [+1.0, +1.0],   # N
            [-1.0, +1.0],   # FN
            [+1.0, -1.0],   # FS
            [-1.0, -1.0],   # S
        ], dtype=placement_cpu.dtype)

        best_o = 0
        best_total = float("inf")

        for o in range(4):
            sx, sy = signs[o].tolist()
            total = 0.0
            for net_idx, m_slot in nets_for_m:
                pins = benchmark.net_pin_nodes[net_idx]
                xs, ys = _net_pin_xy(placement_cpu, benchmark, pins)
                # Recompute m's pin xy under orientation o
                ox, oy = pin_offs[m_slot].tolist()
                # Find rows where owner==m and slot==m_slot; there can be
                # multiple if the same pin appears in the net more than
                # once, but in standard netlists this is a single row.
                owners = pins[:, 0]
                slots = pins[:, 1]
                mask = (owners == m) & (slots == m_slot)
                idx = mask.nonzero(as_tuple=False).flatten()
                for k in idx.tolist():
                    xs[k] = placement_cpu[m, 0] + sx * ox
                    ys[k] = placement_cpu[m, 1] + sy * oy
                total += float(_hpwl(xs, ys))
            if total < best_total:
                best_total = total
                best_o = o
        orientations[m] = best_o

    return orientations


def save_orientations(orientations: torch.Tensor, path: str) -> None:
    """Write Klein-4 orientations as a sidecar.

    Format: dict with {"orientations": int64 tensor, "names": list[str]}.
    Tier 2 reader pulls the int tensor, maps via ORIENT_NAMES, calls
    plc.update_macro_orientation(plc_idx, ORIENT_NAMES[o]).
    """
    payload = {
        "orientations": orientations.to(torch.int64),
        "names": list(ORIENT_NAMES),
    }
    torch.save(payload, path)


def maybe_run_orientation_flip(placement: torch.Tensor, benchmark) -> None:
    """Env-gated entry. Set PLACER_V7_ORIENT_FLIP=1 to run + write sidecar.

    Sidecar path defaults to `./orientations_<benchmark.name>.pt` in CWD;
    override with PLACER_V7_ORIENT_PATH (treated as a directory if it
    ends with '/' or already exists, else as a literal file path).
    """
    if os.environ.get("PLACER_V7_ORIENT_FLIP", "0") not in ("1", "true", "True"):
        return
    orient = compute_optimal_orientations(placement, benchmark)

    raw = os.environ.get("PLACER_V7_ORIENT_PATH", "")
    if raw and (raw.endswith("/") or os.path.isdir(raw)):
        path = os.path.join(raw, f"orientations_{benchmark.name}.pt")
    elif raw:
        path = raw
    else:
        path = f"orientations_{benchmark.name}.pt"

    save_orientations(orient, path)
    counts = torch.bincount(orient, minlength=4).tolist()
    print(
        f"  [orient_flip] {benchmark.name}: "
        f"N={counts[0]} FN={counts[1]} FS={counts[2]} S={counts[3]} "
        f"-> {path}"
    )
