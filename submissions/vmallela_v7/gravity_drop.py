"""gravity_drop.py — n-D simplex bead-sort initialization for macro placement.

USER INTUITION: in 2D, connected macros COLLIDE when satisfying net constraints.
In higher D, there's MORE ROOM — they can stack VERTICALLY without horizontal
collision. Apply gravity to "settle" them onto the 2D canvas. The settling
preserves the horizontal arrangement determined in higher D.

MATHEMATICAL BASIS:
  - Multidimensional scaling: high-D embedding preserves graph-metric
    distances better than direct 2D embedding for graphs with topology
    that doesn't embed isometrically in R^2 (most netlists).
  - Bead sort (Arulanandham 2002): unary representations + gravity sort
    in linear time. Net connections = unary "beads"; gravity drop
    extracts the 2D arrangement.
  - Whitney embedding: any d-dim smooth manifold embeds in R^{2d+1}.
    For 2D placement, sufficient dimension is small (constants, 4-8 typically).
  - Simplex aggressive form: in R^{n-1} (n macros), random Gaussian init is
    near-orthogonal (concentration of measure) — essentially a regular simplex.
    Net forces resolve in orthogonal subspaces with zero conflict.
  - Harmonic gravity homotopy: ramping spring force pulls dims 2..n-1
    toward 0 over time. This is a CONTINUOUS DEFORMATION of the embedding.
    Net forces in collapsing dims transfer structure into the canvas plane.

OUTPUT: a (n_total, 2) placement that is the 2D PROJECTION of the higher-D
configuration. We then score it with compute_proxy_cost.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_v1():
    """Load submissions/vmallela/placer.py as module v1 (gives _load_plc and IncrementalEvaluator)."""
    v1_path = REPO / "submissions" / "vmallela" / "placer.py"
    spec = importlib.util.spec_from_file_location("_v1_grav", str(v1_path))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    return v1


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def gravity_drop(plc, benchmark, n_dim=None, n_iters=500, dt=0.05,
                 k_spring=1.0, k_repel=0.5, repel_range=1.0,
                 gravity_max=0.5, damping=0.92, seed=42,
                 weight_clip=10.0, verbose=False, max_dim_cap=64):
    """n-D physics with harmonic-gravity homotopy collapse → 2D placement.

    n_dim = None ⇒ use min(n_total - 1, max_dim_cap), the (n-1)-simplex
    capped for memory.

    Dims 0, 1 are canvas-aligned (where the final placement lives).
    Dims 2..n_dim are simplex-orthogonal directions for net forces to
    resolve in without conflict. Harmonic gravity ramps over time,
    pulling the extras to 0 and transferring structure into dims 0, 1.
    """
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.macro_positions.shape[0])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    if n_dim is None:
        n_dim = min(n_total - 1, max_dim_cap)
    n_dim = max(n_dim, 3)
    if verbose:
        kind = "simplex" if n_dim == n_total - 1 else "capped simplex"
        print(f"  [grav] n_dim={n_dim} ({kind}) for n_total={n_total}", flush=True)

    rng = np.random.default_rng(seed)

    # --- Map macros + ports ---
    plc_to_macro = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_to_macro[plc.modules_w_pins[plc_idx].get_name()] = bidx
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        plc_to_macro[plc.modules_w_pins[plc_idx].get_name()] = n_hard + bidx
    port_pos = {}
    for plc_idx in plc.port_indices:
        mod = plc.modules_w_pins[plc_idx]
        port_pos[mod.get_name()] = mod.get_pos()

    # --- Build net macro lists + anchors (2D) ---
    net_macros = []
    net_weights = []
    net_anchors_2d = []
    for driver_name, sinks in plc.nets.items():
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = min(float(plc.modules_w_pins[driver_plc_idx].get_weight()), weight_clip)
        nm = []
        na = []
        for pin_name in [driver_name] + sinks:
            parent = pin_name.split("/")[0]
            if parent in plc_to_macro:
                nm.append(plc_to_macro[parent])
            elif parent in port_pos:
                px, py = port_pos[parent]
                na.append((px, py))
        if len(nm) + len(na) < 2:
            continue
        net_macros.append(np.array(nm, dtype=np.int32))
        net_weights.append(weight)
        net_anchors_2d.append(np.array(na, dtype=np.float64) if na else np.zeros((0, 2)))

    # --- Macro sizes (for 2D repulsion) ---
    macro_w = np.zeros(n_total, dtype=np.float64)
    macro_h = np.zeros(n_total, dtype=np.float64)
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        mod = plc.modules_w_pins[plc_idx]
        macro_w[bidx] = mod.get_width()
        macro_h[bidx] = mod.get_height()
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        mod = plc.modules_w_pins[plc_idx]
        macro_w[n_hard + bidx] = mod.get_width()
        macro_h[n_hard + bidx] = mod.get_height()

    # --- Initialize: dims 0,1 = uniform in canvas; extras = Gaussian (simplex) ---
    sim_scale = max(cw, ch) * 0.5
    positions = np.zeros((n_total, n_dim), dtype=np.float64)
    positions[:, 0] = rng.uniform(0, cw, n_total)
    positions[:, 1] = rng.uniform(0, ch, n_total)
    positions[:, 2:] = rng.standard_normal((n_total, n_dim - 2)) * sim_scale
    velocities = np.zeros_like(positions)

    # Anchors live at canvas (x, y), zero in extras (they don't move)
    net_anchors_nd = []
    for anc in net_anchors_2d:
        if anc.shape[0] == 0:
            net_anchors_nd.append(np.zeros((0, n_dim)))
        else:
            padded = np.zeros((anc.shape[0], n_dim))
            padded[:, :2] = anc
            net_anchors_nd.append(padded)

    t0 = time.time()
    for step in range(n_iters):
        # Gravity ramps from 0 at step 0 to gravity_max at step n_iters/2, then holds
        gravity_now = gravity_max * min(1.0, step / max(n_iters * 0.5, 1.0))
        forces = np.zeros_like(positions)

        # 1. Net spring forces (all n_dim dims)
        for nm, nw, na in zip(net_macros, net_weights, net_anchors_nd):
            if nm.size == 0:
                continue
            macro_pos = positions[nm]
            if na.shape[0] > 0:
                all_pos = np.concatenate([macro_pos, na], axis=0)
            else:
                all_pos = macro_pos
            centroid = all_pos.mean(axis=0)
            forces[nm] += k_spring * nw * (centroid - macro_pos)

        # 2. Repulsion only in 2D (canvas plane). Extras already separated by Gaussian init.
        if n_hard > 1 and n_hard <= 800:
            hard_pos_2d = positions[:n_hard, :2]
            sizes = np.sqrt(macro_w[:n_hard] * macro_h[:n_hard])
            avg_size = max(sizes.mean(), 1e-6)
            diff_2d = hard_pos_2d[:, None, :] - hard_pos_2d[None, :, :]
            dist2_2d = (diff_2d * diff_2d).sum(axis=2) + 1e-9
            dist_2d = np.sqrt(dist2_2d)
            threshold = avg_size * repel_range
            sigma = avg_size * 0.8
            mask = (dist_2d < threshold) & (dist_2d > 1e-6)
            ratio = sigma / np.maximum(dist_2d, 1e-3)
            rep_mag = np.where(mask, k_repel * (ratio**7 - 1.0), 0.0)
            rep_mag = np.clip(rep_mag, 0, k_repel * 10.0)
            rep_force_2d = rep_mag[..., None] * diff_2d / np.maximum(dist_2d[..., None], 1e-3)
            forces[:n_hard, :2] += rep_force_2d.sum(axis=1)

        # 3. Harmonic gravity homotopy on extras (continuous projection)
        if n_dim > 2:
            forces[:, 2:] -= gravity_now * positions[:, 2:]

        # Verlet
        velocities += dt * forces
        velocities *= damping
        positions += dt * velocities

        # Clip 2D dims to canvas
        positions[:, 0] = np.clip(positions[:, 0], 0, cw)
        positions[:, 1] = np.clip(positions[:, 1], 0, ch)

        if verbose and (step % 50 == 0 or step == n_iters - 1):
            avg_extra = float(positions[:, 2:].mean()) if n_dim > 2 else 0.0
            x_lo, x_hi = float(positions[:, 0].min()), float(positions[:, 0].max())
            y_lo, y_hi = float(positions[:, 1].min()), float(positions[:, 1].max())
            print(f"  step {step:4d}  avg_extra={avg_extra:.3f}  "
                  f"x=[{x_lo:.1f},{x_hi:.1f}]  y=[{y_lo:.1f},{y_hi:.1f}]  "
                  f"gravity={gravity_now:.3f}", flush=True)

    if verbose:
        print(f"  sim wall: {time.time()-t0:.1f}s", flush=True)

    # Read off the 2D placement, clip to canvas considering macro sizes
    P_2d = positions[:, :2].copy()
    for i in range(n_total):
        w_i = float(macro_w[i])
        h_i = float(macro_h[i])
        if w_i > 0:
            P_2d[i, 0] = max(0.0, min(P_2d[i, 0], cw - w_i))
            P_2d[i, 1] = max(0.0, min(P_2d[i, 1], ch - h_i))
    return P_2d.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True,
                   help="JSON: {positions, soft_positions, cost, ...}")
    p.add_argument("--n-dim", type=int, default=None,
                   help="embedding dim (None ⇒ (n-1)-simplex capped at --max-dim-cap)")
    p.add_argument("--max-dim-cap", type=int, default=64)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.03)
    p.add_argument("--k-spring", type=float, default=0.3)
    p.add_argument("--k-repel", type=float, default=2.0)
    p.add_argument("--repel-range", type=float, default=2.0)
    p.add_argument("--gravity-max", type=float, default=0.05)
    p.add_argument("--damping", type=float, default=0.93)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench-path", default=None,
                   help="path to benchmarks/processed/public/{name}.pt (default: auto)")
    args = p.parse_args()

    v1 = load_v1()
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost

    bench_path = args.bench_path or f"benchmarks/processed/public/{args.benchmark}.pt"
    benchmark = Benchmark.load(bench_path)
    plc = v1._load_plc(args.benchmark)
    if plc is None:
        print(f"FATAL: could not load plc for {args.benchmark}", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    print(f"[grav] benchmark={args.benchmark} n_total={int(benchmark.macro_positions.shape[0])} "
          f"n_hard={int(benchmark.num_hard_macros)} n_dim={args.n_dim}", flush=True)

    P_2d = gravity_drop(
        plc, benchmark,
        n_dim=args.n_dim, n_iters=args.n_iters, dt=args.dt,
        k_spring=args.k_spring, k_repel=args.k_repel, repel_range=args.repel_range,
        gravity_max=args.gravity_max, damping=args.damping, seed=args.seed,
        max_dim_cap=args.max_dim_cap, verbose=True)
    print(f"[grav] simulation wall: {time.time()-t0:.1f}s", flush=True)

    # Score
    full = benchmark.macro_positions.clone()
    full[:] = torch.from_numpy(P_2d)
    costs = compute_proxy_cost(full, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlaps = int(costs.get("overlap_count", -1))
    print(f"[grav] raw proxy={proxy:.4f}  wl={float(costs['wirelength_cost']):.4f}  "
          f"den={float(costs['density_cost']):.4f}  cong={float(costs['congestion_cost']):.4f}  "
          f"overlaps={overlaps}", flush=True)

    n_hard = int(benchmark.num_hard_macros)
    out = {
        "benchmark": args.benchmark,
        "cost": proxy,
        "wirelength_cost": float(costs["wirelength_cost"]),
        "density_cost": float(costs["density_cost"]),
        "congestion_cost": float(costs["congestion_cost"]),
        "overlap_count": overlaps,
        "positions": P_2d[:n_hard].tolist(),
        "soft_positions": P_2d[n_hard:].tolist(),
        "wall_s": time.time() - t0,
        "n_dim": args.n_dim if args.n_dim is not None else min(int(benchmark.macro_positions.shape[0]) - 1, args.max_dim_cap),
        "seed": args.seed,
        "method": "gravity_drop",
    }
    atomic_write_json(args.output, out)
    print(f"[grav] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
