"""Render a static placement plot for a v6 sweep result.

Usage:
    scripts/v6_placement_plot.py <bench> <placement.npy> [out.png]

Reads the saved placement (`PLACER_V6_SAVE_PLACEMENT` output from the
sweep) and the matching benchmark, computes the proxy components via
the official PlacementCost, and writes a PNG showing:
  - canvas border (black box)
  - hard macros (red rectangles, sized to actual w x h)
  - soft macro centroids (light-blue dots)
  - title with proxy / wl / den / cong / overlaps

Style mirrors `assets/ibm01_v4.png` so the v6 plots are directly
comparable to v4's.

Dependencies: matplotlib, numpy, torch (already in pyproject.toml).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def render_placement(bench_name: str, npy_path: str, out_path: str | None = None):
    pt_path = ROOT / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"
    if not pt_path.exists():
        # Try as a literal path
        pt_path = Path(bench_name)
    bench = Benchmark.load(str(pt_path))

    pos_np = np.load(npy_path)
    placement = torch.tensor(pos_np, dtype=torch.float32)

    # Compute proxy via official PlacementCost. Use a fresh PLC each time
    # since compute_proxy_cost mutates plc state.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_v1", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    plc = v1._load_plc(bench.name)
    r = compute_proxy_cost(placement, bench, plc)
    proxy = float(r["proxy_cost"])
    wl = float(r["wirelength_cost"])
    den = float(r["density_cost"])
    cong = float(r["congestion_cost"])
    overlaps = int(r["overlap_count"])

    n_hard = bench.num_hard_macros
    n_total = placement.shape[0]
    sizes = bench.macro_sizes.numpy()
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Canvas border
    ax.add_patch(Rectangle(
        (0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.2))

    # Soft macro centroids (light blue, alpha)
    if n_total > n_hard:
        soft_x = pos_np[n_hard:n_total, 0]
        soft_y = pos_np[n_hard:n_total, 1]
        ax.scatter(soft_x, soft_y, s=3.5, c="#4a90d9", alpha=0.18,
                   marker="s", linewidths=0)

    # Hard macros as red rectangles, sized to actual w x h
    for i in range(n_hard):
        cx, cy = float(pos_np[i, 0]), float(pos_np[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        ax.add_patch(Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            facecolor="#f4a8a8", edgecolor="#c44747", linewidth=0.7,
            alpha=0.85))

    ax.set_xlim(-0.5, cw + 0.5)
    ax.set_ylim(-0.5, ch + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    valid_str = "VALID" if overlaps == 0 else f"INVALID(ov={overlaps})"
    ax.set_title(
        f"{bench_name} — vmallela v6 (GPU + portfolio + consensus)\n"
        f"proxy={proxy:.4f}  wl={wl:.3f}  den={den:.3f}  cong={cong:.3f}  "
        f"{valid_str}",
        fontsize=11)

    if out_path is None:
        out_path = str(ROOT / "assets" / f"v6_{bench_name}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[v6_plot] {bench_name} -> {out_path}  "
          f"(proxy={proxy:.4f} {valid_str})")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <bench_name> <placement.npy> [out.png]",
              file=sys.stderr)
        sys.exit(2)
    bench_name = sys.argv[1]
    npy_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    render_placement(bench_name, npy_path, out_path)


if __name__ == "__main__":
    main()
