"""Per-bench utilization (= total macro area / canvas area).

This is the floor for density: in the continuous limit, the minimum
top-K mean density equals the utilization (achievable by uniform
spread). The discrete grid version is usually within 5% of this.
Combined with the LP HPWL lower bound:

    proxy_lb_combined = wl_LP_LB + 0.5 · utilization + 0 · cong_LB

This is a STRICT lower bound on the achievable proxy. (Honest bound;
the real density+wl+overlap optimum may be higher because density and
HPWL conflict, but each individual minimum bounds the sum from below.)
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    benches = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
               "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
               "ibm16", "ibm17", "ibm18"]
    rows = []
    print(f"{'bench':<8} {'n_macros':>9} {'macro_area':>12} "
          f"{'canvas_area':>12} {'utilization':>11}")
    for b in benches:
        from macro_place.benchmark import Benchmark
        pt = ROOT / "benchmarks" / "processed" / "public" / f"{b}.pt"
        raw = torch.load(str(pt), weights_only=False)
        valid = set(Benchmark.__dataclass_fields__.keys())
        f = {k: v for k, v in raw.items() if k in valid}
        if "num_hard_macros" not in f:
            f["num_hard_macros"] = f["num_macros"]
            f["num_soft_macros"] = 0
        if "soft_macro_indices" not in f: f["soft_macro_indices"] = []
        if "port_positions" not in f: f["port_positions"] = torch.zeros(0, 2)
        if "macro_pin_offsets" not in f: f["macro_pin_offsets"] = []
        bench = Benchmark(**f)
        sizes = bench.macro_sizes.cpu().numpy()
        macro_area = float((sizes[:, 0] * sizes[:, 1]).sum())
        canvas_area = float(bench.canvas_width * bench.canvas_height)
        util = macro_area / canvas_area
        rows.append({"bench": b,
                      "n_macros": int(bench.num_macros),
                      "macro_area": macro_area,
                      "canvas_area": canvas_area,
                      "utilization": util})
        print(f"{b:<8} {bench.num_macros:>9} {macro_area:>12.2f} "
              f"{canvas_area:>12.2f} {util:>11.4f}")

    out = ROOT / "research" / "lower_bounds" / "utilization.csv"
    with open(out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
