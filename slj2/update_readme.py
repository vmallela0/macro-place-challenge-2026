#!/usr/bin/env python3
"""Regenerate the live results table in README.md from slj2/results.csv.

Replaces the block between <!--SLJ2:START--> and <!--SLJ2:END-->. Idempotent.
Comparison column is the published dev-box reference (also shows v4 baseline
so the lift over the un-upgraded pipeline is visible).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "slj2" / "results.csv"
README = REPO / "README.md"
START = "<!--SLJ2:START-->"
END = "<!--SLJ2:END-->"

DEV_V7 = {
    "ibm01": 0.7653, "ibm02": 0.9482, "ibm03": 0.9166, "ibm04": 0.9287,
    "ibm06": 1.0546, "ibm07": 1.0324, "ibm08": 1.0291, "ibm09": 0.7628,
    "ibm10": 0.9492, "ibm11": 0.8013, "ibm12": 1.1557, "ibm13": 0.8757,
    "ibm14": 1.1070, "ibm15": 1.0835, "ibm16": 1.0435, "ibm17": 1.2813,
    "ibm18": 1.2697,
}
V4_BASELINE = {
    "ibm01": 0.7803, "ibm02": 0.9737, "ibm03": 0.9254, "ibm04": 0.9345,
    "ibm06": 1.0755, "ibm07": 1.0432, "ibm08": 1.0550, "ibm09": 0.7785,
    "ibm10": 0.9625, "ibm11": 0.8191, "ibm12": 1.1764, "ibm13": 0.8906,
    "ibm14": 1.1337, "ibm15": 1.1029, "ibm16": 1.0771, "ibm17": 1.3012,
    "ibm18": 1.2865,
}
ORDER = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
         "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
         "ibm16", "ibm17", "ibm18"]


def read_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not CSV_PATH.exists():
        return rows
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r.get("benchmark"):
                rows[r["benchmark"]] = r
    return rows


def fmt_table(rows: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append(f"_Live results — slj2 algorithm on c4d-standard-16. {len(rows)} / 17 complete._")
    lines.append("")
    lines.append("| Bench | slj2 proxy | v4 baseline | dev-box v7 | Δ vs v4 | Δ vs dev-box | Overlaps | Wall (s) | PNG |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    proxies: list[float] = []
    walls: list[int] = []
    overlaps_ok = True
    for b in ORDER:
        v4 = V4_BASELINE.get(b)
        dev = DEV_V7.get(b)
        v4_str = f"{v4:.4f}" if v4 is not None else "—"
        dev_str = f"{dev:.4f}" if dev is not None else "—"
        if b in rows:
            r = rows[b]
            try:
                p = float(r["proxy_cost"])
                proxies.append(p)
            except (TypeError, ValueError):
                p = None
            try:
                w = int(float(r["wall_clock_s"]))
                walls.append(w)
            except (TypeError, ValueError):
                w = None
            ov_raw = r.get("overlap_count", "")
            try:
                ov = int(ov_raw)
            except (TypeError, ValueError):
                ov = None
            ov_disp = "0 ✓" if ov == 0 else f"**{ov_raw}** ✗"
            if ov != 0:
                overlaps_ok = False
            if p is not None:
                p_str = f"{p:.4f}"
                d_v4 = f"{p - v4:+.4f}" if v4 is not None else "—"
                d_dev = f"{p - dev:+.4f}" if dev is not None else "—"
            else:
                p_str = r.get("proxy_cost", "—") or "—"
                d_v4 = "—"
                d_dev = "—"
            w_str = str(w) if w is not None else "—"
            png_path = f"slj2/png/{b}.png"
            png_link = f"[png]({png_path})" if (REPO / png_path).exists() else "—"
            lines.append(
                f"| {b} | {p_str} | {v4_str} | {dev_str} | {d_v4} | {d_dev} | "
                f"{ov_disp} | {w_str} | {png_link} |"
            )
        else:
            lines.append(
                f"| {b} | _pending_ | {v4_str} | {dev_str} | — | — | — | — | — |"
            )
    if proxies:
        mean = sum(proxies) / len(proxies)
        v4_mean = sum(V4_BASELINE[b] for b in ORDER if b in rows) / len(proxies)
        dev_mean = sum(DEV_V7[b] for b in ORDER if b in rows) / len(proxies)
        wall_total = sum(walls)
        lines.append(
            f"| **mean ({len(proxies)}/17)** | **{mean:.4f}** | {v4_mean:.4f} | "
            f"{dev_mean:.4f} | **{mean - v4_mean:+.4f}** | "
            f"**{mean - dev_mean:+.4f}** | — | {wall_total} | — |"
        )
        lines.append("")
        bar = ""
        if len(proxies) == 17:
            if overlaps_ok and mean < v4_mean:
                bar = "✓ beats v4 baseline."
            elif not overlaps_ok:
                bar = "⚠ overlap regression."
            else:
                bar = "⚠ does not beat v4 baseline."
        else:
            bar = "(in progress)"
        lines.append(f"_Running mean: **{mean:.4f}** (v4 baseline {v4_mean:.4f}, dev-box v7 {dev_mean:.4f}). {bar}_")
    else:
        lines.append("")
        lines.append("_Sweep has not started — table will populate as each bench finishes._")
    return "\n".join(lines)


def main() -> int:
    rows = read_rows()
    table = fmt_table(rows)
    text = README.read_text()
    if START not in text or END not in text:
        print(f"markers {START} / {END} not found in README; refusing to edit",
              file=sys.stderr)
        return 1
    pre, _, rest = text.partition(START)
    _, _, post = rest.partition(END)
    new_text = f"{pre}{START}\n{table}\n{END}{post}"
    if new_text != text:
        README.write_text(new_text)
    print(f"updated README — {len(rows)} / 17 rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
