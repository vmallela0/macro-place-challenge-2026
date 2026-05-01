#!/usr/bin/env python3
"""Regenerate the live results table in README.md from lsj/results.csv.

Replaces the block between <!--LSJ:START--> and <!--LSJ:END-->. Idempotent.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "lsj" / "results.csv"
README = REPO / "README.md"
START = "<!--LSJ:START-->"
END = "<!--LSJ:END-->"

DEV_REF = {
    "ibm01": 0.7653, "ibm02": 0.9482, "ibm03": 0.9166, "ibm04": 0.9287,
    "ibm06": 1.0546, "ibm07": 1.0324, "ibm08": 1.0291, "ibm09": 0.7628,
    "ibm10": 0.9492, "ibm11": 0.8013, "ibm12": 1.1557, "ibm13": 0.8757,
    "ibm14": 1.1070, "ibm15": 1.0835, "ibm16": 1.0435, "ibm17": 1.2813,
    "ibm18": 1.2697,
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
    lines.append(f"_Live results — c4d-standard-16, 16 vCPU AMD EPYC Turin. {len(rows)} / 17 complete._")
    lines.append("")
    lines.append("| Bench | Proxy cost | Dev-box ref | Δ (this − dev) | Overlaps | Wall (s) | PNG |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    proxies: list[float] = []
    walls: list[int] = []
    overlaps_ok = True
    for b in ORDER:
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
            ref = DEV_REF.get(b)
            if p is not None and ref is not None:
                delta = p - ref
                p_str = f"{p:.4f}"
                d_str = f"{delta:+.4f}"
            else:
                p_str = r.get("proxy_cost", "—") or "—"
                d_str = "—"
            ref_str = f"{ref:.4f}" if ref is not None else "—"
            w_str = str(w) if w is not None else "—"
            png_path = f"lsj/png/{b}.png"
            png_link = f"[png]({png_path})" if (REPO / png_path).exists() else "—"
            lines.append(f"| {b} | {p_str} | {ref_str} | {d_str} | {ov_disp} | {w_str} | {png_link} |")
        else:
            ref = DEV_REF.get(b)
            ref_str = f"{ref:.4f}" if ref is not None else "—"
            lines.append(f"| {b} | _pending_ | {ref_str} | — | — | — | — |")
    if proxies:
        mean = sum(proxies) / len(proxies)
        ref_mean = sum(DEV_REF[b] for b in ORDER if b in rows and b in DEV_REF) / len(proxies)
        delta_mean = mean - ref_mean
        wall_total = sum(walls)
        target_mean = 1.0003
        bar = f"target ≤ {target_mean:.4f}"
        if len(proxies) == 17:
            verdict = "✓ matches dev box" if abs(mean - target_mean) <= 0.005 else "⚠ deviates"
            verdict += " · overlaps clean" if overlaps_ok else " · overlap regression"
        else:
            verdict = "(in progress)"
        lines.append(
            f"| **mean ({len(proxies)}/17)** | **{mean:.4f}** | {ref_mean:.4f} | "
            f"**{delta_mean:+.4f}** | — | {wall_total} | — |"
        )
        lines.append("")
        lines.append(f"_Running mean: **{mean:.4f}** ({bar}). {verdict}._")
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
