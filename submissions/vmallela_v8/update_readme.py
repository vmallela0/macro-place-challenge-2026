#!/usr/bin/env python3
"""Regenerate the live results table in submissions/vmallela_v8/README.md
from results.csv. Replaces the block between <!--V8:START--> and <!--V8:END-->.

Comparison columns: v7 dev-box reference (per slj2/update_readme.py) and
v7 c4d sweep baseline. Computes per-bench delta; flags regressions > 0.005.
"""
from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"
README = HERE / "README.md"
START = "<!--V8:START-->"
END = "<!--V8:END-->"

V7_DEV = {
    "ibm01": 0.7653, "ibm02": 0.9482, "ibm03": 0.9166, "ibm04": 0.9287,
    "ibm06": 1.0546, "ibm07": 1.0324, "ibm08": 1.0291, "ibm09": 0.7628,
    "ibm10": 0.9492, "ibm11": 0.8013, "ibm12": 1.1557, "ibm13": 0.8757,
    "ibm14": 1.1070, "ibm15": 1.0835, "ibm16": 1.0435, "ibm17": 1.2813,
    "ibm18": 1.2697,
}
ORDER = ["ibm15", "ibm17", "ibm18", "ibm12", "ibm14", "ibm16", "ibm13",
         "ibm04", "ibm06", "ibm07", "ibm08", "ibm09", "ibm10", "ibm11",
         "ibm01", "ibm02", "ibm03"]


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
    lines.append(f"_v8 results — {len(rows)} / 17 complete. ibm15-first ordering._")
    lines.append("")
    lines.append("| Bench | v8 proxy | v7 dev-box | Δ vs v7 | Overlaps | Wall (s) | Status | PNG |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|:---:|")
    proxies: list[float] = []
    deltas: list[float] = []
    walls: list[int] = []
    regressions = 0
    for b in ORDER:
        v7 = V7_DEV.get(b)
        v7_str = f"{v7:.4f}" if v7 is not None else "—"
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

            p_str = f"{p:.4f}" if p is not None else "—"
            ov_str = str(ov) if ov is not None else "—"
            w_str = str(w) if w is not None else "—"

            if p is not None and v7 is not None:
                d = p - v7
                deltas.append(d)
                d_str = f"{d:+.4f}"
                if d > 0.005:
                    status = "REGRESS"
                    regressions += 1
                elif d < -0.005:
                    status = "WIN"
                else:
                    status = "≈"
            else:
                d_str = "—"
                status = "—"

            png = HERE / "png" / f"{b}.png"
            png_str = f"[![]({png.relative_to(HERE.parent.parent)})]({png.relative_to(HERE.parent.parent)})" if png.exists() else "—"
        else:
            p_str = ov_str = w_str = d_str = status = png_str = "—"
        lines.append(f"| {b} | {p_str} | {v7_str} | {d_str} | {ov_str} | {w_str} | {status} | {png_str} |")

    lines.append("")
    if proxies:
        mean = sum(proxies) / len(proxies)
        lines.append(f"**mean v8 proxy:** {mean:.4f}  (target ≤ 0.998)")
        lines.append(f"**regressions > 0.005:** {regressions}")
        if walls:
            lines.append(f"**max wall:** {max(walls)} s  (cap 3600)")
    return "\n".join(lines)


def main() -> None:
    rows = read_rows()
    table = fmt_table(rows)
    if not README.exists():
        README.write_text(f"# vmallela_v8\n\n{START}\n{table}\n{END}\n")
        return
    text = README.read_text()
    if START not in text or END not in text:
        text = text.rstrip() + f"\n\n{START}\n{table}\n{END}\n"
    else:
        before = text.split(START)[0]
        after = text.split(END)[1]
        text = f"{before}{START}\n{table}\n{END}{after}"
    README.write_text(text)


if __name__ == "__main__":
    main()
