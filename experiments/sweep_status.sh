#!/bin/bash
# Quick status of the in-flight v5 sweep.
# Reads results.csv, groups by exp_id, prints mean cost / completion / VALID counts.
set -u
REPO="/home/vedu_mallela/macro-place-challenge-2026"
CSV="$REPO/experiments/results.csv"
TARGET=85
V4_MEAN=1.0186

echo "=== sweep status @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo
if [ ! -f "$CSV" ]; then echo "no results.csv yet"; exit 0; fi

ROWS=$(($(wc -l < "$CSV") - 1))
ALIVE=$(pgrep -fc "macro_place.evaluate" 2>/dev/null || echo 0)
LOAD=$(awk '{print $1, $2, $3}' /proc/loadavg)
DISK=$(df -h /home | awk 'NR==2 {print $4 " free (" $5 " used)"}')
MEM=$(free -h | awk 'NR==2 {print $7 " avail / " $2 " total"}')

printf "rows: %d / %d   alive placers: %s   load(1/5/15): %s\n" "$ROWS" "$TARGET" "$ALIVE" "$LOAD"
printf "disk: %s   mem: %s\n\n" "$DISK" "$MEM"

python3 - "$CSV" <<'PY'
import csv, sys
from collections import defaultdict

V4_PER_BENCH = {
    "ibm01": 0.7803, "ibm02": 0.9737, "ibm03": 0.9254, "ibm04": 0.9345,
    "ibm06": 1.0755, "ibm07": 1.0432, "ibm08": 1.0550, "ibm09": 0.7785,
    "ibm10": 0.9625, "ibm11": 0.8191, "ibm12": 1.1764, "ibm13": 0.8906,
    "ibm14": 1.1337, "ibm15": 1.1029, "ibm16": 1.0771, "ibm17": 1.3012,
    "ibm18": 1.2865,
}
V4_FULL = sum(V4_PER_BENCH.values()) / 17

csvp = sys.argv[1]
by_exp = defaultdict(list)
with open(csvp) as fh:
    for row in csv.DictReader(fh):
        exp = row["exp_id"]
        if "sanity" in exp: continue
        cost = row.get("final_cost") or ""
        if not cost: continue
        try: c = float(cost)
        except ValueError: continue
        by_exp[exp].append((row["benchmark"], c))

if not by_exp:
    print("no completed runs yet"); sys.exit(0)

print(f"{'exp_id':<22} {'n':>3}/17  {'mean':>7}  {'v4_same':>7}  {'Δ':>7}  benches")
print("-" * 88)
for exp in sorted(by_exp):
    rows = sorted(by_exp[exp])
    n = len(rows)
    benches = [b for b, _ in rows]
    v5_mean = sum(c for _, c in rows) / n
    v4_same = sum(V4_PER_BENCH[b] for b in benches) / n   # v4 mean over the SAME benches
    d = v5_mean - v4_same
    flag = "<--" if d < -0.005 else ("(+)" if d > 0.005 else "")
    bs = ",".join(b.replace("ibm", "") for b in benches)
    print(f"{exp:<22} {n:>3}/17  {v5_mean:>7.4f}  {v4_same:>7.4f}  {d:>+7.4f}  {bs} {flag}")

print(f"\nv4 reference (full 17 on Mac): {V4_FULL:.4f}")
PY
