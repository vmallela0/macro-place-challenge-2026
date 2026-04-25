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

REPO="$REPO" V4_MEAN="$V4_MEAN" python3 - "$CSV" <<'PY'
import csv, os, sys
from collections import defaultdict
csvp = sys.argv[1]
v4 = float(os.environ["V4_MEAN"])
by_exp = defaultdict(list)
status = defaultdict(lambda: {"valid": 0, "invalid": 0, "fail": 0})
with open(csvp) as fh:
    r = csv.DictReader(fh)
    for row in r:
        exp = row["exp_id"]
        if "sanity" in exp: continue
        cost = row.get("final_cost") or ""
        if not cost:
            status[exp]["fail"] += 1
            continue
        try:
            c = float(cost)
        except ValueError:
            status[exp]["fail"] += 1
            continue
        # Heuristic: VALID rows have nonzero cost; check log presence later.
        by_exp[exp].append((row["benchmark"], c))
        # Status from notes? CSV has no status col; trust nonzero cost as VALID-ish.
        status[exp]["valid"] += 1

if not by_exp:
    print("no completed runs yet")
    sys.exit(0)

print(f"{'exp_id':<22} {'n':>3}/17  {'mean':>7}  {'vs_v4':>7}  benches")
print("-" * 78)
for exp in sorted(by_exp):
    rows = by_exp[exp]
    n = len(rows)
    mean = sum(c for _, c in rows) / n
    delta = mean - v4
    flag = "<--" if delta < -0.005 else ("(+)" if delta > 0.005 else "")
    benches = ",".join(b.replace("ibm", "") for b, _ in sorted(rows))
    print(f"{exp:<22} {n:>3}/17  {mean:>7.4f}  {delta:>+7.4f}  {benches} {flag}")
PY
