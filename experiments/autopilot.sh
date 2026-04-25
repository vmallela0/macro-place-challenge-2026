#!/bin/bash
# autopilot.sh — runs R6 → R7 → R8 (17-bench × 3300s) autonomously
# and writes experiments/SUMMARY.md when done.
set -u
REPO=/Users/vmallela/Desktop/challenges/macro-place-challenge-2026
cd "$REPO"

log() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

# Wait for any currently-running placers (presumably R6 itself)
log "waiting for R6 to finish"
while pgrep -f "macro_place.evaluate" > /dev/null 2>&1; do
  sleep 30
done
log "R6 done"

# Round 7
log "launching R7"
"$REPO/experiments/launch_round7.sh" 2>&1 | tee /tmp/r7_driver.log
log "R7 done"

# Round 8: full 17-bench × 3300s with the validated winning config
log "launching R8 (17-bench × 3300s × 1 seed)"
WT=/tmp/wt_master
H="$REPO/experiments/run_in_worktree.sh"
WIN="PLACER_SA_T0=0.00005 PLACER_ESC_HARD_DESTROY=80"
PIDS=()

# Batch 1: 10 benches
for b in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11; do
  ( "$H" final_full $WT $b 42 3300 $WIN > "/tmp/r8_${b}_s42.out" 2>&1 ) &
  PIDS+=($!)
done
log "R8 batch 1 launched (10 benches), waiting..."
wait "${PIDS[@]}"
log "R8 batch 1 done"

PIDS=()
# Batch 2: 7 benches
for b in ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  ( "$H" final_full $WT $b 42 3300 $WIN > "/tmp/r8_${b}_s42.out" 2>&1 ) &
  PIDS+=($!)
done
log "R8 batch 2 launched (7 benches), waiting..."
wait "${PIDS[@]}"
log "R8 batch 2 done"

# Generate SUMMARY.md
log "writing SUMMARY.md"
python3 - <<'EOF'
import csv
import statistics
from pathlib import Path

REPO = Path("/Users/vmallela/Desktop/challenges/macro-place-challenge-2026")
csv_path = REPO / "experiments" / "results.csv"
summary_path = REPO / "experiments" / "SUMMARY.md"

baseline = {
    "ibm01": 0.8107, "ibm02": 1.1002, "ibm03": 0.9912, "ibm04": 0.9889,
    "ibm06": 1.1826, "ibm07": 1.1277, "ibm08": 1.1132, "ibm09": 0.8238,
    "ibm10": 1.0989, "ibm11": 0.9133, "ibm12": 1.3199, "ibm13": 1.0010,
    "ibm14": 1.2675, "ibm15": 1.2291, "ibm16": 1.2024, "ibm17": 1.4535,
    "ibm18": 1.3689,
}

# Pick lowest cost per benchmark from final_full exp_id
final = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row["exp_id"] != "final_full":
            continue
        b = row["benchmark"]
        try:
            c = float(row["final_cost"])
        except (ValueError, TypeError):
            continue
        if b not in final or c < final[b]["cost"]:
            final[b] = {"cost": c, "wall": row["wall_clock_s"]}

lines = ["# v4 final 17-bench sweep results", ""]
lines.append("Config: SA T0=5e-5, escape hard destroy=80, all v3 evaluator speedups + exp0 bug fixes")
lines.append("")
lines.append("| Bench | Baseline | v4 | Delta | Wall (s) |")
lines.append("|---|---:|---:|---:|---:|")
deltas = []
costs = []
for b in sorted(baseline.keys()):
    bc = baseline[b]
    if b in final:
        c = final[b]["cost"]
        d = c - bc
        costs.append(c)
        deltas.append(d)
        lines.append(f"| {b} | {bc:.4f} | {c:.4f} | {d:+.4f} | {final[b]['wall']} |")
    else:
        lines.append(f"| {b} | {bc:.4f} | — | (missing) | — |")

if costs:
    mean = statistics.mean(costs)
    base_mean = statistics.mean(baseline.values())
    lines.append("")
    lines.append(f"**v4 mean: {mean:.4f}**")
    lines.append(f"**Baseline mean: {base_mean:.4f}**")
    lines.append(f"**Delta: {mean - base_mean:+.4f} ({100*(mean-base_mean)/base_mean:+.1f}%)**")
    lines.append(f"**Cezar #1 (leaderboard): 1.2224 → our gap: {mean-1.2224:+.4f}**")

summary_path.write_text("\n".join(lines) + "\n")
print("=" * 60)
print(f"Wrote {summary_path}")
print("=" * 60)
print("\n".join(lines))
EOF

log "autopilot done."
