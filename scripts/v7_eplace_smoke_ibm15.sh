#!/bin/bash
# Wrapper for the ePlace smoke. Logs to /tmp/v7_eplace_<ts>/
set -u
cd "$(dirname "$0")/.."

OUT="/tmp/v7_eplace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "v7 ePlace smoke on ibm15" > "$OUT/sweep.log"
echo "  started: $(date)" >> "$OUT/sweep.log"
echo "  Sequential: Run A (.plc init) then Run B (ePlace init), 1200s each." >> "$OUT/sweep.log"

t_start=$(date +%s)
.venv/bin/python scripts/v7_eplace_smoke_ibm15.py > "$OUT/smoke.log" 2>&1
rc=$?
t_end=$(date +%s)
elapsed=$((t_end - t_start))

echo "" >> "$OUT/sweep.log"
echo "FINAL wall=${elapsed}s rc=$rc" | tee -a "$OUT/sweep.log"
echo "" | tee -a "$OUT/sweep.log"
echo "Verdict block:" | tee -a "$OUT/sweep.log"
grep -A 12 "VERDICT" "$OUT/smoke.log" 2>/dev/null | tee -a "$OUT/sweep.log"
echo "DONE" >> "$OUT/sweep.log"
