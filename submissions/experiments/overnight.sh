#!/bin/bash
# Overnight: run placer_submission.py on all 17 IBM benchmarks.
# Budget per benchmark varies with size. Total ~2h target.
#
#   ibm01-09: 300-450s each   (small-medium)
#   ibm10-14: 500-600s each  (medium)
#   ibm15-18: 700-800s each  (large/sparse)

set -u
OUT=submissions/experiments/results/overnight
mkdir -p $OUT

log() { echo "[$(date +%Y-%m-%d' '%H:%M:%S)] $*"; }

run () {
    local bench=$1
    local budget=$2
    local soft=$3
    local of="$OUT/${bench}.log"
    log "START $bench ${budget}s (soft ${soft}s)"
    PLACER_TOTAL_BUDGET=$budget PLACER_SOFT_BUDGET=$soft PLACER_PARALLEL_WORKERS=0 \
        uv run evaluate submissions/experiments/placer_submission.py -b $bench \
        > "$of" 2>&1
    local line=$(grep -E "^proxy=|FAILED" "$of" | tail -1)
    log "DONE  $bench $line"
}

# 17 benchmarks, total ~2h
run ibm01 400  280
run ibm02 550  380
run ibm03 500  340
run ibm04 500  340
run ibm06 550  380
run ibm07 550  380
run ibm08 550  380
run ibm09 400  280
run ibm10 550  350
run ibm11 600  400
run ibm12 600  400
run ibm13 600  400
run ibm14 650  430
run ibm15 700  460
run ibm16 700  460
run ibm17 800  500
run ibm18 750  480

log "All done. Summarizing..."

echo "" >> $OUT/SUMMARY.md
echo "# Overnight run $(date +%Y-%m-%d)" > $OUT/SUMMARY.md
echo "" >> $OUT/SUMMARY.md
echo "| Benchmark | Proxy | Overlaps | Time |" >> $OUT/SUMMARY.md
echo "|-----------|-------|----------|------|" >> $OUT/SUMMARY.md
for f in $OUT/ibm*.log; do
    bench=$(basename "$f" .log)
    line=$(grep -E "^proxy=" "$f" | tail -1)
    ov=$(grep -oE "VALID|INVALID \\([0-9]+ overlaps\\)" "$f" | tail -1)
    printf "| %s | %s | %s | %s |\n" "$bench" \
        "$(echo $line | grep -oE 'proxy=[0-9.]+' | head -1)" \
        "$ov" \
        "$(echo $line | grep -oE '\\[[0-9.]+s\\]' | tail -1)" \
        >> $OUT/SUMMARY.md
done
cat $OUT/SUMMARY.md
