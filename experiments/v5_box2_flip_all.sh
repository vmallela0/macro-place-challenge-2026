#!/bin/bash
# experiments/v5_box2_flip_all.sh — apply flip optimizer to all 17 ICCAD04 benches
# in parallel (one per core). Outputs optimized .plc files to flipped_plcs/.
set -u
BENCHES=(ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18)
PLC_SOURCE="${1:-tilos}"   # 'tilos' = use initial.plc; or path prefix to placer-output .plc files
OUTDIR="${2:-flipped_plcs}"
mkdir -p "$OUTDIR"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for B in "${BENCHES[@]}"; do
  if [ "$PLC_SOURCE" = "tilos" ]; then
    INPUT="$REPO_ROOT/external/MacroPlacement/Testcases/ICCAD04/$B/initial.plc"
  else
    INPUT="$PLC_SOURCE/$B.plc"
  fi
  if [ ! -f "$INPUT" ]; then
    echo "[flip_all] WARN: skip $B - $INPUT not found"
    continue
  fi
  OUT="$OUTDIR/$B.plc"
  $REPO_ROOT/.venv/bin/python -u $REPO_ROOT/experiments/v5_box2_flip_optimizer.py \
    "$B" "$INPUT" "$OUT" > "$OUTDIR/$B.log" 2>&1 &
  echo "[flip_all] launched $B → $OUT"
done

wait
echo "[flip_all] DONE all 17 benches"
echo "=== summary ==="
for B in "${BENCHES[@]}"; do
  initial=$(grep "initial proxy" "$OUTDIR/$B.log" 2>/dev/null | head -1 | grep -oE "proxy=[0-9.]+" | cut -d= -f2)
  final=$(grep "FINAL proxy" "$OUTDIR/$B.log" 2>/dev/null | head -1 | grep -oE "proxy=[0-9.]+" | cut -d= -f2)
  red=$(grep "reduction" "$OUTDIR/$B.log" 2>/dev/null | tail -1 | grep -oE "[+-][0-9.]+%" | head -1)
  printf "%-8s initial=%s  final=%s  Δ=%s\n" "$B" "${initial:-?}" "${final:-?}" "${red:-?}"
done
