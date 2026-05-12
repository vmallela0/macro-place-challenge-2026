#!/bin/bash
# zeus_supreme_autopilot — runs zeus_supreme_ab to completion, picks the
# winning arm from the 17×4 factorial, then launches a full-pipeline
# verification of that arm on all 17 benches (zeus_full17.sh wrapper).
#
# Designed to run detached (parent=init) — survives SSH/claude/laptop
# death. Writes ~/zeus_runs/AUTOPILOT_STATUS.md continuously.
#
# Decision rule:
#   arm winner = argmin mean Δ (Δ = proxy - verified)
#   tie-break: simpler arm wins (base > rudy > hmc > rudy_hmc)
#   abort if best mean Δ > 0.005 (worse than albania1)
#
# Diagnostic outputs in STATUS.md:
#   - clean A/B Δ matrix
#   - winning arm + chosen config
#   - full-pipeline result mean

set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
ROOT="${ZEUS_RUN_DIR:-$HOME/zeus_runs}"
STATUS="$ROOT/AUTOPILOT_STATUS.md"
mkdir -p "$ROOT"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[supreme_autopilot $(ts)] $*"; }
write_status() {
    cat > "$STATUS" <<EOF
# Zeus supreme-autopilot status — last update $(ts)

$1
EOF
}

# ── Phase 1: launch supreme_ab inline (we ARE detached already) ────────
log "launching zeus_supreme_ab.sh as in-process child"
write_status "## Phase 1: 17×4 factorial clean A/B starting

Step 1: v4+Lap save for all 17 benches in parallel (~40 min, 17 cores).
Step 2: Hessian-only 17×4 = 68 placers in waves of 30 (~30 min).

Logs:
- master log: $(readlink -f "$0" || echo "$0").log
- supreme_ab will print OUT_DIR=/tmp/zeus_supreme_<ts> when done"

SUPREME_LOG="$ROOT/supreme_ab_$(date +%Y%m%d_%H%M%S).log"
bash "$REPO/scripts/zeus_supreme_ab.sh" > "$SUPREME_LOG" 2>&1 &
sup_pid=$!
log "supreme_ab pid=$sup_pid log=$SUPREME_LOG"

# Poll while it runs, writing status with a fresh tail.
while kill -0 $sup_pid 2>/dev/null; do
    sleep 60
    # Try to identify the OUT_DIR
    out_dir=$(grep -E "^  out:" "$SUPREME_LOG" 2>/dev/null | head -1 | awk '{print $2}')
    csv="$out_dir/results.csv"
    n_rows=0
    if [ -n "$out_dir" ] && [ -f "$csv" ]; then
        n_rows=$(($(wc -l < "$csv") - 1))
    fi
    write_status "## Phase 1: supreme_ab RUNNING

- pid: $sup_pid
- master log: \`$SUPREME_LOG\`
- out dir:   \`$out_dir\`
- rows so far: $n_rows (target: 17 v4lap + 68 hess = 85)

Live tail ($(ts)):
\`\`\`
$(tail -25 "$SUPREME_LOG" 2>/dev/null)
\`\`\`"
done
wait $sup_pid; sup_rc=$?
log "supreme_ab exit rc=$sup_rc"

# ── Phase 2: locate results.csv ────────────────────────────────────────
out_dir=$(grep -E "^  out:" "$SUPREME_LOG" 2>/dev/null | head -1 | awk '{print $2}')
csv="$out_dir/results.csv"
if [ ! -f "$csv" ]; then
    out_dir=$(ls -dt /tmp/zeus_supreme_* 2>/dev/null | head -1)
    csv="$out_dir/results.csv"
fi
if [ ! -f "$csv" ]; then
    write_status "## ERROR: no supreme_ab results.csv

Master log: \`$SUPREME_LOG\`
Searched: \`/tmp/zeus_supreme_*/results.csv\` — not found.

Manual inspection needed."
    exit 1
fi
log "results csv: $csv"

# ── Phase 3: pick winning arm ──────────────────────────────────────────
decision=$(.venv/bin/python -u <<PY
import csv
rows = [r for r in csv.DictReader(open('$csv'))
        if r.get('stage') == 'hess']
arms = {}
for r in rows:
    try: d = float(r['delta'])
    except (ValueError, KeyError): continue
    arms.setdefault(r['arm'], []).append(d)
mean = {a: sum(v)/len(v) for a, v in arms.items() if v}
if not mean:
    print("WINNER=ABORT MEAN=NA REASON=no_data")
else:
    prefer = ['base','rudy','hmc','rudy_hmc']
    sorted_a = sorted(mean.items(),
                       key=lambda kv: (kv[1],
                                       prefer.index(kv[0]) if kv[0] in prefer else 99))
    best, best_d = sorted_a[0]
    if best_d > 0.005:
        print(f"WINNER=ABORT MEAN={best_d:.4f} REASON=all_arms_regress")
    else:
        print(f"WINNER={best} MEAN={best_d:.4f} REASON=ramp")
print("TABLE:")
for a, d in sorted_a:
    cells = ' '.join(f'{x:+.4f}' for x in arms[a])
    print(f"  {a:10} mean={d:+.4f} n={len(arms[a])} per_bench=[{cells}]")
PY
)
echo "$decision" | tee -a "$SUPREME_LOG"
WINNER=$(echo "$decision" | grep -E "^WINNER=" | head -1 | sed -E 's/WINNER=([^ ]+).*/\1/')
MEAN=$(echo "$decision" | grep -E "^WINNER=" | head -1 | sed -E 's/.*MEAN=([^ ]+).*/\1/')
REASON=$(echo "$decision" | grep -E "^WINNER=" | head -1 | sed -E 's/.*REASON=([^ ]+).*/\1/')
TABLE=$(echo "$decision" | sed -n '/^TABLE:/,$p' | tail -n +2)

if [ "$REASON" != "ramp" ]; then
    write_status "## Decision: ABORT full-17 full-pipeline ramp

Winner: $WINNER  mean Δ = $MEAN  reason: $REASON

Per-arm table:
\`\`\`
$TABLE
\`\`\`

clean A/B csv: \`$csv\`

The supreme A/B finished but no arm cleanly beats albania1
(mean Δ ≤ 0.005). Manual debugging needed before launching full-17."
    log "ramp aborted: $REASON"
    exit 0
fi

# ── Phase 4: launch zeus_full17.sh with the winning arm ────────────────
case "$WINNER" in
    base)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0" ;;
    rudy)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0" ;;
    hmc)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5" ;;
    rudy_hmc)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5" ;;
    *)
        write_status "## ERROR: unknown winner '$WINNER'"
        exit 1 ;;
esac

WRAPPER="$ROOT/full17_wrapper_$(date +%H%M%S).sh"
cat > "$WRAPPER" <<WRAP
#!/bin/bash
set -u
cd "$REPO"
$(for kv in $EXTRA_ENV; do echo "export $kv"; done)
exec bash scripts/zeus_full17.sh
WRAP
chmod +x "$WRAPPER"

write_status "## Phase 4: launching full-17 full-pipeline with arm='$WINNER'

clean A/B table (decision basis):
\`\`\`
$TABLE
\`\`\`

Env injected:
\`\`\`
$EXTRA_ENV
\`\`\`

Detached launch via zeus_run_detached.sh — monitor in next phase."

bash "$REPO/scripts/zeus_run_detached.sh" full17 bash "$WRAPPER" \
    > /dev/null 2>&1
full17_pid=$(cat "$ROOT/latest_full17/pid" 2>/dev/null)
full17_log="$ROOT/latest_full17/run.log"
log "full17 launched pid=$full17_pid"

# ── Phase 5: monitor full-17 ───────────────────────────────────────────
while kill -0 $full17_pid 2>/dev/null; do
    sleep 120
    f17_out=$(grep -E "OUT_DIR=" "$full17_log" 2>/dev/null | tail -1 | sed -E 's/OUT_DIR=//')
    if [ -z "$f17_out" ]; then
        f17_out=$(ls -dt /tmp/zeus_full17_* 2>/dev/null | head -1)
    fi
    n_done=0
    if [ -n "$f17_out" ] && [ -f "$f17_out/results.csv" ]; then
        n_done=$(($(wc -l < "$f17_out/results.csv") - 1))
    fi
    write_status "## Phase 5: full-17 full-pipeline RUNNING

- selected arm: $WINNER  (clean A/B mean Δ = $MEAN)
- full17 pid:   $full17_pid
- full17 log:   \`$full17_log\`
- benches done: $n_done / 17

clean A/B table:
\`\`\`
$TABLE
\`\`\`

Live tail ($(ts)):
\`\`\`
$(tail -25 "$full17_log" 2>/dev/null)
\`\`\`"
done

# ── Phase 6: final summary ─────────────────────────────────────────────
f17_out=$(grep -E "OUT_DIR=" "$full17_log" 2>/dev/null | tail -1 | sed -E 's/OUT_DIR=//')
if [ -z "$f17_out" ] || [ ! -d "$f17_out" ]; then
    f17_out=$(ls -dt /tmp/zeus_full17_* 2>/dev/null | head -1)
fi
final=""
if [ -n "$f17_out" ] && [ -f "$f17_out/results.csv" ]; then
    final=$(.venv/bin/python -u <<PY
import csv
rows = list(csv.DictReader(open('$f17_out/results.csv')))
ps = []
for r in rows:
    try: ps.append(float(r['proxy_cost']))
    except (ValueError, KeyError): pass
n = len(ps); m = sum(ps)/n if ps else float('nan')
print(f"  full-17 mean proxy: {m:.4f}  ({n}/17 valid)")
for r in rows:
    print(f"    {r['benchmark']:6}  proxy={r.get('proxy_cost','NA'):>8}  Δ={r.get('delta','NA')}")
PY
)
fi

write_status "## DONE — supreme autopilot complete

- supreme A/B csv: \`$csv\`
- selected arm:    $WINNER  (clean A/B mean Δ = $MEAN)
- full-17 csv:     \`$f17_out/results.csv\`

clean A/B table:
\`\`\`
$TABLE
\`\`\`

Final full-17 full-pipeline result:
\`\`\`
$final
\`\`\`

Compare against the albania1 baseline (mean proxy 0.9975) and
verified mean (1.0003) to evaluate the lift.

Files of interest:
- \`$csv\`
- \`$f17_out/results.csv\`
- \`$ROOT/latest_full17/run.log\`"

log "DONE"
