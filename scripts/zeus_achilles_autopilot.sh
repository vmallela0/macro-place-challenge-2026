#!/bin/bash
# zeus_achilles_autopilot — orchestrates the achilles bet portfolio.
#
# Stage A: launch zeus_achilles_ab.sh as a child (3-bench screening).
# Stage B: poll until child exits, parse results, pick top-N best arms.
# Stage C: launch zeus_achilles_full17.sh with the chosen arms.
# Stage D: poll until full-17 done; write final ranking.
#
# All output to ~/zeus_runs/ACHILLES_STATUS.md (updated every loop).
# Lives as a detached PPID=1 process via zeus_run_detached.sh.

set -u
cd "$(dirname "$0")/.."

STATUS="$HOME/zeus_runs/ACHILLES_STATUS.md"
SCREEN_OUT_FILE="$HOME/zeus_runs/.achilles_screen_out"
FULL17_OUT_FILE="$HOME/zeus_runs/.achilles_full17_out"

LOG_DIR="$HOME/zeus_runs"
TOP_N="${TOP_N:-4}"
SCREEN_BENCHES="${SCREEN_BENCHES:-ibm06 ibm12 ibm15}"
FULL17_BENCHES="ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18"
ALL_ARMS="${ALL_ARMS:-baseline yoshida replica l1_cong linf_cong nesterov dmc jko free_energy smc rg catastrophe neb yoshida_replica dmc_smc rg_nesterov hmc_full}"

# ────────────────────────────────────────────────────────────────────────
log_status() {
    local stage="$1" message="$2"
    {
        echo "# zeus achilles autopilot status — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
        echo "## $stage"
        echo ""
        echo "$message"
    } > "$STATUS"
}

log_status "STARTING" "Bringing up Stage A (3-bench screen)..."

# ────────────────────────────────────────────────────────────────────────
# Stage A: launch screening.
# ────────────────────────────────────────────────────────────────────────
SCREEN_LOG="$LOG_DIR/achilles_screen_$(date +%Y%m%d_%H%M%S).log"
ARMS="$ALL_ARMS" BENCHES="$SCREEN_BENCHES" \
    setsid nohup bash scripts/zeus_achilles_ab.sh > "$SCREEN_LOG" 2>&1 < /dev/null &
SCREEN_PID=$!
disown 2>/dev/null || true
echo "$SCREEN_PID" > "$SCREEN_OUT_FILE"
echo "[autopilot $(date -u +%Y-%m-%dT%H:%M:%SZ)] launched screen pid=$SCREEN_PID log=$SCREEN_LOG"

log_status "Stage A: SCREEN RUNNING" "
- pid: $SCREEN_PID
- log: \`$SCREEN_LOG\`
- arms ($(echo $ALL_ARMS | wc -w)): \`$ALL_ARMS\`
- screen benches: \`$SCREEN_BENCHES\`
- expected wall: ~3-5h depending on wave width
"

# ────────────────────────────────────────────────────────────────────────
# Stage B: poll until child exits, then parse.
# ────────────────────────────────────────────────────────────────────────
while kill -0 "$SCREEN_PID" 2>/dev/null; do
    # Update status with tail of screen log.
    n_arms=$(echo "$ALL_ARMS" | wc -w)
    n_benches=$(echo "$SCREEN_BENCHES" | wc -w)
    n_expected=$((n_arms * n_benches))
    # Find current out dir from latest_achilles_ab symlink.
    OUT_DIR=$(readlink -f "$HOME/zeus_runs/latest_achilles_ab" 2>/dev/null || echo "")
    n_done=0
    if [ -f "$OUT_DIR/results.csv" ]; then
        n_done=$(awk -F, 'NR>1' "$OUT_DIR/results.csv" | wc -l)
    fi
    tail=$(tail -25 "$SCREEN_LOG" 2>/dev/null)
    log_status "Stage A: SCREEN RUNNING ($n_done/$n_expected)" "
- pid: $SCREEN_PID
- log: \`$SCREEN_LOG\`
- out: \`$OUT_DIR\`
- expected jobs: $n_expected
- jobs done so far: $n_done

### Live tail:
\`\`\`
$tail
\`\`\`
"
    sleep 60
done

echo "[autopilot $(date -u +%Y-%m-%dT%H:%M:%SZ)] screen exited"

# ────────────────────────────────────────────────────────────────────────
# Stage B parse: rank arms by mean Δ across the 3 screening benches.
# ────────────────────────────────────────────────────────────────────────
OUT_DIR=$(readlink -f "$HOME/zeus_runs/latest_achilles_ab" 2>/dev/null || echo "")
if [ ! -f "$OUT_DIR/results.csv" ]; then
    log_status "Stage B: PARSE FAILED" "Could not find results.csv at $OUT_DIR. Halting."
    exit 1
fi

# Rank arms by their mean signed Δ. Lower (more negative) is better.
RANKING=$(awk -F, '
    NR>1 && $1=="ab" {
        arm=$2; bench=$3; delta=$11
        if (delta == "" || delta == "NA") next
        # Strip leading "+" for awk number parsing
        gsub(/^\+/, "", delta)
        n[arm]++
        s[arm] += delta
    }
    END {
        for (a in n) {
            printf "%.6f %d %s\n", s[a]/n[a], n[a], a
        }
    }
' "$OUT_DIR/results.csv" | sort -n)

TOP_ARMS=$(echo "$RANKING" | head -n "$TOP_N" | awk '{print $3}' | tr '\n' ' ')
TOP_TABLE=$(echo "$RANKING" | awk '{printf "  %-22s  mean Δ=%+.4f  n=%d\n", $3, $1, $2}')

log_status "Stage B: PARSED + READY TO RAMP" "
### Ranking by mean Δ on $SCREEN_BENCHES:
\`\`\`
$TOP_TABLE
\`\`\`

### Top $TOP_N selected for full-17:
\`$TOP_ARMS\`
"

# ────────────────────────────────────────────────────────────────────────
# Stage C: launch full-17 with top arms.
# ────────────────────────────────────────────────────────────────────────
FULL17_LOG="$LOG_DIR/achilles_full17_$(date +%Y%m%d_%H%M%S).log"
ARMS="$TOP_ARMS" BENCHES="$FULL17_BENCHES" \
    setsid nohup bash scripts/zeus_achilles_ab.sh > "$FULL17_LOG" 2>&1 < /dev/null &
FULL17_PID=$!
disown 2>/dev/null || true
echo "$FULL17_PID" > "$FULL17_OUT_FILE"
echo "[autopilot $(date -u +%Y-%m-%dT%H:%M:%SZ)] launched full17 pid=$FULL17_PID log=$FULL17_LOG"

# ────────────────────────────────────────────────────────────────────────
# Stage D: poll until full-17 done, then final report.
# ────────────────────────────────────────────────────────────────────────
while kill -0 "$FULL17_PID" 2>/dev/null; do
    n_arms=$(echo "$TOP_ARMS" | wc -w)
    n_expected=$((n_arms * 17))
    OUT_DIR_F=$(readlink -f "$HOME/zeus_runs/latest_achilles_ab" 2>/dev/null || echo "")
    n_done=0
    if [ -f "$OUT_DIR_F/results.csv" ]; then
        n_done=$(awk -F, 'NR>1' "$OUT_DIR_F/results.csv" | wc -l)
    fi
    tail=$(tail -25 "$FULL17_LOG" 2>/dev/null)
    log_status "Stage C: FULL-17 RUNNING ($n_done/$n_expected)" "
### Selected top-$TOP_N arms from screen:
\`$TOP_ARMS\`

### Full-17 progress:
- pid: $FULL17_PID
- log: \`$FULL17_LOG\`
- out: \`$OUT_DIR_F\`
- expected: $n_expected jobs
- done:     $n_done

### Live tail:
\`\`\`
$tail
\`\`\`
"
    sleep 120
done

echo "[autopilot $(date -u +%Y-%m-%dT%H:%M:%SZ)] full17 exited"

# Final report.
OUT_DIR_F=$(readlink -f "$HOME/zeus_runs/latest_achilles_ab" 2>/dev/null || echo "")
FINAL_RANKING=$(awk -F, '
    NR>1 && $1=="ab" {
        arm=$2; delta=$11
        if (delta == "" || delta == "NA") next
        gsub(/^\+/, "", delta)
        n[arm]++
        s[arm] += delta
    }
    END {
        for (a in n) {
            printf "%.6f %d %s\n", s[a]/n[a], n[a], a
        }
    }
' "$OUT_DIR_F/results.csv" | sort -n)
FINAL_TABLE=$(echo "$FINAL_RANKING" | awk '{printf "  %-22s  mean Δ=%+.4f  n=%d\n", $3, $1, $2}')

log_status "Stage D: COMPLETE" "
### Final full-17 ranking (mean Δ vs verified):
\`\`\`
$FINAL_TABLE
\`\`\`

### Out dir: \`$OUT_DIR_F\`
"

echo "[autopilot $(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE"
