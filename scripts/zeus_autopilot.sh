#!/bin/bash
# zeus_autopilot — wait for the detached clean_ab to finish, pick the
# winning arm, then launch the full-17 sweep also detached.
#
# Runs as a child of init (launched via zeus_run_detached.sh) — survives
# SSH drops, claude crashes, laptop death.
#
# Writes a single human-readable status file at
#   ~/zeus_runs/AUTOPILOT_STATUS.md
# that the user (or a future claude) can read at any time to know the
# current state.
#
# Decision rule:
#   - Compute mean Δ per arm (lower=better, Δ vs verified baseline).
#   - Pick the arm with the lowest mean Δ.
#   - If two arms tie, prefer simpler (base > rudy > rudy_hmc).
#   - If ALL arms regress vs verified (Δ > 0.005), abort full-17 and
#     write a "needs human" message in STATUS.md.

set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
ROOT="${ZEUS_RUN_DIR:-$HOME/zeus_runs}"
STATUS="$ROOT/AUTOPILOT_STATUS.md"
mkdir -p "$ROOT"

# Helpers
ts()       { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()      { echo "[autopilot $(ts)] $*"; }
write_status() {
    cat > "$STATUS" <<EOF
# Zeus autopilot status — last update $(ts)

$1
EOF
}

# ── Phase 1: wait for clean_ab to finish ───────────────────────────────
clean_ab_dir="$ROOT/latest_clean_ab"
if [ ! -e "$clean_ab_dir" ]; then
    write_status "ERROR: no \`~/zeus_runs/latest_clean_ab\` symlink found.
clean_ab was probably never launched."
    log "no clean_ab dir; abort"
    exit 1
fi
clean_ab_dir=$(readlink -f "$clean_ab_dir")
pidfile="$clean_ab_dir/pid"
log_file="$clean_ab_dir/run.log"

if [ ! -f "$pidfile" ]; then
    write_status "ERROR: no pid file at \`$pidfile\`"
    exit 1
fi
clean_ab_pid=$(cat "$pidfile")
log "watching clean_ab pid=$clean_ab_pid log=$log_file"

write_status "## Phase 1: waiting on clean_ab

- pid: $clean_ab_pid (running)
- run dir: \`$clean_ab_dir\`
- log: \`$log_file\`
- started: $(stat -c '%y' "$pidfile" | cut -d. -f1)

Polling every 60 s. When clean_ab finishes, autopilot picks the winning
arm and launches the full-17 sweep automatically (also detached).

Live log tail:
\`\`\`
$(tail -10 "$log_file" 2>/dev/null)
\`\`\`"

while kill -0 "$clean_ab_pid" 2>/dev/null; do
    sleep 60
    # Update status with a fresh tail every 60 s so the user can see
    # progress without re-tailing the log themselves.
    write_status "## Phase 1: waiting on clean_ab

- pid: $clean_ab_pid (RUNNING)
- run dir: \`$clean_ab_dir\`
- log: \`$log_file\`

Live log tail ($(ts)):
\`\`\`
$(tail -20 "$log_file" 2>/dev/null)
\`\`\`"
done
log "clean_ab pid $clean_ab_pid no longer running"

# ── Phase 2: parse results.csv ─────────────────────────────────────────
# Find the actual results.csv (zeus_clean_ab.sh writes to /tmp/...)
csv=$(grep -E "out:" "$log_file" 2>/dev/null | head -1 | awk '{print $2}')
if [ -n "$csv" ] && [ -d "$csv" ]; then
    csv="$csv/results.csv"
else
    # Fallback: pick most recent /tmp/zeus_clean_ab_*
    csv=$(ls -dt /tmp/zeus_clean_ab_*/results.csv 2>/dev/null | head -1)
fi
log "results csv: $csv"

if [ ! -f "$csv" ]; then
    write_status "## ERROR — clean_ab finished but no results.csv

- clean_ab pid: $clean_ab_pid (exited)
- log: \`$log_file\`
- expected csv: \`$csv\`

The clean A/B finished but the results.csv was not found. The Hessian-
phase A/B may have failed before any data was logged. Manual inspection
needed: \`tail $log_file\` and \`ls /tmp/zeus_clean_ab_*\`"
    exit 1
fi

# Use python to pick the winning arm — safer than awk for the decision.
# Output: a single line "arm=<NAME> mean_delta=<F> verdict=<TEXT>"
decision=$(.venv/bin/python -u <<EOF
import csv
rows = [r for r in csv.DictReader(open('$csv'))
        if r.get('stage') == 'hess']
by_arm = {}
for r in rows:
    try: d = float(r['delta'])
    except (ValueError, KeyError): continue
    by_arm.setdefault(r['arm'], []).append(d)
mean = {a: sum(v)/len(v) for a, v in by_arm.items() if v}
if not mean:
    print("arm=ABORT mean_delta=NA verdict=no_arm_data")
else:
    # Lowest mean delta wins; on tie, prefer simpler arm.
    prefer = ['base', 'rudy', 'rudy_hmc']
    sorted_arms = sorted(mean.items(),
                          key=lambda kv: (kv[1], prefer.index(kv[0])
                                          if kv[0] in prefer else 99))
    best, best_d = sorted_arms[0]
    # If best arm STILL regresses vs verified by >0.005, that's worse
    # than albania1; abort full-17 ramp.
    if best_d > 0.005:
        print(f"arm=ABORT mean_delta={best_d:.4f} verdict=all_arms_regress")
    else:
        print(f"arm={best} mean_delta={best_d:.4f} verdict=ramp")
print("table:")
for a, d in sorted_arms:
    cells = ' '.join(f'{x:+.4f}' for x in by_arm[a])
    print(f"  {a:10} mean={d:+.4f} per-bench=[{cells}]")
EOF
)
log "decision: $decision"

best_arm=$(echo "$decision" | grep -E "^arm=" | head -1 | sed -E 's/arm=([^ ]+).*/\1/')
mean_d=$(echo "$decision" | grep -E "^arm=" | head -1 | sed -E 's/.*mean_delta=([^ ]+).*/\1/')
verdict=$(echo "$decision" | grep -E "^arm=" | head -1 | sed -E 's/.*verdict=([^ ]+).*/\1/')
table=$(echo "$decision" | tail -n +2)

if [ "$verdict" != "ramp" ]; then
    write_status "## Decision: ABORT full-17 ramp

clean_ab finished but the best arm ($best_arm, mean Δ $mean_d) does
NOT improve over the albania1 baseline (mean Δ > 0.005 vs verified).

Per-arm results:
\`\`\`
$table
\`\`\`

Full clean_ab CSV: \`$csv\`

Next steps require human review — likely debugging RUDY or HMC paths,
not a full-17 launch."
    log "verdict=$verdict; not ramping"
    exit 0
fi

# ── Phase 3: launch full-17 with the winning arm's env ─────────────────
EXTRA_ENV=""
case "$best_arm" in
    base)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=0 PLACER_V7_HESSIAN_RUDY=0 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0"
        ;;
    rudy)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=0 PLACER_V7_HESSIAN_HMC_TRAJ=0"
        ;;
    rudy_hmc)
        EXTRA_ENV="PLACER_V7_HESSIAN_CONG=1 PLACER_V7_HESSIAN_RUDY=1 PLACER_V7_HESSIAN_CONG_WEIGHT=0.5 PLACER_V7_HESSIAN_RUDY_MARGIN=4 PLACER_V7_HESSIAN_RUDY_MAX_WINDOW=64 PLACER_V7_HESSIAN_HMC_K=6 PLACER_V7_HESSIAN_HMC_TRAJ=16 PLACER_V7_HESSIAN_HMC_L=12 PLACER_V7_HESSIAN_HMC_STEP=0.5"
        ;;
    *)
        write_status "## ERROR — unknown winning arm '$best_arm'"
        exit 1
        ;;
esac

write_status "## Phase 2 → 3: launching full-17 with arm='$best_arm'

clean_ab table:
\`\`\`
$table
\`\`\`

Launching detached full-17 with env:
\`\`\`
$EXTRA_ENV
\`\`\`"

# Wrap the existing zeus_full17.sh script with env-override so the
# launched run uses the chosen arm's config. zeus_full17.sh hard-codes
# RUDY=1 in its body — \`env $EXTRA_ENV\` after it OVERRIDES via export.
# We use a tiny wrapper script for cleanliness.
WRAPPER="$ROOT/full17_wrapper.sh"
cat > "$WRAPPER" <<WRAP
#!/bin/bash
set -u
cd "$REPO"
# Override env to match autopilot decision.
$(for kv in $EXTRA_ENV; do echo "export $kv"; done)
exec bash scripts/zeus_full17.sh
WRAP
chmod +x "$WRAPPER"

bash "$REPO/scripts/zeus_run_detached.sh" full17 bash "$WRAPPER"
full17_pid=$(cat "$ROOT/latest_full17/pid" 2>/dev/null || echo "?")
full17_log="$ROOT/latest_full17/run.log"
log "launched full17 pid=$full17_pid"

# ── Phase 4: monitor full-17, write status periodically ────────────────
while kill -0 "$full17_pid" 2>/dev/null; do
    sleep 120
    # Try to get a live row count from results.csv
    f17_csv=$(grep -E "out:" "$full17_log" 2>/dev/null | head -1 | sed -E 's/.*OUT_DIR=//;s/.*out: //' | awk '{print $1}')
    if [ -z "$f17_csv" ] || [ ! -d "$f17_csv" ]; then
        f17_csv=$(ls -dt /tmp/zeus_full17_*/ 2>/dev/null | head -1)
    fi
    if [ -n "$f17_csv" ] && [ -f "$f17_csv/results.csv" ]; then
        n_done=$(($(wc -l < "$f17_csv/results.csv") - 1))
    else
        n_done="?"
    fi
    write_status "## Phase 3 (running): full-17 sweep

- selected arm: $best_arm  (mean Δ on 3-bench A/B = $mean_d)
- full17 pid: $full17_pid (RUNNING)
- full17 log: \`$full17_log\`
- benches completed: $n_done / 17

clean_ab table (decision basis):
\`\`\`
$table
\`\`\`

Live log tail ($(ts)):
\`\`\`
$(tail -25 "$full17_log" 2>/dev/null)
\`\`\`"
done

# ── Phase 5: final summary ─────────────────────────────────────────────
f17_csv=$(grep -E "out:" "$full17_log" 2>/dev/null | head -1 | sed -E 's/.*OUT_DIR=//;s/.*out: //' | awk '{print $1}')
if [ -z "$f17_csv" ] || [ ! -d "$f17_csv" ]; then
    f17_csv=$(ls -dt /tmp/zeus_full17_*/ 2>/dev/null | head -1)
fi
final_summary=""
if [ -n "$f17_csv" ] && [ -f "$f17_csv/results.csv" ]; then
    final_summary=$(.venv/bin/python -u <<EOF
import csv
rows = list(csv.DictReader(open('$f17_csv/results.csv')))
ps = []
for r in rows:
    try: ps.append(float(r['proxy_cost']))
    except (ValueError, KeyError): pass
n = len(ps)
m = sum(ps)/n if ps else float('nan')
print(f"  full-17: {n}/17 valid, mean proxy = {m:.4f}")
for r in rows:
    p = r.get('proxy_cost', 'NA')
    d = r.get('delta', 'NA')
    print(f"    {r['benchmark']:6} proxy={p:8} Δ={d}")
EOF
)
fi

write_status "## DONE — full-17 sweep finished

- selected arm: $best_arm  (mean Δ on 3-bench A/B = $mean_d)
- full17 csv: \`$f17_csv/results.csv\`

clean_ab table:
\`\`\`
$table
\`\`\`

Final full-17 result:
\`\`\`
$final_summary
\`\`\`

Inspect: \`bash $REPO/scripts/zeus_status.sh full17\`"
log "autopilot DONE"
