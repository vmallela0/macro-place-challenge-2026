#!/bin/bash
# v8 top-level launcher.
# Modes:
#   --tests           Run all v8 math tests (pure-numpy, no torch needed)
#   --parity          Run cross-platform parity tests (needs torch)
#   --smoke           Single ibm15 smoke with v8 phases enabled
#   --full-sweep      Launch the full 17-bench sweep + per-bench auto-push
#                     (mirrors slj2/run_pipeline.sh: detached daemon + watcher)
#   --phase-a-gate    Run ibm15 with only ARC enabled
#   --phase-b-gate    Run ibm15 with ARC + PT enabled
#   --phase-c-gate    Run ibm15 with ARC + PT + Riemannian enabled (full)
#
# Env vars passed through:
#   PLACER_SLJ2_POOL  — defaults 8 on this pod
#   PLACER_V8_ARC     — defaults 1 in --smoke / --full-sweep
#   PLACER_V8_REPLICA
#   PLACER_V8_RIEMANNIAN
set -eu

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

mode="${1:-}"
shift || true

V8_DIR="$REPO/submissions/vmallela_v8"
TESTS_DIR="$V8_DIR/tests"
RUNLOG="$V8_DIR/RUNLOG.md"

stamp() { date -u +%FT%TZ; }
log_runlog() { echo "[$(stamp)] $*" >> "$RUNLOG"; }

run_tests() {
  log_runlog "tests start: $1"
  echo "running $1 ..."
  if .venv/bin/python "$1"; then
    log_runlog "tests PASS: $1"
    return 0
  else
    rc=$?
    log_runlog "tests FAIL: $1 (rc=$rc)"
    return $rc
  fi
}

run_smoke_with_env() {
  local label="$1"; shift
  log_runlog "$label start (env: $*)"
  TMP_OUT="/tmp/v8_${label}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$TMP_OUT"
  echo "$TMP_OUT" > "$V8_DIR/last_smoke_dir.txt"

  export PLACER_SLJ2_POOL=${PLACER_SLJ2_POOL:-8}
  export PYTHONHASHSEED=42
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  # v7 outer scaffolding (mirrors slj2 smoke env)
  export PLACER_TOTAL_BUDGET=2300
  export PLACER_V6_WORKERS=1
  export PLACER_V6_GPU_WORKERS=1
  export PLACER_V6_CONSENSUS=0
  export PLACER_SA_T0=0.00005
  export PLACER_ESC_HARD_DESTROY=80
  export PLACER_V7_LAPLACIAN=1
  export PLACER_V7_LAPLACIAN_PASSES=2
  export PLACER_V7_LAPLACIAN_BUDGET_FRAC=0.04
  export PLACER_V7_BASIN_HOPS=0
  export PLACER_V7_HESSIAN=1
  export PLACER_V7_HESSIAN_STEPS="0.02,-0.02,0.05,-0.05"
  export PLACER_V7_HESSIAN_BUDGET=1000
  export PLACER_V7_HESSIAN_LANCZOS=50
  export PLACER_V7_HESSIAN_MAX_ITERS=1
  # phase env gates — set by caller via "$@"
  for kv in "$@"; do
    export "$kv"
  done

  t0=$(date +%s)
  .venv/bin/python -m macro_place.evaluate \
    submissions/vmallela_v8/placer.py --benchmark ibm15 \
    > "$TMP_OUT/smoke.log" 2>&1
  rc=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  line=$(grep -E "^proxy=" "$TMP_OUT/smoke.log" | tail -1)
  proxy=$(echo "$line" | sed -E 's/.*proxy=([0-9.]+).*/\1/' | head -c 12)
  if echo "$line" | grep -q "VALID"; then
    overlaps=0
  else
    overlaps=$(grep -E "overlaps=" "$TMP_OUT/smoke.log" | tail -1 | sed -E 's/.*overlaps=([0-9]+).*/\1/' | head -c 8)
    [ -z "$overlaps" ] && overlaps="?"
  fi
  log_runlog "$label result: proxy=$proxy overlaps=$overlaps wall=${elapsed}s rc=$rc log=$TMP_OUT/smoke.log"
  echo "$label: proxy=$proxy overlaps=$overlaps wall=${elapsed}s rc=$rc"
  return $rc
}

case "$mode" in
  --tests)
    run_tests "$TESTS_DIR/test_arc_math.py"
    run_tests "$TESTS_DIR/test_replica_exchange_math.py"
    run_tests "$TESTS_DIR/test_riemannian_math.py"
    ;;
  --parity)
    run_tests "$TESTS_DIR/test_platform_parity.py"
    ;;
  --smoke)
    run_smoke_with_env "smoke_full" \
      PLACER_V8_ARC=1 PLACER_V8_REPLICA=1 PLACER_V8_RIEMANNIAN=1
    ;;
  --phase-a-gate)
    run_smoke_with_env "phaseA" \
      PLACER_V8_ARC=1 PLACER_V8_REPLICA=0 PLACER_V8_RIEMANNIAN=0
    ;;
  --phase-b-gate)
    run_smoke_with_env "phaseB" \
      PLACER_V8_ARC=1 PLACER_V8_REPLICA=1 PLACER_V8_RIEMANNIAN=0
    ;;
  --phase-c-gate)
    run_smoke_with_env "phaseC" \
      PLACER_V8_ARC=1 PLACER_V8_REPLICA=1 PLACER_V8_RIEMANNIAN=1
    ;;
  --full-sweep)
    branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$branch" != "v8" ]; then
      log_runlog "full-sweep ABORTED: must be on branch 'v8' (currently '$branch')"
      echo "must be on the 'v8' branch (currently '$branch')" >&2
      exit 1
    fi
    if pgrep -af "(v8_full_sweep|slj2_full_sweep)\.sh" >/dev/null; then
      echo "a sweep is already running:" >&2
      pgrep -af "(v8_full_sweep|slj2_full_sweep)\.sh" >&2
      exit 1
    fi

    export PLACER_SLJ2_POOL=${PLACER_SLJ2_POOL:-8}
    log_runlog "full-sweep launch: pool=$PLACER_SLJ2_POOL arc=${PLACER_V8_ARC:-1} pt=${PLACER_V8_REPLICA:-1} riem=${PLACER_V8_RIEMANNIAN:-1}"
    TMP="/tmp/v8_pre_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$TMP"
    setsid nohup bash scripts/v8_full_sweep.sh \
      > "${TMP}/launcher.log" 2>&1 < /dev/null &
    SWEEP_PID=$!
    echo "sweep pid: $SWEEP_PID"

    ACTUAL_DIR=""
    for _ in $(seq 1 60); do
      d=$(ls -1dt /tmp/v8_sweep_* 2>/dev/null | head -1 || true)
      if [ -n "$d" ] && [ -f "$d/sweep.log" ]; then
        ACTUAL_DIR="$d"
        break
      fi
      sleep 1
    done
    if [ -z "$ACTUAL_DIR" ]; then
      log_runlog "full-sweep ABORTED: could not find sweep dir after 60s"
      echo "could not find sweep dir after 60s — aborting" >&2
      exit 1
    fi
    echo "$ACTUAL_DIR" > "$V8_DIR/sweep_dir.txt"
    log_runlog "full-sweep dir: $ACTUAL_DIR"
    echo "sweep dir: $ACTUAL_DIR"

    # Per-bench watcher (lsj/watcher.sh; data_dir = submissions/vmallela_v8)
    setsid nohup bash lsj/watcher.sh "$ACTUAL_DIR" "submissions/vmallela_v8" \
      > /dev/null 2>&1 < /dev/null &
    WATCHER_PID=$!
    echo "watcher pid: $WATCHER_PID"

    echo
    echo "Both processes detached. Tail logs:"
    echo "  tail -f $ACTUAL_DIR/sweep.log"
    echo "  tail -f $V8_DIR/watcher.log"
    ;;
  *)
    cat <<EOF
usage: $0 <mode>

Modes:
  --tests           Run all v8 math tests
  --parity          Run cross-platform parity tests
  --smoke           Single ibm15 smoke with all v8 phases enabled
  --phase-a-gate    Single ibm15 with only ARC enabled
  --phase-b-gate    Single ibm15 with ARC + PT enabled
  --phase-c-gate    Full v8 ibm15 (ARC + PT + Riemannian)
  --full-sweep      Detached 17-bench sweep + per-bench auto-push
EOF
    exit 1
    ;;
esac
