#!/bin/bash
# Pretty-print all albania2 experiment results.

cd "$(dirname "$0")/.."
echo "=================================================================="
echo "                      albania2 results summary"
echo "=================================================================="
echo ""

show_log() {
    local label="$1"
    local pattern="$2"
    local out=$(ls -td $pattern 2>/dev/null | head -1)
    if [ -z "$out" ]; then
        echo "  [$label] no run found"
        return
    fi
    echo "  [$label]   $out"
    if grep -q "^DONE$" "$out/sweep.log" 2>/dev/null; then
        echo "    status: DONE"
    else
        echo "    status: RUNNING"
        local active=$(ls "$out"/*.log 2>/dev/null | tail -1)
        if [ -n "$active" ]; then
            echo "    last log line: $(tail -1 "$active" 2>/dev/null)"
        fi
    fi
    if [ -f "$out/results.csv" ]; then
        echo "    --- results.csv ---"
        cat "$out/results.csv" | sed 's/^/      /'
    fi
    if [ -f "$out/sweep.log" ]; then
        echo "    --- summary tail ---"
        tail -10 "$out/sweep.log" | grep -E "mean|Δ|proxy=|verified" | sed 's/^/      /'
    fi
    echo ""
}

show_log "stage-1 spectral A/B  " "/tmp/albania2_spectral_ab_2*"
show_log "stage-2 kdim+combined " "/tmp/albania2_kdim_combined_ab_2*"
show_log "stage-3 gain sweep    " "/tmp/albania2_spectral_gain_sweep_2*"
show_log "Bet B B2B A/B         " "/tmp/albania2_b2b_ab_2*"
show_log "Bet A Phase 0 A/B     " "/tmp/albania2_phase0_ab_2*"

echo "------------------------------------------------------------------"
echo "                     albania1 reference (winning sweep)"
echo "------------------------------------------------------------------"
echo "  17/17 VALID, mean=0.9975, verified=1.0003, Δ=-0.0028"
echo "  see: research/sweeps/albania1_full17/results.csv"
echo ""

echo "------------------------------------------------------------------"
echo "                   active processes & battery"
echo "------------------------------------------------------------------"
ps aux | grep -E "evaluate|placer|caffeinate" | grep -v grep | head -10
echo ""
pmset -g batt 2>/dev/null | head -3
