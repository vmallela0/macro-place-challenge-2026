#!/bin/bash
# Quick analyzer for a zeus_clean_ab output directory.
# Usage: bash scripts/zeus_analyze.sh /tmp/zeus_clean_ab_YYYYmmdd_HHMMSS

set -u
OUT="${1:-}"
if [ -z "$OUT" ]; then
  OUT=$(ls -dt /tmp/zeus_clean_ab_* 2>/dev/null | head -1)
fi
if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
  echo "usage: $0 <dir>" >&2; exit 1
fi
echo "Analyzing: $OUT"
echo
echo "=== sweep log ==="
cat "$OUT/sweep.log"
echo
echo "=== Hessian phase details per arm/bench ==="
for arm_log in $OUT/base_*.log $OUT/rudy_*.log $OUT/rudy_hmc_*.log; do
  [ -f "$arm_log" ] || continue
  base=$(basename "$arm_log" .log)
  echo "--- $base ---"
  grep -E "LOAD_POST_LAP|RUDY|HESSIAN|lambda_min|n_pairs|hessian iter|HESSIAN WIN|hessian: no|^proxy=|surrogate-improving|kdim Newton|hmc_t|feasibility" "$arm_log" 2>/dev/null | head -40
  echo
done
echo "=== summary table ==="
.venv/bin/python -u <<EOF
import csv
rows = [r for r in csv.DictReader(open('$OUT/results.csv'))
        if r['stage'] == 'hess']
if not rows:
    print("NO Hessian-stage results yet")
else:
    benches = sorted(set(r['benchmark'] for r in rows))
    arms = sorted(set(r['arm'] for r in rows))
    print('       ' + '   '.join(f'{b:>10}' for b in benches) + '     mean Δ')
    for a in arms:
        cells, ds = [], []
        for b in benches:
            m = [r for r in rows if r['arm']==a and r['benchmark']==b]
            if not m:
                cells.append('       N/A')
            else:
                r = m[0]
                cells.append(f"{r['delta']:>+10}")
                try: ds.append(float(r['delta']))
                except: pass
        md = sum(ds)/len(ds) if ds else float('nan')
        print(f"{a:7}" + '   '.join(cells) + f'     {md:+.4f}')
EOF
