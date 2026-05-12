#!/bin/bash
# zeus_analyze_supreme — rich analysis of a supreme_ab results.csv.
#
# Usage: bash scripts/zeus_analyze_supreme.sh [/tmp/zeus_supreme_<ts>]
# (with no arg: picks the most recent /tmp/zeus_supreme_*)
#
# Prints:
#   1. Per-arm × per-bench Δ matrix
#   2. Per-arm mean Δ over all 17, and over the 6 historical regression
#      benches (ibm01, ibm10, ibm13, ibm14, ibm15, ibm16)
#   3. 2×2 factorial decomposition: does the (rudy_hmc - base) lift
#      decompose into a RUDY main effect + HMC main effect + interaction?
#   4. Per-bench winner (which arm gave lowest proxy on each bench)

set -u
OUT="${1:-}"
if [ -z "$OUT" ]; then
    OUT=$(ls -dt /tmp/zeus_supreme_*/ 2>/dev/null | head -1)
fi
if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
    echo "usage: $0 [/tmp/zeus_supreme_<ts>]" >&2; exit 1
fi
CSV="$OUT/results.csv"
if [ ! -f "$CSV" ]; then
    echo "no $CSV"; exit 1
fi
echo "Analyzing: $OUT"
echo "Results CSV: $CSV"
echo
.venv/bin/python -u <<EOF
import csv
import math
rows = [r for r in csv.DictReader(open('$CSV')) if r['stage']=='hess']
if not rows:
    print('NO Hessian rows in $CSV — Step 2 may not be done yet')
    raise SystemExit(0)

# albania1 verified per-bench baselines for Δ math.
VERIFIED = {
    'ibm01':0.7653,'ibm02':0.9482,'ibm03':0.9166,'ibm04':0.9287,
    'ibm06':1.0546,'ibm07':1.0324,'ibm08':1.0291,'ibm09':0.7628,
    'ibm10':0.9492,'ibm11':0.8013,'ibm12':1.1557,'ibm13':0.8757,
    'ibm14':1.1070,'ibm15':1.0835,'ibm16':1.0435,'ibm17':1.2813,
    'ibm18':1.2697,
}
# Historical regressions in albania1 0.9975 sweep.
REGRESS_6 = {'ibm01','ibm10','ibm13','ibm14','ibm15','ibm16'}

arms = sorted(set(r['arm'] for r in rows),
              key=lambda a: ['base','rudy','hmc','rudy_hmc'].index(a)
              if a in ['base','rudy','hmc','rudy_hmc'] else 99)
benches = sorted(set(r['benchmark'] for r in rows))

print('=' * 100)
print('1. Δ vs verified (lower = better)')
print('=' * 100)
hdr = f"{'arm':10}" + ' '.join(f'{b:>8}' for b in benches) + '   mean   regr_mean'
print(hdr)
for a in arms:
    cells, all_d, regr_d = [], [], []
    for b in benches:
        m = [r for r in rows if r['arm']==a and r['benchmark']==b]
        if not m or m[0].get('delta','?') in ('?','NA',''):
            cells.append('     N/A')
            continue
        try: d = float(m[0]['delta'])
        except ValueError: cells.append('     N/A'); continue
        cells.append(f"{d:>+8.4f}")
        all_d.append(d)
        if b in REGRESS_6: regr_d.append(d)
    m1 = sum(all_d)/len(all_d) if all_d else float('nan')
    m2 = sum(regr_d)/len(regr_d) if regr_d else float('nan')
    print(f"{a:10}" + ' '.join(cells) + f"   {m1:+.4f}   {m2:+.4f}")

print()
print('=' * 100)
print('2. Per-arm mean Δ (overall + 6-regress-bench subset)')
print('=' * 100)
print(f"{'arm':10}  mean_all   mean_regr6   n_valid_all   n_valid_regr")
for a in arms:
    rs = [r for r in rows if r['arm']==a]
    all_d, regr_d = [], []
    n_v_all, n_v_regr = 0, 0
    for r in rs:
        try: d = float(r['delta'])
        except (ValueError, KeyError): continue
        all_d.append(d); n_v_all += 1
        if r['benchmark'] in REGRESS_6:
            regr_d.append(d); n_v_regr += 1
    m1 = sum(all_d)/len(all_d) if all_d else float('nan')
    m2 = sum(regr_d)/len(regr_d) if regr_d else float('nan')
    print(f"{a:10}  {m1:+.4f}    {m2:+.4f}      {n_v_all:>2}/{len(benches)}        {n_v_regr:>1}/{len(REGRESS_6)}")

print()
print('=' * 100)
print('3. 2x2 factorial decomposition: main effects + interaction')
print('=' * 100)
if all(a in arms for a in ['base','rudy','hmc','rudy_hmc']):
    # Compute per-bench effects, then average.
    def cell(arm, b):
        m = [r for r in rows if r['arm']==arm and r['benchmark']==b]
        if not m: return None
        try: return float(m[0]['delta'])
        except (ValueError, KeyError): return None

    eff_rudy, eff_hmc, eff_interact = [], [], []
    for b in benches:
        b_ = cell('base',b); r_ = cell('rudy',b)
        h_ = cell('hmc',b);  rh = cell('rudy_hmc',b)
        if None in (b_, r_, h_, rh): continue
        # Main effect of RUDY: ((rudy - base) + (rudy_hmc - hmc)) / 2
        e_r = ((r_ - b_) + (rh - h_)) / 2.0
        # Main effect of HMC: ((hmc - base) + (rudy_hmc - rudy)) / 2
        e_h = ((h_ - b_) + (rh - r_)) / 2.0
        # Interaction: (rudy_hmc - rudy) - (hmc - base)
        e_i = (rh - r_) - (h_ - b_)
        eff_rudy.append(e_r); eff_hmc.append(e_h); eff_interact.append(e_i)
    if eff_rudy:
        print(f"  mean RUDY main effect:  {sum(eff_rudy)/len(eff_rudy):+.4f}  (n={len(eff_rudy)})")
        print(f"  mean HMC  main effect:  {sum(eff_hmc)/len(eff_hmc):+.4f}  (n={len(eff_hmc)})")
        print(f"  mean RUDY×HMC interact: {sum(eff_interact)/len(eff_interact):+.4f}  (n={len(eff_interact)})")
        print()
        print('  Interpretation:')
        print('    main effect of X < 0  → X helps proxy (Δ vs verified shrinks)')
        print('    interaction < 0       → combining RUDY+HMC is super-additive')
        print('    interaction ~ 0       → effects add linearly')
        print('    interaction > 0       → combination underperforms additive sum')
    else:
        print('  (no benches had all 4 arms complete)')
else:
    print(f'  not a full 4-arm factorial — arms present: {arms}')

print()
print('=' * 100)
print('4. Per-bench winner (best arm by lowest delta)')
print('=' * 100)
print(f"{'bench':8} {'winner':10} {'Δ_win':>8} {'2nd':10} {'Δ_2nd':>8} {'gap':>8}")
for b in benches:
    cells_for_b = []
    for a in arms:
        m = [r for r in rows if r['arm']==a and r['benchmark']==b]
        if not m: continue
        try: d = float(m[0]['delta']); cells_for_b.append((a, d))
        except (ValueError, KeyError): continue
    if not cells_for_b:
        print(f"{b:8} (no valid runs)"); continue
    cells_for_b.sort(key=lambda t: t[1])
    w_arm, w_d = cells_for_b[0]
    if len(cells_for_b) > 1:
        r2_arm, r2_d = cells_for_b[1]
        print(f"{b:8} {w_arm:10} {w_d:>+8.4f} {r2_arm:10} {r2_d:>+8.4f} {r2_d-w_d:>+8.4f}")
    else:
        print(f"{b:8} {w_arm:10} {w_d:>+8.4f} (only 1 valid)")
EOF
