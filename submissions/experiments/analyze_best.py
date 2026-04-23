"""Pick best result per benchmark across overnight phases and compare to baselines."""
import re
from pathlib import Path

ROOT = Path(__file__).parent / "results"
OVERNIGHT_DIRS = [
    ROOT / "overnight",
    ROOT / "overnight_v3",
    ROOT / "overnight_v3_long",
    ROOT / "targeted_v80",
]

SA = {"ibm01": 1.3166, "ibm02": 1.9072, "ibm03": 1.7401, "ibm04": 1.5037,
      "ibm06": 2.5057, "ibm07": 2.0229, "ibm08": 1.9239, "ibm09": 1.3875,
      "ibm10": 2.1108, "ibm11": 1.7111, "ibm12": 2.8261, "ibm13": 1.9141,
      "ibm14": 2.2750, "ibm15": 2.3000, "ibm16": 2.2337, "ibm17": 3.6726,
      "ibm18": 2.7755}
RE = {"ibm01": 0.9976, "ibm02": 1.8370, "ibm03": 1.3222, "ibm04": 1.3024,
      "ibm06": 1.6187, "ibm07": 1.4633, "ibm08": 1.4285, "ibm09": 1.1194,
      "ibm10": 1.5009, "ibm11": 1.1774, "ibm12": 1.7261, "ibm13": 1.3355,
      "ibm14": 1.5436, "ibm15": 1.5159, "ibm16": 1.4780, "ibm17": 1.6446,
      "ibm18": 1.7722}
BENCHES = list(SA.keys())


def parse_log(f):
    try:
        text = f.read_text()
    except Exception:
        return None, None, True
    m = re.search(r"^proxy=([0-9.]+)", text, re.M)
    t = re.search(r"VALID\s+\[([0-9.]+)s\]", text)
    invalid = "INVALID" in text
    return (float(m.group(1)) if m else None,
            float(t.group(1)) if t else None,
            invalid)


def main():
    # Build table: benchmark -> list of (phase, cost, time, invalid)
    by_bench = {b: [] for b in BENCHES}
    for d in OVERNIGHT_DIRS:
        if not d.exists():
            continue
        phase_name = d.name
        for f in sorted(d.glob("ibm*.log")):
            bench = f.stem
            cost, runtime, invalid = parse_log(f)
            if cost is not None:
                by_bench[bench].append((phase_name, cost, runtime, invalid))

    print(f"{'Bench':>8}  {'Best':>8}  {'Phase':>22}  {'SA':>8}  {'RePlAce':>8}  "
          f"{'vs SA':>7}  {'vs RePlAce':>10}  {'Time':>6}")
    print("-" * 100)
    ours_vals = []
    sa_vals = []
    re_vals = []
    n_done = 0
    for bench in BENCHES:
        results = by_bench.get(bench, [])
        valid = [r for r in results if not r[3]]
        if not valid:
            print(f"{bench:>8}  RUNNING")
            continue
        # Pick min cost
        best = min(valid, key=lambda r: r[1])
        phase, cost, runtime, _ = best
        n_done += 1
        sa = SA[bench]
        re_ = RE[bench]
        vs_sa = (sa - cost) / sa * 100
        vs_re = (re_ - cost) / re_ * 100
        print(f"{bench:>8}  {cost:>8.4f}  {phase:>22}  {sa:>8.4f}  {re_:>8.4f}  "
              f"{vs_sa:>+6.1f}%  {vs_re:>+9.1f}%  {runtime:>5.0f}s")
        ours_vals.append(cost)
        sa_vals.append(sa)
        re_vals.append(re_)

    if ours_vals:
        print("-" * 100)
        avg_ours = sum(ours_vals) / len(ours_vals)
        avg_sa = sum(sa_vals) / len(sa_vals)
        avg_re = sum(re_vals) / len(re_vals)
        print(f"{'AVG':>8}  {avg_ours:>8.4f}  {'':>22}  {avg_sa:>8.4f}  {avg_re:>8.4f}  "
              f"{(avg_sa - avg_ours) / avg_sa * 100:>+6.1f}%  "
              f"{(avg_re - avg_ours) / avg_re * 100:>+9.1f}%")
        print(f"\nBenchmarks complete: {n_done}/17")
        print(f"vs prior vmallela 1.4156: {(1.4156 - avg_ours) / 1.4156 * 100:+.1f}%")
        print(f"vs Cezar target 1.2224: {(1.2224 - avg_ours) / 1.2224 * 100:+.1f}%")


if __name__ == "__main__":
    main()
