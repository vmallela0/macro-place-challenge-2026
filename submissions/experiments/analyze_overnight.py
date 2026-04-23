"""Parse overnight/*.log, print comparison table vs baselines."""
import re
from pathlib import Path

OVERNIGHT_DIR = Path(__file__).parent / "results" / "overnight"

# Baselines from COMPETITION.md
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


def parse_log(f):
    text = f.read_text()
    m = re.search(r"^proxy=([0-9.]+)", text, re.M)
    t = re.search(r"VALID\s+\[([0-9.]+)s\]", text)
    invalid = "INVALID" in text
    return (float(m.group(1)) if m else None,
            float(t.group(1)) if t else None,
            invalid)


def main():
    rows = []
    for f in sorted(OVERNIGHT_DIR.glob("ibm*.log")):
        name = f.stem
        cost, runtime, invalid = parse_log(f)
        rows.append((name, cost, runtime, invalid))

    print(f"{'Bench':>8}  {'Ours':>8}  {'SA':>8}  {'RePlAce':>8}  "
          f"{'vs SA':>7}  {'vs RePlAce':>10}  {'Time':>7}")
    print("-" * 78)
    ours_vals = []
    sa_vals = []
    re_vals = []
    for name, cost, runtime, invalid in rows:
        if cost is None:
            print(f"{name:>8}  RUNNING / no result")
            continue
        flag = " INV" if invalid else ""
        sa = SA.get(name, 0)
        re_ = RE.get(name, 0)
        vs_sa = (sa - cost) / sa * 100 if sa else 0
        vs_re = (re_ - cost) / re_ * 100 if re_ else 0
        print(f"{name:>8}  {cost:>8.4f}  {sa:>8.4f}  {re_:>8.4f}  "
              f"{vs_sa:>+6.1f}%  {vs_re:>+9.1f}%  {runtime:>6.0f}s{flag}")
        if not invalid:
            ours_vals.append(cost)
            sa_vals.append(sa)
            re_vals.append(re_)

    if ours_vals:
        print("-" * 78)
        avg_ours = sum(ours_vals) / len(ours_vals)
        avg_sa = sum(sa_vals) / len(sa_vals)
        avg_re = sum(re_vals) / len(re_vals)
        print(f"{'AVG':>8}  {avg_ours:>8.4f}  {avg_sa:>8.4f}  {avg_re:>8.4f}  "
              f"{(avg_sa - avg_ours) / avg_sa * 100:>+6.1f}%  "
              f"{(avg_re - avg_ours) / avg_re * 100:>+9.1f}%")
        print(f"\nBenchmarks complete: {len(ours_vals)}/17")
        print(f"Prior submission (vmallela 1.4156). Improvement: "
              f"{(1.4156 - avg_ours) / 1.4156 * 100:+.1f}%")


if __name__ == "__main__":
    main()
