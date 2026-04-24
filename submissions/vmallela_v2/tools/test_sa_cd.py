"""Quick sanity check for SA-enabled _coord_descent on ibm01.

Runs 60s greedy vs 60s SA (sa_T0=0.001, cooling=0.9995) from the same
legalized starting point and prints both best_costs.

Invoke:
    uv run python submissions/vmallela_v2/tools/test_sa_cd.py
Optional args:
    --time N   (per-run budget, default 60s)
    --T0 F     (SA initial temperature, default 0.001)
    --cool F   (SA cooling, default 0.9995)
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from macro_place.benchmark import Benchmark

_placer_path = ROOT / "submissions" / "vmallela" / "placer.py"
_spec = importlib.util.spec_from_file_location("_vmallela_placer", str(_placer_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator
_coord_descent = _mod._coord_descent
_push_apart = _mod._push_apart
_legalize = _mod._legalize
_refine_toward_initial = _mod._refine_toward_initial


def _pick_legal_start(benchmark, plc_eval, budget_s=90.0):
    """Reproduce the v2 legalize tournament briefly to get a feasible start."""
    from macro_place.objective import compute_proxy_cost

    n_hard = benchmark.num_hard_macros
    init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
    sizes_np = benchmark.macro_sizes[:n_hard].numpy()

    def _has_overlap(pos_np):
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes_np[i, 0] + sizes_np[j, 0]) / 2 and
                    abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes_np[i, 1] + sizes_np[j, 1]) / 2):
                    return True
            if time.time() - _pick_legal_start._deadline > 0:  # time bail
                return False
        return False

    best_pos = None
    best_cost = float("inf")
    t0 = time.time()
    _pick_legal_start._deadline = t0 + budget_s * 2

    pushed = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d)
              for mi, d in [(300, 0.4), (500, 0.6), (800, 0.8)]]
    starts = [(p, f"push_{k}") for k, p in enumerate(pushed)] + [(init_pos, "raw")]

    seen = set()
    n_tried = 0
    n_accepted = 0
    for ot in range(30):
        if time.time() - t0 > budget_s: break
        for sm in [0.05, 0.08, 0.12, 0.18]:
            if time.time() - t0 > budget_s: break
            for sp, _ in starts:
                if time.time() - t0 > budget_s: break
                legal = _legalize(sp, benchmark, order_type=ot, step_mult=sm)
                refined = _refine_toward_initial(legal, init_pos, benchmark)
                h = hash(np.round(refined * 10).astype(np.int32).tobytes())
                if h in seen: continue
                seen.add(h)
                n_tried += 1
                full = benchmark.macro_positions.clone()
                full[:n_hard] = torch.tensor(refined, dtype=torch.float32)
                r = compute_proxy_cost(full, benchmark, plc_eval)
                if r["overlap_count"] == 0 and r["proxy_cost"] < best_cost:
                    n_accepted += 1
                    best_cost = r["proxy_cost"]
                    best_pos = refined.copy()

    print(f"[legalize] tried={n_tried} accepted={n_accepted}", flush=True)
    return best_pos, best_cost


def run_one(label, start_pos, benchmark, plc_cd, max_time, sa_T0, sa_cooling, sa_seed):
    incr_eval = IncrementalEvaluator(plc_cd, benchmark)
    t0 = time.time()
    best_pos, best_cost = _coord_descent(
        start_pos.copy(), benchmark, plc_cd, max_time=max_time, incr_eval=incr_eval,
        sa_T0=sa_T0, sa_cooling=sa_cooling, sa_rng_seed=sa_seed)
    elapsed = time.time() - t0
    print(f"[{label}] elapsed={elapsed:.1f}s  best_cost={best_cost:.6f}  "
          f"sa_T0={sa_T0}  sa_cool={sa_cooling}", flush=True)
    return best_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ibm01")
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--T0", type=float, default=0.001)
    ap.add_argument("--cool", type=float, default=0.9995)
    ap.add_argument("--legalize_budget", type=float, default=90.0)
    args = ap.parse_args()

    pt_path = ROOT / "benchmarks" / "processed" / "public" / f"{args.bench}.pt"
    print(f"[load] {pt_path}", flush=True)
    benchmark = Benchmark.load(str(pt_path))

    plc_eval = _load_plc(benchmark.name)
    if plc_eval is None:
        print(f"[err] _load_plc returned None for {benchmark.name}", file=sys.stderr)
        sys.exit(1)

    print(f"[legalize] tournament budget {args.legalize_budget:.0f}s", flush=True)
    start_pos, start_cost = _pick_legal_start(
        benchmark, plc_eval, budget_s=args.legalize_budget)
    if start_pos is None:
        print("[err] failed to find legal start", file=sys.stderr)
        sys.exit(1)
    print(f"[legalize] start_cost={start_cost:.6f}", flush=True)

    plc_cd = _load_plc(benchmark.name)

    # Run 1: greedy (spec says sa_T0=0 ≡ greedy — code treats <=0 as greedy).
    greedy_cost = run_one(
        "greedy", start_pos, benchmark, plc_cd,
        max_time=args.time, sa_T0=None, sa_cooling=args.cool, sa_seed=None)

    # Run 2: SA
    sa_cost = run_one(
        "sa", start_pos, benchmark, plc_cd,
        max_time=args.time, sa_T0=args.T0, sa_cooling=args.cool, sa_seed=0)

    # Summary
    delta = greedy_cost - sa_cost
    rel = delta / greedy_cost * 100.0 if greedy_cost > 0 else 0.0
    verdict = ("SA_BETTER" if delta > 1e-8
               else ("TIE" if abs(delta) <= 1e-8 else "SA_WORSE"))
    print(f"[result] greedy={greedy_cost:.6f}  sa={sa_cost:.6f}  "
          f"delta={delta:+.6f} ({rel:+.3f}%)  verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
