"""grav_polish.py — A/B test gravity_drop vs default init under coordinate descent.

Pipeline (per bench):
  1. Load benchmark + plc.
  2. Run v1 _coord_descent from gravity-drop hard positions → cost_GRAV.
  3. Run v1 _coord_descent from default hard positions → cost_DEF.
  4. Report cost_GRAV vs cost_DEF and the delta.

This is the lightweight Mac-friendly stand-in for "v7 polish on top of
gravity init" — _coord_descent is the polish kernel inside v7's pipeline,
and is what determines whether the basin reached from the init is better.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_v1():
    v1_path = REPO / "submissions" / "vmallela" / "placer.py"
    spec = importlib.util.spec_from_file_location("_v1_polish", str(v1_path))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    return v1


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def polish(v1, plc, benchmark, init_hard_pos, cd_time, seed,
           legalize_iters=0, legalize_damping=0.6,
           init_soft_pos=None):
    """Run v1 _coord_descent from the given init.
    init_soft_pos: optional (n_soft, 2) ndarray. If provided, sets soft macro
    positions BEFORE building the IncrementalEvaluator (mutates a fresh copy
    of benchmark.macro_positions on a clone so we don't corrupt caller state).
    If legalize_iters > 0, first apply _push_apart to bridge the legalization gap.
    Returns (final_pos, final_cost, init_cost, post_legalize_cost).
    """
    import random as _rand
    _rand.seed(seed)
    np.random.seed(seed)

    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.macro_positions.shape[0])

    # If soft positions are provided, patch the benchmark in-place. Reverted at end.
    saved_default = None
    if init_soft_pos is not None:
        soft_np = np.asarray(init_soft_pos, dtype=np.float32).reshape(n_total - n_hard, 2)
        saved_default = benchmark.macro_positions.clone()
        new_full = benchmark.macro_positions.clone()
        new_full[n_hard:] = torch.from_numpy(soft_np)
        benchmark.macro_positions = new_full

    try:
        incr = v1.IncrementalEvaluator(plc, benchmark)
        init = np.asarray(init_hard_pos, dtype=np.float64).reshape(n_hard, 2).copy()
        incr.sync_positions(init.astype(np.float32))
        init_cost = float(incr.get_proxy_cost())

        post_legal = init_cost
        if legalize_iters > 0:
            legalized = v1._push_apart(init, benchmark,
                                       max_iters=legalize_iters, damping=legalize_damping)
            incr.sync_positions(legalized.astype(np.float32))
            post_legal = float(incr.get_proxy_cost())
            init = legalized

        pos, cost = v1._coord_descent(init, benchmark, plc, max_time=cd_time, incr_eval=incr)
    finally:
        if saved_default is not None:
            benchmark.macro_positions = saved_default

    return pos, float(cost), init_cost, post_legal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--grav-init", required=True,
                   help="gravity_drop JSON output (used for init hard positions)")
    p.add_argument("--output", required=True)
    p.add_argument("--cd-time", type=float, default=120.0,
                   help="seconds for coordinate descent polish per arm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", default="grav,default",
                   help="comma-separated arms to run (subset of {grav,default})")
    p.add_argument("--legalize-iters", type=int, default=0,
                   help="apply _push_apart with this many iters before CD (0 disables)")
    p.add_argument("--legalize-damping", type=float, default=0.6)
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    v1 = load_v1()
    from macro_place.benchmark import Benchmark

    bench_path = f"benchmarks/processed/public/{args.benchmark}.pt"
    benchmark = Benchmark.load(bench_path)
    plc = v1._load_plc(args.benchmark)
    if plc is None:
        print(f"FATAL: plc load failed for {args.benchmark}", file=sys.stderr)
        sys.exit(2)
    n_hard = int(benchmark.num_hard_macros)

    results = {"benchmark": args.benchmark, "n_hard": n_hard, "arms": {}}
    t0 = time.time()

    # Arm A: gravity init
    if "grav" in arms:
        with open(args.grav_init) as f:
            g = json.load(f)
        grav_hard = np.asarray(g["positions"], dtype=np.float64)
        if grav_hard.shape != (n_hard, 2):
            print(f"FATAL: grav init shape {grav_hard.shape} != ({n_hard}, 2)", file=sys.stderr)
            sys.exit(2)
        # Optional: soft macro positions from the init (lets diffusion_init pass
        # its spectral soft coords through so the polish sees the same eval).
        grav_soft = None
        if "soft_positions" in g:
            soft_arr = np.asarray(g["soft_positions"], dtype=np.float32)
            n_soft = int(benchmark.macro_positions.shape[0]) - n_hard
            if soft_arr.shape == (n_soft, 2):
                grav_soft = soft_arr
        print(f"[polish/grav] benchmark={args.benchmark} CD={args.cd_time}s "
              f"legalize_iters={args.legalize_iters} "
              f"soft={'spec' if grav_soft is not None else 'default'} ...", flush=True)
        ta = time.time()
        gpos, gcost, ginit, gpostlegal = polish(
            v1, plc, benchmark, grav_hard, args.cd_time, args.seed,
            legalize_iters=args.legalize_iters, legalize_damping=args.legalize_damping,
            init_soft_pos=grav_soft)
        wa = time.time() - ta
        print(f"[polish/grav] init={ginit:.4f}  post_legal={gpostlegal:.4f}  "
              f"final={gcost:.4f}  delta_total={gcost-ginit:+.4f}  wall={wa:.1f}s",
              flush=True)
        results["arms"]["grav"] = {
            "init_cost": ginit, "post_legalize_cost": gpostlegal,
            "final_cost": gcost, "wall_s": wa,
            "positions": gpos.tolist(),
        }

    # Arm B: default benchmark init
    if "default" in arms:
        default_hard = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
        print(f"[polish/default] benchmark={args.benchmark} CD={args.cd_time}s ...", flush=True)
        tb = time.time()
        # Default arm: skip legalize (default init is already legal)
        dpos, dcost, dinit, _ = polish(
            v1, plc, benchmark, default_hard, args.cd_time, args.seed,
            legalize_iters=0)
        wb = time.time() - tb
        print(f"[polish/default] init={dinit:.4f}  final={dcost:.4f}  delta={dcost-dinit:+.4f}  wall={wb:.1f}s",
              flush=True)
        results["arms"]["default"] = {
            "init_cost": dinit, "final_cost": dcost, "wall_s": wb,
            "positions": dpos.tolist(),
        }

    results["wall_s"] = time.time() - t0
    if "grav" in results["arms"] and "default" in results["arms"]:
        g = results["arms"]["grav"]["final_cost"]
        d = results["arms"]["default"]["final_cost"]
        results["delta_grav_minus_default"] = g - d
        print(f"\n[A/B] {args.benchmark}: grav={g:.4f}  default={d:.4f}  Δ={g-d:+.4f}  "
              f"{'GRAV WINS' if g < d else 'DEFAULT WINS'}", flush=True)

    atomic_write_json(args.output, results)
    print(f"[polish] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
