"""Generate a v6 convergence GIF for one benchmark.

Usage:
    scripts/make_v6_gif.py <bench_name> [budget_s] [out_path]

Builds an animated GIF showing the placement evolving through:
  Phase 0: initial benchmark placement
  Phase 1: push-apart (overlap resolution, 4 chunks)
  Phase 2: legalize + refine_toward_initial
  Phase 3: instrumented coordinate descent (snapshot every 8 accepted moves)

Used by the v6 sweep to visualize hard benchmarks (proxy > 1.0) so we can
see WHERE in the optimization the placer is getting stuck.

Lighter than the full v6 portfolio: single CPU worker, no consensus,
short CD budget (default 60 s). The GIF shows the trajectory, not the
final optimal cost — diagnostic only.

Style mirrors `scripts/make_vmallela_gif.py` (which it heavily reuses).
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the existing GIF infrastructure verbatim — render_frame,
# instrumented_cd, full_cost, count_overlaps are reused unchanged.
_gif_spec = importlib.util.spec_from_file_location(
    "_v_gif", str(ROOT / "scripts" / "make_vmallela_gif.py"))
_gif_mod = importlib.util.module_from_spec(_gif_spec)
_gif_spec.loader.exec_module(_gif_mod)

render_frame = _gif_mod.render_frame
instrumented_cd = _gif_mod.instrumented_cd
full_cost = _gif_mod.full_cost
count_overlaps = _gif_mod.count_overlaps
placer_mod = _gif_mod.placer_mod
load_benchmark_from_dir = _gif_mod.load_benchmark_from_dir


def make_gif(bench_name: str, cd_budget: float = 60.0,
             out_path: str | None = None,
             snapshot_every: int = 8):
    if out_path is None:
        out_path = str(ROOT / "assets" / f"v6_{bench_name}.gif")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bench_dir = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name
    print(f"[v6_gif] loading {bench_name} from {bench_dir}...")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n_hard = benchmark.num_hard_macros
    init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

    frames = []
    hist_steps, hist_costs, hist_best = [], [], []
    step_counter = [0]
    best_valid = [None]

    def push_frame(pos, phase, delta, cost, overlaps):
        step_counter[0] += 1
        if cost is not None and overlaps == 0:
            if best_valid[0] is None or cost < best_valid[0]:
                best_valid[0] = cost
        display_best = best_valid[0]
        if cost is not None:
            hist_steps.append(step_counter[0])
            hist_costs.append(cost)
            hist_best.append(display_best if display_best is not None else np.nan)
        frames.append(render_frame(
            pos, benchmark, phase,
            step=step_counter[0], delta=delta,
            cost=cost, best=display_best, overlaps=overlaps,
            history_steps=hist_steps,
            history_costs=hist_costs,
            history_best=hist_best,
        ))

    # Phase 0: initial
    cost, ov = full_cost(init_pos, benchmark, plc)
    push_frame(init_pos, f"{bench_name} · initial placement", None, cost, ov)
    print(f"  initial: cost={cost:.4f} overlaps={ov}")

    # Phase 1: push-apart (4 chunks)
    pos = init_pos.copy()
    for k, iters in enumerate([40, 80, 160, 320]):
        pos = placer_mod._push_apart(pos, benchmark, max_iters=iters, damping=0.5)
        ov = count_overlaps(pos, benchmark)
        cost_pa, _ = full_cost(pos, benchmark, plc)
        push_frame(pos, f"{bench_name} · phase 1 · push-apart",
                   None, cost_pa, ov)
        print(f"  push_apart {k}: cost={cost_pa:.4f} overlaps={ov}")

    # Phase 2: legalize + refine
    legal = placer_mod._legalize(pos, benchmark, order_type=0, step_mult=0.08)
    ov = count_overlaps(legal, benchmark)
    cost_l, _ = full_cost(legal, benchmark, plc)
    push_frame(legal, f"{bench_name} · phase 2 · legalization", None, cost_l, ov)
    print(f"  legalized: cost={cost_l:.4f} overlaps={ov}")

    refined = placer_mod._refine_toward_initial(legal, init_pos, benchmark,
                                                 n_passes=20)
    ov = count_overlaps(refined, benchmark)
    cost_r, _ = full_cost(refined, benchmark, plc)
    push_frame(refined, f"{bench_name} · phase 2 · refinement",
               None, cost_r, ov)
    print(f"  refined: cost={cost_r:.4f} overlaps={ov}")

    # Phase 3: instrumented CD
    print(f"  phase 3 CD ({cd_budget}s)...")
    incr = placer_mod.IncrementalEvaluator(plc, benchmark)

    def cd_snapshot(pos_now, cd_step, delta, cost_now, best_now):
        push_frame(pos_now, f"{bench_name} · phase 3 · coordinate descent",
                   f"{delta:.2f}" if delta is not None else None,
                   cost_now, 0)

    # Use the existing instrumented_cd helper from make_vmallela_gif.
    # It already honors SNAPSHOT_EVERY internally.
    _gif_mod.SNAPSHOT_EVERY = snapshot_every
    instrumented_cd(refined, benchmark, incr, cd_budget, cd_snapshot)
    print(f"  CD frames captured. Total frames: {len(frames)}")

    # Hold final frame for ~1.5s
    for _ in range(6):
        frames.append(frames[-1])

    print(f"[v6_gif] writing {out_path}...")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[v6_gif] done. {out_path.name}: {len(frames)} frames, "
          f"{size_mb:.2f} MB")
    return out_path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <bench_name> [budget_s] [out_path]",
              file=sys.stderr)
        sys.exit(2)
    bench = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    out = sys.argv[3] if len(sys.argv) > 3 else None
    make_gif(bench, cd_budget=budget, out_path=out)


if __name__ == "__main__":
    main()
