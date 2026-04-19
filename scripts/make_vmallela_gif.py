"""Generate an animated GIF of the vmallela macro placer on ibm01.

Two-panel layout per frame:
  - Left: current placement (macros as boxes)
  - Right: cost curve (proxy cost vs step, with current step highlighted)

Annotations: step, delta (CD step-size schedule), current cost, best cost,
overlaps, phase label.

Captures snapshots during:
  1. Initial placement
  2. Push-apart iterations (overlap resolution)
  3. Legalization + refinement
  4. Coordinate descent (custom instrumented loop, snapshots every N improvements)
"""

import importlib.util
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "vmallela_placer", ROOT / "submissions" / "vmallela" / "placer.py"
)
placer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placer_mod)

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

BENCH = "ibm01"
CD_BUDGET = 40.0           # seconds of CD to run
OUT_PATH = ROOT / "assets" / "vmallela_ibm01.gif"
DPI = 100
FIG_W, FIG_H = 14, 7       # inches (bigger for two panels)
SNAPSHOT_EVERY = 8         # capture a frame every N accepted CD moves


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(
    pos_np,
    benchmark,
    phase,
    step,
    delta,
    cost,
    best,
    overlaps,
    history_steps,
    history_costs,
    history_best,
):
    """Render a two-panel frame: placement (left) + cost curve (right)."""
    fig, (ax_p, ax_c) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), dpi=DPI,
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.15},
    )

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy()
    movable = benchmark.get_movable_mask()[:n_hard].numpy()

    # ── Placement panel ────────────────────────────────────────────────────
    ax_p.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="#222", linewidth=1.5))
    for i in range(n_hard):
        x, y = pos_np[i]
        w, h = sizes[i]
        color = "#c46a6a" if not movable[i] else "#4a7ab0"
        edge = "#222"
        ax_p.add_patch(Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=color, alpha=0.6, edgecolor=edge, linewidth=0.4,
        ))
    ax_p.set_xlim(-0.5, cw + 0.5)
    ax_p.set_ylim(-0.5, ch + 0.5)
    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])

    # Title: phase name; subtitle: cost / best / overlaps
    bits = []
    if cost is not None:
        bits.append(f"cost {cost:.4f}")
    if best is not None:
        bits.append(f"best {best:.4f}")
    if overlaps is not None:
        bits.append(f"overlaps {overlaps}")
    subtitle = "   ".join(bits)
    ax_p.set_title(
        f"ibm01 — {phase}\n{subtitle}",
        fontsize=13, family="monospace", pad=10,
    )

    # ── Cost curve panel ───────────────────────────────────────────────────
    ax_c.set_title("proxy cost over optimization", fontsize=14, family="monospace", pad=10)
    if history_steps:
        xs = np.array(history_steps)
        ys = np.array(history_costs)
        bs = np.array(history_best)
        ax_c.plot(xs, ys, color="#4a7ab0", linewidth=1.5, label="current", alpha=0.75)
        ax_c.plot(xs, bs, color="#2e7d32", linewidth=2.0, label="best", alpha=0.9)
        ax_c.scatter([xs[-1]], [ys[-1]], color="#d94a4a", s=60, zorder=5,
                     edgecolor="white", linewidth=1.5, label="now")
        ax_c.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax_c.set_xlabel("step", family="monospace", fontsize=11)
    ax_c.set_ylabel("proxy cost", family="monospace", fontsize=11)
    ax_c.grid(True, linestyle=":", alpha=0.4)

    # Y axis: fixed range if we have history (ignore NaN from pre-valid frames)
    if history_costs:
        all_vals = [v for v in (history_costs + history_best) if not (isinstance(v, float) and np.isnan(v))]
        if all_vals:
            ymin = min(all_vals) - 0.005
            ymax = max(all_vals) + 0.005
            ax_c.set_ylim(ymin, ymax)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_overlaps(pos_np, benchmark):
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy()
    count = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes[i, 0] + sizes[j, 0]) / 2 - 1e-3
                and abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes[i, 1] + sizes[j, 1]) / 2 - 1e-3):
                count += 1
    return count


def full_cost(pos_np, benchmark, plc):
    n_hard = benchmark.num_hard_macros
    full = benchmark.macro_positions.clone()
    full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
    r = compute_proxy_cost(full, benchmark, plc)
    return r["proxy_cost"], r["overlap_count"]


# ---------------------------------------------------------------------------
# Instrumented coordinate descent (mirrors _coord_descent, emits snapshots)
# ---------------------------------------------------------------------------

def instrumented_cd(pos_np, benchmark, incr_eval, max_time, on_snapshot):
    """Custom CD loop that matches placer.py's _coord_descent but calls
    on_snapshot(pos, step, delta, cost, best) periodically."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_hard)])
    movable_sorted = sorted(movable_idx.tolist(), key=lambda i: -net_count[i])

    macro_max_dim = np.maximum(sizes[:, 0], sizes[:, 1])
    avg_macro_size = macro_max_dim[movable_idx].mean() if len(movable_idx) > 0 else 1.0
    size_scale = macro_max_dim / max(avg_macro_size, 1e-6)

    pos = pos_np.copy()
    gap = 0.05
    incr_eval.sync_positions(pos)

    def check_overlap(idx):
        ddx = np.abs(pos[idx, 0] - pos[:, 0])
        ddy = np.abs(pos[idx, 1] - pos[:, 1])
        o = (ddx < sep_x[idx] + gap) & (ddy < sep_y[idx] + gap)
        o[idx] = False
        return o.any()

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    delta_schedule = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35,
                      0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]

    current_cost = incr_eval.get_proxy_cost()
    best_pos = pos.copy()
    best_cost = current_cost
    step = 0
    accepted_since_snap = 0

    t0 = time.time()

    # Initial snapshot
    on_snapshot(pos.copy(), step, None, current_cost, best_cost)

    for _pass in range(100):
        if time.time() - t0 >= max_time:
            break
        pass_improved = False

        for delta in delta_schedule:
            if time.time() - t0 >= max_time:
                break

            for i in movable_sorted:
                if time.time() - t0 >= max_time:
                    break

                scaled_delta = delta * size_scale[i]
                old_x, old_y = pos[i, 0], pos[i, 1]

                best_dir_cost = current_cost
                best_dir_pos = None

                for ddx, ddy in dirs:
                    nx = np.clip(old_x + scaled_delta * ddx, half_w[i], cw - half_w[i])
                    ny = np.clip(old_y + scaled_delta * ddy, half_h[i], ch - half_h[i])
                    if abs(nx - old_x) < 0.001 and abs(ny - old_y) < 0.001:
                        continue

                    pos[i, 0] = nx
                    pos[i, 1] = ny
                    if check_overlap(i):
                        pos[i, 0] = old_x
                        pos[i, 1] = old_y
                        continue

                    cost = incr_eval.move_macro(i, nx, ny)
                    if cost < best_dir_cost:
                        best_dir_cost = cost
                        best_dir_pos = (nx, ny)
                    incr_eval.undo_move()

                if best_dir_pos is not None:
                    incr_eval.move_macro(i, best_dir_pos[0], best_dir_pos[1])
                    pos[i, 0] = best_dir_pos[0]
                    pos[i, 1] = best_dir_pos[1]
                    current_cost = best_dir_cost
                    step += 1
                    accepted_since_snap += 1
                    pass_improved = True
                    if best_dir_cost < best_cost:
                        best_cost = best_dir_cost
                        best_pos = pos.copy()

                    if accepted_since_snap >= SNAPSHOT_EVERY:
                        on_snapshot(pos.copy(), step, delta, current_cost, best_cost)
                        accepted_since_snap = 0

        if not pass_improved:
            break

    # Final snapshot
    on_snapshot(pos.copy(), step, delta_schedule[-1], current_cost, best_cost)
    return best_pos, best_cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {BENCH}...")
    bench_dir = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / BENCH
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n_hard = benchmark.num_hard_macros
    init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)

    frames = []

    # Shared state for history plot (built across phases)
    hist_steps, hist_costs, hist_best = [], [], []
    step_counter = [0]
    best_valid = [None]  # best cost among zero-overlap placements only

    def push_frame(pos, phase, delta, cost, overlaps):
        step_counter[0] += 1
        # Only update "best" on valid (overlap-free) placements.
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
            step=step_counter[0],
            delta=delta,
            cost=cost,
            best=display_best,
            overlaps=overlaps,
            history_steps=hist_steps,
            history_costs=hist_costs,
            history_best=hist_best,
        ))

    # ── Phase 0: initial ───────────────────────────────────────────────────
    cost, ov = full_cost(init_pos, benchmark, plc)
    push_frame(init_pos, "initial placement", None, cost, ov)
    print(f"initial: cost={cost:.4f} overlaps={ov}")

    # ── Phase 1: push-apart (several chunks) ───────────────────────────────
    pos = init_pos.copy()
    for k, iters in enumerate([40, 80, 160, 320]):
        pos = placer_mod._push_apart(pos, benchmark, max_iters=iters, damping=0.5)
        ov = count_overlaps(pos, benchmark)
        cost_pa, _ = full_cost(pos, benchmark, plc)
        push_frame(pos, "phase 1 · push-apart", None, cost_pa, ov)
        print(f"push_apart {k}: cost={cost_pa:.4f} overlaps={ov}")

    # ── Phase 2: legalize + refine ─────────────────────────────────────────
    legal = placer_mod._legalize(pos, benchmark, order_type=0, step_mult=0.08)
    ov = count_overlaps(legal, benchmark)
    cost_l, _ = full_cost(legal, benchmark, plc)
    push_frame(legal, "phase 2 · legalization", None, cost_l, ov)
    print(f"legalized: cost={cost_l:.4f} overlaps={ov}")

    refined = placer_mod._refine_toward_initial(legal, init_pos, benchmark, n_passes=20)
    ov = count_overlaps(refined, benchmark)
    cost_r, _ = full_cost(refined, benchmark, plc)
    push_frame(refined, "phase 2 · refinement", None, cost_r, ov)
    print(f"refined: cost={cost_r:.4f} overlaps={ov}")

    # ── Phase 3: coordinate descent (instrumented) ─────────────────────────
    print(f"Phase 3: coordinate descent (budget {CD_BUDGET}s)...")
    incr = placer_mod.IncrementalEvaluator(plc, benchmark)

    def cd_snapshot(pos_now, cd_step, delta, cost_now, best_now):
        # overlaps should always be 0 from here on; skip expensive recount
        push_frame(pos_now, "phase 3 · coordinate descent",
                   f"{delta:.2f}" if delta is not None else None,
                   cost_now, 0)

    instrumented_cd(refined, benchmark, incr, CD_BUDGET, cd_snapshot)
    print(f"CD frames captured. Total: {len(frames)}")

    # Hold final frame
    for _ in range(6):
        frames.append(frames[-1])

    # ── Write GIF ──────────────────────────────────────────────────────────
    print(f"Writing {OUT_PATH}...")
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        optimize=True,
    )
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Done. {OUT_PATH.name}: {len(frames)} frames, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
