"""Soft-macro CD with UCB1-ranked probe direction selection (idea D12).

Diff from _softmacro.py: instead of probing all 8 directions in fixed order,
maintain per-macro (success, trials) counters per direction and order probes
by UCB1 score. Still probes all 8 eventually (we accept any improvement),
but orders them so recently-successful directions go first — if they win,
subsequent probes in the same (macro, delta) slot can be skipped via an
`early-accept-if-improved` gate.

Not a full multi-armed-bandit loop (we visit every arm each round) — the
only saving is early-accept on the first improving probe for a given
(macro, delta). This is a cheap test of whether probe-direction ordering
matters at all on ibm01 within 220 s.
"""
import time
import math
import numpy as np


def soft_macro_cd_ucb(pos_np, benchmark, incr_eval, max_time, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

    incr_eval.sync_positions(pos_np)

    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    sizes = np.zeros((n_total, 2), dtype=np.float64)
    if hasattr(benchmark.macro_sizes, 'numpy'):
        s = benchmark.macro_sizes.numpy()
        sizes[:s.shape[0]] = s

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = incr_eval.macro_pos.copy()

    dirs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]

    # Per-direction (success, trials) aggregated across all macros.
    # UCB1 score: success/trials + sqrt(2*ln(total_trials)/trials)
    n_dirs = len(dirs)
    succ = np.ones(n_dirs, dtype=np.float64)   # +1 smoothing
    tries = np.ones(n_dirs, dtype=np.float64)

    t0 = time.time()
    pass_num = 0
    n_moves = 0
    n_accepts_early = 0
    while time.time() - t0 < max_time:
        pass_num += 1
        pass_improved = False
        total = float(tries.sum())
        ucb = succ / tries + np.sqrt(2.0 * math.log(max(total, 2.0)) / tries)
        dir_order = list(np.argsort(-ucb))  # highest UCB first
        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])
                hw = sizes[i, 0] / 2
                hh = sizes[i, 1] / 2
                best_dir_cost = current_cost
                best_nx = None
                best_ny = None
                best_dir_idx = -1
                found_first_improving = False
                for d_idx in dir_order:
                    dx, dy = dirs[d_idx]
                    nx = float(np.clip(ox + delta * dx, hw, cw - hw))
                    ny = float(np.clip(oy + delta * dy, hh, ch - hh))
                    if abs(nx - ox) < 0.001 and abs(ny - oy) < 0.001:
                        tries[d_idx] += 1.0
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    tries[d_idx] += 1.0
                    improved = c < best_dir_cost
                    if improved:
                        succ[d_idx] += 1.0
                        if not found_first_improving:
                            # Early-accept: take this move, skip remaining dirs
                            best_dir_cost = c
                            best_nx = nx
                            best_ny = ny
                            best_dir_idx = d_idx
                            found_first_improving = True
                            incr_eval.undo_move()
                            n_accepts_early += 1
                            break
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_dir_cost
                    if best_dir_cost < best_cost:
                        best_cost = best_dir_cost
                        best_pos = incr_eval.macro_pos.copy()
                    pass_improved = True
                    n_moves += 1
        if not pass_improved:
            break

    if verbose:
        print(f"    soft_macro_cd_ucb: {pass_num} passes, {n_moves} moves "
              f"(early={n_accepts_early}), cost {current_cost:.6f}")
    return pos_np, best_cost
