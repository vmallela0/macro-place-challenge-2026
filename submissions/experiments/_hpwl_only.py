"""HPWL-only fast probe for soft macros.

For soft macros (zero size, no blockage), each move's impact is:
  - HPWL changes (per-net bbox recompute)
  - Density UNCHANGED (no mass)
  - Congestion changes (net re-routing)

Full incr_eval.move_macro does all three. For cheap probes, we can
compute HPWL delta analytically without touching density/congestion:

  dHPWL_net = new_bbox_width + new_bbox_height - old_bbox_width - old_bbox_height

Scaled by net_weight, summed over affected nets.

This is NOT the true proxy delta — it ignores congestion. But it's a
good FIRST-STAGE filter: if dHPWL is big-positive, move is likely bad.
Only run full probe for dHPWL-improving candidates.

Cost: per probe ~5-10 us (vs 50 ms for full incr). 5000x speedup on
filter stage.
"""
import time
import numpy as np


def hpwl_delta_for_soft_move(incr_eval, macro_idx, new_x, new_y):
    """Return HPWL delta (new - old) for moving macro_idx to (new_x, new_y)."""
    old_x = float(incr_eval.macro_pos[macro_idx, 0])
    old_y = float(incr_eval.macro_pos[macro_idx, 1])

    delta = 0.0
    for nid in incr_eval.macro_nets[macro_idx]:
        start = int(incr_eval.net_starts[nid])
        end = int(incr_eval.net_starts[nid + 1])
        w = float(incr_eval.net_weight[nid])

        # Current HPWL of this net (cached)
        old_hpwl = float(incr_eval.net_hpwl[nid])

        # Compute new HPWL: iterate pins, replace this macro's pin positions
        xs = []
        ys = []
        for pidx in range(start, end):
            pm = int(incr_eval.pin_macro[pidx])
            xo = float(incr_eval.pin_xoff[pidx])
            yo = float(incr_eval.pin_yoff[pidx])
            if pm < 0:
                xs.append(xo)
                ys.append(yo)
            elif pm == macro_idx:
                xs.append(new_x + xo)
                ys.append(new_y + yo)
            else:
                xs.append(float(incr_eval.macro_pos[pm, 0]) + xo)
                ys.append(float(incr_eval.macro_pos[pm, 1]) + yo)
        new_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        delta += w * (new_hpwl - old_hpwl)

    return delta


def soft_cd_hpwl_filter(pos_np, benchmark, incr_eval, max_time,
                       filter_keep_frac=0.3, verbose=False):
    """Soft CD with HPWL-filter first stage.

    For each soft macro, probe 8 directions with HPWL-only (fast).
    Keep top filter_keep_frac (e.g. 30%) by HPWL-delta.
    Verify those with full incr eval; accept if proxy cost improves.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    n_moves = 0
    n_filtered = 0
    n_verified = 0
    t0 = time.time()
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        improved = False
        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])

                # Stage 1: HPWL-only ranking of 8 directions
                candidates = []
                for dx, dy in dirs:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    dhpwl = hpwl_delta_for_soft_move(incr_eval, i, nx, ny)
                    candidates.append((dhpwl, nx, ny))
                if not candidates:
                    continue
                candidates.sort()  # ascending: most HPWL-reducing first
                # Keep top-K
                keep_k = max(2, int(len(candidates) * filter_keep_frac))
                top = candidates[:keep_k]
                n_filtered += len(candidates) - keep_k

                # Stage 2: verify with full proxy eval
                best_c = current_cost
                best_nx, best_ny = None, None
                for dhpwl, nx, ny in top:
                    n_verified += 1
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_c:
                        best_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()

                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_c
                    if best_c < best_cost:
                        best_cost = best_c
                    n_moves += 1
                    improved = True
        if not improved:
            break

    if verbose:
        print(f"    hpwl_filter: {pass_num} passes, {n_moves} moves, "
              f"{n_verified} verified, {n_filtered} filtered, cost {best_cost:.6f}")
    return pos_np, best_cost
