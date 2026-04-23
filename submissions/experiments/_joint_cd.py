"""Joint hard+soft CD: interleave one hard move with K soft moves per 'round'.

Hypothesis: when a hard macro shifts, nearby softs want to shift too.
Alternating hard/soft exploits this coupling faster than alternating
phases.
"""
import time
import numpy as np


def joint_hard_soft_cd(pos_np, benchmark, incr_eval, max_time,
                       softs_per_hard=4, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    gap = 0.05
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_hard_idx = np.where(movable)[0]
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    hard_sorted = sorted(movable_hard_idx.tolist(), key=lambda i: -net_count[i])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    hard_deltas = [2.0, 1.0, 0.5, 0.25, 0.12]
    soft_deltas = [1.0, 0.5, 0.25, 0.12]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_hard_moves = 0
    n_soft_moves = 0

    while time.time() - t0 < max_time:
        pass_improved = False
        for hi, h in enumerate(hard_sorted):
            if time.time() - t0 > max_time:
                break

            # Hard move probe
            ox = float(incr_eval.macro_pos[h, 0])
            oy = float(incr_eval.macro_pos[h, 1])
            for delta in hard_deltas:
                best_dir_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in dirs:
                    nx = float(np.clip(ox + delta * dx, half_w[h], cw - half_w[h]))
                    ny = float(np.clip(oy + delta * dy, half_h[h], ch - half_h[h]))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    # Check overlap
                    ddx = np.abs(nx - incr_eval.macro_pos[:n_hard, 0])
                    ddy = np.abs(ny - incr_eval.macro_pos[:n_hard, 1])
                    ov = (ddx < sep_x[h] + gap) & (ddy < sep_y[h] + gap)
                    ov[h] = False
                    if ov.any():
                        continue
                    c = incr_eval.move_macro(h, nx, ny)
                    if c < best_dir_c:
                        best_dir_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(h, best_nx, best_ny)
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c
                    n_hard_moves += 1
                    pass_improved = True
                    break

            # After hard move, try K random softs (focus on net-neighbors of h)
            nearby_softs = set()
            for nid in incr_eval.macro_nets[h]:
                for m in incr_eval.net_macros[nid]:
                    if m in soft_movable:
                        nearby_softs.add(m)
            nearby_softs = list(nearby_softs)[:softs_per_hard]

            for m in nearby_softs:
                if time.time() - t0 > max_time:
                    break
                mx = float(incr_eval.macro_pos[m, 0])
                my = float(incr_eval.macro_pos[m, 1])
                for delta in soft_deltas[:3]:  # 3 deltas only — cheap
                    best_c = current_cost
                    best_nmx, best_nmy = None, None
                    for dx, dy in dirs:
                        nx = float(max(0, min(cw, mx + delta * dx)))
                        ny = float(max(0, min(ch, my + delta * dy)))
                        if abs(nx - mx) < 1e-4 and abs(ny - my) < 1e-4:
                            continue
                        c = incr_eval.move_macro(m, nx, ny)
                        if c < best_c:
                            best_c = c
                            best_nmx, best_nmy = nx, ny
                        incr_eval.undo_move()
                    if best_nmx is not None:
                        incr_eval.move_macro(m, best_nmx, best_nmy)
                        current_cost = best_c
                        if best_c < best_cost:
                            best_cost = best_c
                        n_soft_moves += 1
                        break
        if not pass_improved:
            break

    if verbose:
        print(f"    joint: {n_hard_moves} hard, {n_soft_moves} soft, cost {best_cost:.6f}")
    return pos_np, best_cost
