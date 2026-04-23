"""Targeted soft-macro LNS: destroy softs in the worst congestion cell.

Hypothesis: soft macros in/near a congestion hotspot are the ones we most
want to relocate. Random soft-LNS wastes attempts on softs that are
already near-optimal.
"""
import time
import numpy as np


def targeted_soft_lns_phase(pos_np, benchmark, incr_eval, max_time,
                            n_destroy=10, n_candidates=30, verbose=False):
    """Soft LNS targeted at congestion hotspots."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable_set = {n_hard + i for i in range(n_total - n_hard)
                        if not macro_fixed[n_hard + i]}
    if len(soft_movable_set) < 2:
        return pos_np, incr_eval.get_proxy_cost()

    grid_w = incr_eval.grid_width
    grid_h = incr_eval.grid_height
    grid_col = incr_eval.grid_col

    incr_eval.sync_positions(pos_np)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost

    rng = np.random.RandomState(54321)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        n_iter += 1

        # Pick a hotspot cell
        combined = incr_eval.V_routing_smooth + incr_eval.H_routing_smooth
        topk = np.argsort(combined)[-20:]
        cell_idx = int(rng.choice(topk))
        row = cell_idx // grid_col
        col = cell_idx % grid_col
        cell_cx = (col + 0.5) * grid_w
        cell_cy = (row + 0.5) * grid_h

        # Find closest n_destroy soft macros
        dists = []
        for m in soft_movable_set:
            dx = incr_eval.macro_pos[m, 0] - cell_cx
            dy = incr_eval.macro_pos[m, 1] - cell_cy
            d = dx * dx + dy * dy
            dists.append((d, m))
        dists.sort(key=lambda x: x[0])
        subset = [m for _, m in dists[:n_destroy]]
        if len(subset) < 2:
            continue

        rng.shuffle(subset)
        move_trail = []
        moved_any = False

        for m in subset:
            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])
            cands = []
            # Net-centroid biased
            cx_s, cy_s, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if m2 != m:
                        cx_s += float(incr_eval.macro_pos[m2, 0])
                        cy_s += float(incr_eval.macro_pos[m2, 1])
                        cnt += 1
            if cnt > 0:
                cx_c = cx_s / cnt
                cy_c = cy_s / cnt
                for _ in range(10):
                    cands.append((cx_c + rng.uniform(-0.8, 0.8),
                                  cy_c + rng.uniform(-0.8, 0.8)))
            # Anti-hotspot: positions AWAY from cell_cx/cy
            for _ in range(n_candidates - 10):
                cands.append((rng.uniform(0, cw), rng.uniform(0, ch)))

            best_cx, best_cy, best_c = ox, oy, current_cost
            kept_old = True
            for cx, cy in cands:
                cx = float(max(0.0, min(cw, cx)))
                cy = float(max(0.0, min(ch, cy)))
                c = incr_eval.move_macro(m, cx, cy)
                if c < best_c:
                    best_c = c
                    best_cx = cx
                    best_cy = cy
                    kept_old = False
                incr_eval.undo_move()
            if not kept_old:
                incr_eval.move_macro(m, best_cx, best_cy)
                move_trail.append((m, ox, oy))
                current_cost = best_c
                moved_any = True

        if not moved_any:
            continue

        if current_cost < best_cost - 1e-7:
            best_cost = current_cost
            n_accepted += 1
        else:
            for m, ox, oy in reversed(move_trail):
                incr_eval.move_macro(m, ox, oy)
            current_cost = best_cost

    if verbose:
        print(f"    tgt_soft_lns: {n_iter} iters, {n_accepted} accepted, "
              f"cost {best_cost:.6f}")
    return pos_np, best_cost
