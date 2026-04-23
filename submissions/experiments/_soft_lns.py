"""Soft-macro LNS: destroy a connected subset of softs, re-insert greedily.

Softs have no overlap constraint, so reinsertion is trivial (any position
in canvas is feasible). We pick a connected subset, snapshot positions,
try best-of-K candidate positions for each, commit if total improves.

Because softs are the dominant cost source (set_position of softs changes
HPWL and congestion through net routing), moving them in concert can
unlock local minima that our per-macro soft CD misses.
"""
import time
import numpy as np


def soft_lns_phase(pos_np, benchmark, incr_eval, max_time,
                   n_destroy=8, n_candidates=30, verbose=False):
    """LNS over soft macros. pos_np is hard positions (unchanged)."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if len(soft_movable) < 2:
        return pos_np, incr_eval.get_proxy_cost()

    incr_eval.sync_positions(pos_np)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost

    rng = np.random.RandomState(9876)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        n_iter += 1
        # Pick a connected subset via net-BFS
        seed = int(rng.choice(soft_movable))
        subset = [seed]
        visited = {seed}
        queue = [seed]
        while queue and len(subset) < n_destroy:
            m = queue.pop(0)
            nbrs = []
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if m2 >= n_hard and not macro_fixed[m2] and m2 not in visited:
                        nbrs.append(m2)
            rng.shuffle(nbrs)
            for n2 in nbrs:
                if len(subset) >= n_destroy:
                    break
                subset.append(n2)
                visited.add(n2)
                queue.append(n2)
        if len(subset) < 2:
            continue

        # Snapshot and reinsert
        rng.shuffle(subset)
        move_trail = []
        moved_any = False

        for m in subset:
            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])

            # Net-centroid bias
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if m2 != m:
                        cx_sum += float(incr_eval.macro_pos[m2, 0])
                        cy_sum += float(incr_eval.macro_pos[m2, 1])
                        cnt += 1
            cands = []
            if cnt > 0:
                cx_c = cx_sum / cnt
                cy_c = cy_sum / cnt
                for _ in range(8):
                    cands.append((cx_c + rng.uniform(-1.0, 1.0),
                                  cy_c + rng.uniform(-1.0, 1.0)))
            for _ in range(n_candidates - 8):
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
        print(f"    soft_lns: {n_iter} iters, {n_accepted} accepted, "
              f"final {best_cost:.6f}")
    return pos_np, best_cost
