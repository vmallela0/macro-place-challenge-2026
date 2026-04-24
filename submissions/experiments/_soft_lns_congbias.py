"""Soft-macro LNS with net-count-weighted seed selection (idea A19 proxy).

Diff from _soft_lns.py: instead of uniform random seed, sample seed
proportional to sum of incident-net sizes (proxy for congestion
contribution). Rationale: LNS from a highly-connected macro destroys a
high-leverage neighborhood; uniform sampling wastes iterations on
peripheral macros.
"""
import time
import numpy as np


def soft_lns_congbias_phase(pos_np, benchmark, incr_eval, max_time,
                             n_destroy=8, n_candidates=30, verbose=False):
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

    # Seed-selection weights = sum of incident net sizes (larger net = more impact)
    weights = np.array([
        sum(len(incr_eval.net_macros[nid]) for nid in incr_eval.macro_nets[i])
        for i in soft_movable
    ], dtype=np.float64)
    weights = np.maximum(weights, 1.0)
    probs = weights / weights.sum()

    rng = np.random.RandomState(9876)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        n_iter += 1
        seed_idx = int(rng.choice(len(soft_movable), p=probs))
        seed = soft_movable[seed_idx]
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

        # Snapshot + destroy-repair (identical to _soft_lns.py from here on)
        snap = {m: (float(incr_eval.macro_pos[m, 0]), float(incr_eval.macro_pos[m, 1])) for m in subset}
        snap_cost = current_cost

        for m in subset:
            x = rng.uniform(0, cw)
            y = rng.uniform(0, ch)
            incr_eval.macro_pos[m, 0] = x
            incr_eval.macro_pos[m, 1] = y
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()

        improved_in_repair = False
        for m in subset:
            best_c = incr_eval.get_proxy_cost()
            best_x = float(incr_eval.macro_pos[m, 0])
            best_y = float(incr_eval.macro_pos[m, 1])
            for _ in range(n_candidates):
                cx = rng.uniform(0, cw)
                cy = rng.uniform(0, ch)
                c = incr_eval.move_macro(m, cx, cy)
                if c < best_c:
                    best_c = c
                    best_x = cx
                    best_y = cy
                incr_eval.undo_move()
            incr_eval.move_macro(m, best_x, best_y)
        new_cost = incr_eval.get_proxy_cost()

        if new_cost < snap_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
            n_accepted += 1
        else:
            # Revert snapshot
            for m, (ox, oy) in snap.items():
                incr_eval.macro_pos[m, 0] = ox
                incr_eval.macro_pos[m, 1] = oy
            incr_eval._recompute_pin_positions()
            incr_eval._full_recompute_wl()
            incr_eval._full_recompute_density()
            incr_eval._full_recompute_congestion()
            current_cost = snap_cost

    if verbose:
        print(f"    soft_lns_congbias: {n_iter} iters, {n_accepted} accepted, cost {current_cost:.6f}")
    return pos_np, best_cost
