"""Exhaustive permutation swap on small hard-macro subsets.

Pick K=4 connected hard macros. For each of 4!=24 permutations of their
positions (swapping positions amongst themselves only), evaluate.
Accept the best valid permutation.

Unlike regular pairwise swap, this finds cyclical improvements (A→B→C→A)
that pair-only swap misses.
"""
import time
import itertools
import numpy as np


def permutation_swap_hard(pos_np, benchmark, incr_eval, max_time,
                          k=4, verbose=False):
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    gap = 0.05
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_set = set(np.where(movable)[0].tolist())

    # Build net adjacency for hards
    adj = {i: set() for i in movable_set}
    for nid in range(incr_eval.n_nets):
        hard_in_net = [m for m in incr_eval.net_macros[nid] if m in movable_set]
        for a in hard_in_net:
            for b in hard_in_net:
                if a != b:
                    adj[a].add(b)

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    rng = np.random.RandomState(2024)
    t0 = time.time()
    n_tried = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        # Pick a random starting hard macro and BFS k-1 neighbors
        if not movable_set:
            break
        seed = rng.choice(list(movable_set))
        subset = [seed]
        visited = {seed}
        queue = [seed]
        while queue and len(subset) < k:
            m = queue.pop(0)
            for nbr in sorted(adj[m], key=lambda x: rng.random()):
                if nbr not in visited and len(subset) < k:
                    subset.append(nbr)
                    visited.add(nbr)
                    queue.append(nbr)
        if len(subset) < 3:
            continue

        # Original positions
        orig = [(m, float(incr_eval.macro_pos[m, 0]), float(incr_eval.macro_pos[m, 1]))
                for m in subset]

        # Try all permutations; identity is first in permutations(), skip it
        for perm in itertools.permutations(range(len(subset))):
            if perm == tuple(range(len(subset))):
                continue  # identity
            n_tried += 1
            # Each macro i gets position of macro at perm[i]
            new_positions = [(subset[i], orig[perm[i]][1], orig[perm[i]][2])
                             for i in range(len(subset))]

            # Check overlap: each macro at new position vs ALL other movable macros
            # Skip check vs cluster members — since cluster is just re-assigning
            # positions within itself, internal non-overlap is preserved only if
            # sizes allow. Need to check.
            feasible = True
            for m, nx, ny in new_positions:
                if nx < half_w[m] or nx > cw - half_w[m] or \
                   ny < half_h[m] or ny > ch - half_h[m]:
                    feasible = False
                    break
                # Against all OTHER macros (not in cluster)
                ddx = np.abs(nx - incr_eval.macro_pos[:n_hard, 0])
                ddy = np.abs(ny - incr_eval.macro_pos[:n_hard, 1])
                ov = (ddx < sep_x[m] + gap) & (ddy < sep_y[m] + gap)
                ov[m] = False
                for k2 in subset:
                    ov[k2] = False
                if ov.any():
                    feasible = False
                    break
            if not feasible:
                continue

            # Need to also check non-overlap WITHIN cluster at new positions
            ok_internal = True
            for ii in range(len(new_positions)):
                mi, nxi, nyi = new_positions[ii]
                for jj in range(ii + 1, len(new_positions)):
                    mj, nxj, nyj = new_positions[jj]
                    if (abs(nxi - nxj) < sep_x[mi, mj] + gap and
                        abs(nyi - nyj) < sep_y[mi, mj] + gap):
                        ok_internal = False
                        break
                if not ok_internal:
                    break
            if not ok_internal:
                continue

            # Apply permutation moves
            saved = [(m, float(incr_eval.macro_pos[m, 0]), float(incr_eval.macro_pos[m, 1]))
                     for m in subset]
            for m, nx, ny in new_positions:
                incr_eval.move_macro(m, nx, ny)
            new_cost = incr_eval.get_proxy_cost()

            if new_cost < current_cost - 1e-7:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                n_accepted += 1
            else:
                # Revert
                for m, ox, oy in saved:
                    incr_eval.move_macro(m, ox, oy)

    if verbose:
        print(f"    perm_swap: {n_tried} tried, {n_accepted} accepted, "
              f"cost {best_cost:.6f}")
    return pos_np, best_cost
