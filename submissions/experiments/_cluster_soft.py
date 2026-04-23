"""Cluster translation of connected soft macros.

Softs have NO overlap constraint, so rigid cluster translation is always
feasible (just check canvas bounds). Moving all softs in a clique by (dx,dy)
together preserves internal net HPWL contributions — only changes external
pins/macros.
"""
import time
import numpy as np


def cluster_soft_translate(pos_np, benchmark, incr_eval, max_time,
                            max_cluster_size=8, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = set(n_hard + i for i in range(n_total - n_hard)
                       if not macro_fixed[n_hard + i])
    if len(soft_movable) < 2:
        return pos_np, incr_eval.get_proxy_cost()

    # Build soft-soft net adjacency (both in same net)
    adj = {m: set() for m in soft_movable}
    for nid in range(incr_eval.n_nets):
        softs_in_net = [m for m in incr_eval.net_macros[nid] if m in soft_movable]
        for a in softs_in_net:
            for b in softs_in_net:
                if a != b:
                    adj[a].add(b)

    # Form clusters via greedy BFS
    degree = {m: len(adj[m]) for m in soft_movable}
    assigned = {m: False for m in soft_movable}
    clusters = []
    for start in sorted(soft_movable, key=lambda m: -degree[m]):
        if assigned[start]:
            continue
        cluster = [start]
        assigned[start] = True
        queue = [start]
        while queue and len(cluster) < max_cluster_size:
            m = queue.pop(0)
            for nbr in sorted(adj[m], key=lambda x: -degree[x]):
                if not assigned[nbr] and len(cluster) < max_cluster_size:
                    cluster.append(nbr)
                    assigned[nbr] = True
                    queue.append(nbr)
        if len(cluster) >= 2:
            clusters.append(cluster)

    deltas = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5),
              (1.0, 0), (-1.0, 0), (0, 1.0), (0, -1.0),
              (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3)]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_improved = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        any_improved = False
        for cluster in clusters:
            if time.time() - t0 > max_time:
                break
            for dx, dy in deltas:
                # Check canvas bounds
                ok = True
                new_coords = []
                for m in cluster:
                    nx = float(incr_eval.macro_pos[m, 0]) + dx
                    ny = float(incr_eval.macro_pos[m, 1]) + dy
                    if nx < 0 or nx > cw or ny < 0 or ny > ch:
                        ok = False
                        break
                    new_coords.append((nx, ny))
                if not ok:
                    continue

                # Apply cluster move (sequential move_macro calls)
                saved = [(m, float(incr_eval.macro_pos[m, 0]),
                             float(incr_eval.macro_pos[m, 1])) for m in cluster]
                for (nx, ny), m in zip(new_coords, cluster):
                    incr_eval.move_macro(m, nx, ny)

                new_cost = incr_eval.get_proxy_cost()

                if new_cost < current_cost - 1e-7:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                    n_improved += 1
                    any_improved = True
                else:
                    # Revert sequentially
                    for m, ox, oy in saved:
                        incr_eval.move_macro(m, ox, oy)
        if not any_improved:
            break

    if verbose:
        print(f"    cluster_soft: {len(clusters)} clusters, {pass_num} passes, "
              f"{n_improved} improved, cost {best_cost:.6f}")
    return pos_np, best_cost
