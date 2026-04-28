"""Novel move operators not present in the current pipeline.

Each takes (pos_np, benchmark, incr_eval, max_time) and returns
(new_pos_np, new_cost). Leaves incr_eval synced to returned pos.
"""

import time
import numpy as np


def _build_adjacency(incr_eval, n_hard):
    """Net-based adjacency between hard macros via incr_eval data."""
    adj = [set() for _ in range(n_hard)]
    for nid in range(incr_eval.n_nets):
        hard_in_net = [m for m in incr_eval.net_macros[nid] if 0 <= m < n_hard]
        for a in hard_in_net:
            for b in hard_in_net:
                if a != b:
                    adj[a].add(b)
    return adj


def cluster_translate_phase(pos_np, benchmark, incr_eval, max_time,
                            max_cluster_size=6, verbose=False):
    """Try translating connected macro clusters as rigid units.

    Addresses CD's fundamental weakness: CD cannot discover correlated
    multi-macro moves. If a group of connected macros would benefit from
    a common shift (e.g., pulled toward an IO port, or shifted out of a
    congestion hotspot), per-macro CD won't find it because each individual
    step alone may look neutral or bad.
    """
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    gap = 0.05
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()

    # Clusters via BFS on net-adjacency, starting from highest-connectivity nodes.
    adj = _build_adjacency(incr_eval, n_hard)
    degree = [len(adj[i]) for i in range(n_hard)]
    assigned = [False] * n_hard
    clusters = []
    for start in sorted(range(n_hard), key=lambda i: -degree[i]):
        if assigned[start] or not movable[start]:
            continue
        cluster = [start]
        assigned[start] = True
        queue = [start]
        while queue and len(cluster) < max_cluster_size:
            node = queue.pop(0)
            for nbr in sorted(adj[node], key=lambda x: -degree[x]):
                if not assigned[nbr] and movable[nbr] and len(cluster) < max_cluster_size:
                    cluster.append(nbr)
                    assigned[nbr] = True
                    queue.append(nbr)
        if len(cluster) >= 2:
            clusters.append(cluster)

    if verbose:
        print(f"    cluster_translate: {len(clusters)} clusters, "
              f"sizes {[len(c) for c in clusters[:10]]}...")

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = pos.copy()

    # Delta magnitudes (in microns) — probe short and long range.
    # Using absolute deltas keeps intra-cluster gaps invariant so rigid
    # translation preserves internal non-overlaps.
    deltas = [
        (0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3),
        (0.7, 0), (-0.7, 0), (0, 0.7), (0, -0.7),
        (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5),
        (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5),
    ]

    t0 = time.time()
    n_improved = 0
    n_tried = 0
    pass_num = 0
    while time.time() - t0 < max_time:
        pass_num += 1
        pass_improved = False
        for cluster in clusters:
            if time.time() - t0 > max_time:
                break
            cluster_set = set(cluster)
            for dx, dy in deltas:
                n_tried += 1
                # Feasibility: canvas bounds + non-cluster non-overlap
                feasible = True
                new_coords = []
                for i in cluster:
                    nx = pos[i, 0] + dx
                    ny = pos[i, 1] + dy
                    if nx < half_w[i] or nx > cw - half_w[i]:
                        feasible = False
                        break
                    if ny < half_h[i] or ny > ch - half_h[i]:
                        feasible = False
                        break
                    # Check overlap vs non-cluster macros
                    ddx = np.abs(nx - pos[:, 0])
                    ddy = np.abs(ny - pos[:, 1])
                    ov = (ddx < sep_x[i] + gap) & (ddy < sep_y[i] + gap)
                    ov[i] = False
                    # Mask out cluster members (handled by rigid translation)
                    for k in cluster_set:
                        ov[k] = False
                    if ov.any():
                        feasible = False
                        break
                    new_coords.append((nx, ny))
                if not feasible:
                    continue

                # Apply all moves
                saved = [(float(pos[i, 0]), float(pos[i, 1])) for i in cluster]
                for (nx, ny), i in zip(new_coords, cluster):
                    incr_eval.move_macro(i, nx, ny)
                    pos[i, 0] = nx
                    pos[i, 1] = ny
                new_cost = incr_eval.get_proxy_cost()

                if new_cost < current_cost - 1e-7:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_pos = pos.copy()
                        n_improved += 1
                        pass_improved = True
                else:
                    for (ox, oy), i in zip(saved, cluster):
                        incr_eval.move_macro(i, ox, oy)
                        pos[i, 0] = ox
                        pos[i, 1] = oy
        if not pass_improved:
            break

    if verbose:
        print(f"    cluster_translate: {n_improved}/{n_tried} accepted, "
              f"cost {current_cost:.6f} (best {best_cost:.6f}), {pass_num} passes")

    # Sync incr_eval to best_pos
    incr_eval.sync_positions(best_pos)
    return best_pos, best_cost


def lns_destroy_repair_phase(pos_np, benchmark, incr_eval, max_time,
                             n_destroy=4, n_candidates=40, verbose=False):
    """Ruin-and-recreate: remove a connected subset, re-insert greedily.

    Novel move type: the subset's positions are re-chosen jointly in
    random order, each at the best-of-K candidate that's non-overlapping
    and minimizes current proxy cost. Creates coupled multi-macro moves
    beyond what CD or swaps can do.
    """
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    gap = 0.05
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = pos.copy()

    rng = np.random.RandomState(12345)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0

    while time.time() - t0 < max_time:
        n_iter += 1
        # Choose a connected seed + neighbors as the "destroy set"
        seed = int(rng.choice(movable_idx))
        subset = [seed]
        visited = {seed}
        queue = [seed]
        while queue and len(subset) < n_destroy:
            m = queue.pop(0)
            nbrs = []
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if 0 <= m2 < n_hard and movable[m2] and m2 not in visited:
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

        # Snapshot positions for rollback
        snapshot = [(m, float(pos[m, 0]), float(pos[m, 1])) for m in subset]

        # Reinsert in random order, best-of-K candidates
        insertion_order = list(subset)
        rng.shuffle(insertion_order)
        move_trail = []  # (macro, old_x, old_y) in order applied

        for m in insertion_order:
            old_x = float(pos[m, 0])
            old_y = float(pos[m, 1])
            # Candidate positions: mix of (current ± small, random, net-centroid)
            cands = []
            # Net-centroid bias (average of connected macros' positions)
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                for m2 in incr_eval.net_macros[nid]:
                    if 0 <= m2 < n_hard and m2 != m:
                        cx_sum += pos[m2, 0]
                        cy_sum += pos[m2, 1]
                        cnt += 1
            if cnt > 0:
                cx_sum /= cnt
                cy_sum /= cnt
                # Add a few jittered net-centroid candidates
                for _ in range(6):
                    jx = rng.uniform(-1.0, 1.0)
                    jy = rng.uniform(-1.0, 1.0)
                    cands.append((cx_sum + jx, cy_sum + jy))
            # Random uniform candidates
            for _ in range(n_candidates - 6):
                cands.append((rng.uniform(half_w[m], cw - half_w[m]),
                              rng.uniform(half_h[m], ch - half_h[m])))

            # Evaluate each candidate
            best_cx, best_cy, best_c = old_x, old_y, current_cost
            kept_old = True
            for cx, cy in cands:
                cx = float(np.clip(cx, half_w[m], cw - half_w[m]))
                cy = float(np.clip(cy, half_h[m], ch - half_h[m]))
                # overlap check vs all macros
                ddx = np.abs(cx - pos[:, 0])
                ddy = np.abs(cy - pos[:, 1])
                ov = (ddx < sep_x[m] + gap) & (ddy < sep_y[m] + gap)
                ov[m] = False
                if ov.any():
                    continue
                c = incr_eval.move_macro(m, cx, cy)
                if c < best_c:
                    best_c = c
                    best_cx = cx
                    best_cy = cy
                    kept_old = False
                incr_eval.undo_move()

            # Commit best candidate (best_c might equal current cost, in which case skip)
            if not kept_old:
                incr_eval.move_macro(m, best_cx, best_cy)
                move_trail.append((m, old_x, old_y))
                pos[m, 0] = best_cx
                pos[m, 1] = best_cy
                current_cost = best_c

        # Check overall improvement
        final_cost = incr_eval.get_proxy_cost()
        if final_cost < best_cost - 1e-7:
            best_cost = final_cost
            best_pos = pos.copy()
            n_accepted += 1
        else:
            # Roll back the trail (in reverse order)
            for m, ox, oy in reversed(move_trail):
                incr_eval.move_macro(m, ox, oy)
                pos[m, 0] = ox
                pos[m, 1] = oy
            current_cost = best_cost

    if verbose:
        print(f"    lns: {n_iter} iters, {n_accepted} accepted, final {best_cost:.6f}")

    incr_eval.sync_positions(best_pos)
    return best_pos, best_cost


def gradient_step_phase(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """Gradient-steered per-macro moves: probe ±eps in x, y to compute
    the local proxy-cost gradient, then step along -gradient with adaptive
    magnitude. Different from 8-direction CD probe in that the step direction
    is *not constrained* to 8 axes.
    """
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    gap = 0.05
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    # Most-connected first
    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_hard)])
    order = sorted(movable_idx.tolist(), key=lambda i: -net_count[i])

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = pos.copy()

    t0 = time.time()
    EPS = 0.2  # μm probe distance
    STEPS = [2.0, 1.0, 0.5, 0.25, 0.12]  # μm step trials
    n_moves = 0

    pass_num = 0
    while time.time() - t0 < max_time:
        pass_num += 1
        pass_improved = False
        for i in order:
            if time.time() - t0 > max_time:
                break
            ox, oy = float(pos[i, 0]), float(pos[i, 1])

            # Probe +x
            def probe(nx, ny):
                # Check canvas
                if nx < half_w[i] or nx > cw - half_w[i]:
                    return None
                if ny < half_h[i] or ny > ch - half_h[i]:
                    return None
                ddx = np.abs(nx - pos[:, 0])
                ddy = np.abs(ny - pos[:, 1])
                ov = (ddx < sep_x[i] + gap) & (ddy < sep_y[i] + gap)
                ov[i] = False
                if ov.any():
                    return None
                c = incr_eval.move_macro(i, nx, ny)
                incr_eval.undo_move()
                return c

            cx_p = probe(ox + EPS, oy)
            cx_m = probe(ox - EPS, oy)
            cy_p = probe(ox, oy + EPS)
            cy_m = probe(ox, oy - EPS)

            # Compute gradient (only using valid probes)
            gx = None
            gy = None
            if cx_p is not None and cx_m is not None:
                gx = (cx_p - cx_m) / (2 * EPS)
            elif cx_p is not None:
                gx = (cx_p - current_cost) / EPS
            elif cx_m is not None:
                gx = (current_cost - cx_m) / EPS
            if cy_p is not None and cy_m is not None:
                gy = (cy_p - cy_m) / (2 * EPS)
            elif cy_p is not None:
                gy = (cy_p - current_cost) / EPS
            elif cy_m is not None:
                gy = (current_cost - cy_m) / EPS

            if gx is None and gy is None:
                continue
            gx = gx or 0.0
            gy = gy or 0.0
            gmag = (gx * gx + gy * gy) ** 0.5
            if gmag < 1e-8:
                continue

            # Unit direction along -gradient
            ux = -gx / gmag
            uy = -gy / gmag

            # Try multiple step magnitudes, take best improver
            best_step_cost = current_cost
            best_step_nx = None
            best_step_ny = None
            for s in STEPS:
                nx = float(np.clip(ox + s * ux, half_w[i], cw - half_w[i]))
                ny = float(np.clip(oy + s * uy, half_h[i], ch - half_h[i]))
                if abs(nx - ox) < 0.001 and abs(ny - oy) < 0.001:
                    continue
                ddx = np.abs(nx - pos[:, 0])
                ddy = np.abs(ny - pos[:, 1])
                ov = (ddx < sep_x[i] + gap) & (ddy < sep_y[i] + gap)
                ov[i] = False
                if ov.any():
                    continue
                c = incr_eval.move_macro(i, nx, ny)
                if c < best_step_cost:
                    best_step_cost = c
                    best_step_nx = nx
                    best_step_ny = ny
                incr_eval.undo_move()

            if best_step_nx is not None and best_step_cost < current_cost - 1e-7:
                incr_eval.move_macro(i, best_step_nx, best_step_ny)
                pos[i, 0] = best_step_nx
                pos[i, 1] = best_step_ny
                current_cost = best_step_cost
                if best_step_cost < best_cost:
                    best_cost = best_step_cost
                    best_pos = pos.copy()
                pass_improved = True
                n_moves += 1
        if not pass_improved:
            break

    if verbose:
        print(f"    gradient_step: {pass_num} passes, {n_moves} moves, cost {best_cost:.6f}")
    return best_pos, best_cost
