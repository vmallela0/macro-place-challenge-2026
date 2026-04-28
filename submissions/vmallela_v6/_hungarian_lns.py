"""Hungarian-assignment LNS repair (T1.2).

Replaces v4's `_moves.lns_destroy_repair_phase` greedy random-best-of-K
reinsertion with a min-cost-bipartite-matching:

  1. Pick a destroy set of n_destroy connected macros (BFS over the net
     graph, congestion-biased seed selector).
  2. Generate K shared candidate positions (mix of destroy-set net
     centroids, current positions, and random uniform). K >> n_destroy so
     the matching has slack.
  3. GPU computes an (n_destroy x K) cost matrix where C[i, j] = the
     approximate proxy cost if destroy_set[i] moves alone to candidate[j]
     (other destroy-set macros stay at their current positions during the
     marginal evaluation — Hungarian's separability assumption). One MLX/
     torch dispatch via `score_candidates_multimacro`.
  4. `scipy.optimize.linear_sum_assignment` solves the matching → assigns
     each destroy_set macro to a unique candidate position.
  5. Validate the joint assignment on the CPU IncrementalEvaluator: apply
     all moves, check overlaps & cost. Accept iff (a) zero overlaps,
     (b) joint proxy cost improves. Roll back via `sync_positions` on reject.

Why this beats greedy reinsertion
---------------------------------
v4's reinsertion commits each macro's position before considering the
others — early choices lock the search. Hungarian solves the joint
optimal under the separable-cost approximation; for n_destroy small (5-12)
relative to n_hard (~250), the marginal cost is a good proxy for the
true joint cost.

The Hungarian solution may be infeasible (two macros land overlapping or
the joint cost is worse than expected). When that happens, we fall back
to v4's greedy path for that iteration. Most iterations succeed.
"""
from __future__ import annotations
import time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


# Reuse the congestion-biased seed selector from v4 directly.
def _pick_congestion_seed(incr_eval, movable, hotspot_frac=0.10):
    n_cells = incr_eval.n_cells
    V_total = incr_eval.V_routing_smooth + incr_eval.V_macro_raw / incr_eval.grid_v_routes
    H_total = incr_eval.H_routing_smooth + incr_eval.H_macro_raw / incr_eval.grid_h_routes
    per_cell = np.maximum(V_total, H_total)
    top_n = max(1, int(n_cells * hotspot_frac))
    hot_cells = set(int(c) for c in np.argpartition(per_cell, -top_n)[-top_n:])
    candidates = []
    for m in range(len(movable)):
        if not movable[m]:
            continue
        for flat, _, _ in incr_eval.macro_blockage_cache.get(m, []):
            if flat in hot_cells:
                candidates.append(m)
                break
    if not candidates:
        return None
    return candidates


def _bfs_destroy_set(seed, incr_eval, movable, n_hard, n_destroy, rng):
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
    return subset


def _generate_shared_candidates(destroy_set, pos_np, incr_eval, n_hard,
                                K, half_w, half_h, cw, ch, rng,
                                sizes=None, jitter_sigma_frac=1.5):
    """Produce K shared candidate positions clustered around the destroy
    set's current positions and net-centroids, where free space is most
    likely to exist.

    Composition (total = K):
      - n_destroy: each destroy-macro's *current* position (definitely
        feasible).
      - K_jitter: small-jitter candidates around each destroy-macro's
        current position (sigma proportional to that macro's size).
      - K_centroid: net-centroid of each destroy-macro plus jitter.

    We deliberately drop uniform-canvas candidates: on dense benchmarks
    (ibm10/12/14, hundreds of fixed macros), uniform candidates are almost
    always overlapping a fixed macro and force Hungarian into +inf entries.
    Clustering near destroy-set positions guarantees most candidates are
    feasible, since the destroy-set members were already feasible.
    """
    n_destroy = len(destroy_set)
    cands = []
    # 1) Current positions (keep is always feasible).
    for m in destroy_set:
        cands.append((float(pos_np[m, 0]), float(pos_np[m, 1])))

    # 2) Per-destroy-macro small-jitter candidates.
    # Total budget for this section: ~ 60% of K.
    K_jitter_per = max(4, int((0.6 * K) // max(1, n_destroy)))
    for m in destroy_set:
        if sizes is not None:
            sigma = float(jitter_sigma_frac * max(sizes[m, 0], sizes[m, 1]))
        else:
            sigma = jitter_sigma_frac * 0.5
        cx0 = float(pos_np[m, 0])
        cy0 = float(pos_np[m, 1])
        for _ in range(K_jitter_per):
            jx = rng.normal(0.0, sigma)
            jy = rng.normal(0.0, sigma)
            cands.append((cx0 + jx, cy0 + jy))

    # 3) Net-centroid candidates + jitter.
    K_centroid_per = max(2, (K - len(cands)) // max(1, n_destroy))
    for m in destroy_set:
        cx_sum = cy_sum = 0.0
        cnt = 0
        for nid in incr_eval.macro_nets[m]:
            for m2 in incr_eval.net_macros[nid]:
                if 0 <= m2 < n_hard and m2 != m:
                    cx_sum += pos_np[m2, 0]
                    cy_sum += pos_np[m2, 1]
                    cnt += 1
        if cnt == 0:
            continue
        cx_sum /= cnt
        cy_sum /= cnt
        for _ in range(K_centroid_per):
            jx = rng.uniform(-1.5, 1.5)
            jy = rng.uniform(-1.5, 1.5)
            cands.append((cx_sum + jx, cy_sum + jy))

    # 4) Top up with more small-jitter (in case earlier sections under-filled).
    while len(cands) < K:
        m = destroy_set[rng.randint(0, n_destroy)]
        sigma = (float(jitter_sigma_frac * max(sizes[m, 0], sizes[m, 1]))
                 if sizes is not None else jitter_sigma_frac * 0.5)
        cx0 = float(pos_np[m, 0])
        cy0 = float(pos_np[m, 1])
        cands.append((cx0 + rng.normal(0.0, sigma),
                      cy0 + rng.normal(0.0, sigma)))
    cands = cands[:K]
    return np.asarray(cands, dtype=np.float32)


def hungarian_lns_phase(pos_np, benchmark, incr_eval, gpu_eval, max_time,
                        n_destroy=8, K=128,
                        seed_selector="congestion",
                        rng_seed=12345, verbose=False):
    """Min-cost-bipartite-matching LNS repair on top of GPU candidate scoring.

    Parameters
    ----------
    pos_np : (n_hard, 2) float64
        Current hard positions (modified in-place on accept).
    incr_eval : CPU IncrementalEvaluator (source of truth for accept).
    gpu_eval : TorchBatchEvaluator (used to compute the cost matrix).
    max_time : seconds budget.
    n_destroy : size of the destroy set (BFS over net-graph from a seed).
    K : candidate-pool size (must be >= n_destroy; rectangular Hungarian
        used for K > n_destroy).
    seed_selector : "congestion" (default) or "random".
    """
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    gap = 0.05
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]

    pos = pos_np.copy()
    incr_eval.sync_positions(pos)
    gpu_eval.notify_full_resync()
    current_cost = float(incr_eval.get_proxy_cost())
    best_cost = current_cost
    best_pos = pos.copy()

    rng = np.random.RandomState(rng_seed)
    t0 = time.time()
    n_iter = 0
    n_accepted = 0
    n_infeasible = 0  # Hungarian solution had overlaps or cost regression

    while time.time() - t0 < max_time:
        n_iter += 1

        # 1. Pick destroy set
        if seed_selector == "congestion":
            hot = _pick_congestion_seed(incr_eval, movable)
            seed = int(rng.choice(hot)) if hot else int(rng.choice(movable_idx))
        else:
            seed = int(rng.choice(movable_idx))
        destroy_set = _bfs_destroy_set(seed, incr_eval, movable, n_hard,
                                       n_destroy, rng)
        n_d = len(destroy_set)
        if n_d < 2:
            continue

        # 2. Generate shared K candidate positions (clustered around the
        #    destroy set, where free space is most likely).
        candidates = _generate_shared_candidates(
            destroy_set, pos, incr_eval, n_hard, K,
            half_w, half_h, cw, ch, rng,
            sizes=sizes, jitter_sigma_frac=1.5)
        # Clip to per-macro canvas: since each row uses a different macro's
        # half-extent, we can't pre-clip uniformly. We'll filter infeasible
        # (out-of-canvas) entries by setting their cost matrix entry to +inf
        # below.

        # 3. Build cost matrix on GPU via score_candidates_multimacro.
        # macro_ids[i*K + j] = destroy_set[i]; positions[i*K + j] = candidates[j]
        macro_ids = np.repeat(np.asarray(destroy_set, dtype=np.int64), K)
        cands_flat = np.tile(candidates, (n_d, 1))
        # Per-row canvas clip mask: for entry (i, j), is candidates[j]
        # inside macro destroy_set[i]'s canvas range?
        cand_x = candidates[:, 0]
        cand_y = candidates[:, 1]
        m_arr = np.asarray(destroy_set)
        clip_x = (cand_x[None, :] >= half_w[m_arr][:, None]) & \
                 (cand_x[None, :] <= cw - half_w[m_arr][:, None])
        clip_y = (cand_y[None, :] >= half_h[m_arr][:, None]) & \
                 (cand_y[None, :] <= ch - half_h[m_arr][:, None])
        in_canvas = clip_x & clip_y  # (n_d, K)

        scores = gpu_eval.score_candidates_multimacro(macro_ids, cands_flat)
        C = scores.detach().cpu().numpy().reshape(n_d, K).astype(np.float64)
        # Mark out-of-canvas entries as +inf so Hungarian skips them.
        C[~in_canvas] = 1e9

        # Mask candidates that overlap any NON-destroy macro for each row.
        # For each row i: candidate j is feasible iff macro destroy_set[i]
        # at candidate j doesn't overlap any non-destroy macro.
        non_destroy_mask = np.ones(n_hard, dtype=bool)
        non_destroy_mask[m_arr] = False
        non_destroy_pos = pos[non_destroy_mask]   # (F, 2)
        non_destroy_w = sizes[non_destroy_mask, 0]
        non_destroy_h = sizes[non_destroy_mask, 1]
        for i, mi in enumerate(destroy_set):
            sx = (sizes[mi, 0] + non_destroy_w) / 2.0 + gap   # (F,)
            sy = (sizes[mi, 1] + non_destroy_h) / 2.0 + gap   # (F,)
            # Vectorized over candidates: compute |cand_x[j] - non_destroy_pos[f, 0]|
            # and check overlap. Shape: (K, F)
            ddx = np.abs(cand_x[:, None] - non_destroy_pos[None, :, 0])
            ddy = np.abs(cand_y[:, None] - non_destroy_pos[None, :, 1])
            ov = (ddx < sx[None, :]) & (ddy < sy[None, :])  # (K, F)
            infeasible_j = ov.any(axis=1)                     # (K,)
            C[i, infeasible_j] = 1e9

        # 4. Solve linear sum assignment.
        try:
            row_idx, col_idx = linear_sum_assignment(C)
        except ValueError:
            continue  # no feasible assignment

        # If Hungarian had to use +inf entries (all candidates infeasible
        # for some row), the assignment is unusable.
        chosen_costs = C[row_idx, col_idx]
        if (chosen_costs >= 1e8).any():
            n_infeasible += 1
            continue

        chosen = candidates[col_idx]  # (n_d, 2)

        # 5. Joint within-destroy feasibility check (non-destroy already
        # handled in the cost matrix above).
        overlap_within = False
        for i in range(n_d):
            xi, yi = chosen[i, 0], chosen[i, 1]
            mi = destroy_set[i]
            for k in range(i + 1, n_d):
                mk = destroy_set[k]
                if (abs(xi - chosen[k, 0]) < (sizes[mi, 0] + sizes[mk, 0]) / 2 + gap and
                        abs(yi - chosen[k, 1]) < (sizes[mi, 1] + sizes[mk, 1]) / 2 + gap):
                    overlap_within = True
                    break
            if overlap_within:
                break
        if overlap_within:
            n_infeasible += 1
            continue

        # Commit moves to the CPU evaluator and check joint cost.
        snapshot = pos.copy()
        for i in range(n_d):
            mi = destroy_set[i]
            incr_eval.move_macro(mi, float(chosen[i, 0]), float(chosen[i, 1]))
            pos[mi, 0] = float(chosen[i, 0])
            pos[mi, 1] = float(chosen[i, 1])
        joint_cost = float(incr_eval.get_proxy_cost())
        if joint_cost < current_cost - 1e-7:
            # Accept
            current_cost = joint_cost
            for mi in destroy_set:
                gpu_eval.notify_committed_move(int(mi))
            n_accepted += 1
            if joint_cost < best_cost:
                best_cost = joint_cost
                best_pos = pos.copy()
        else:
            # Roll back via sync (cheaper than n_destroy undo_moves; the CPU
            # eval doesn't stack-undo more than 1 deep).
            pos = snapshot
            incr_eval.sync_positions(pos)
            gpu_eval.notify_full_resync()

    if verbose:
        print(f"  [hungarian_lns] {n_iter} iters, {n_accepted} accepted, "
              f"{n_infeasible} infeasible, final cost={best_cost:.6f}",
              flush=True)
    incr_eval.sync_positions(best_pos)
    gpu_eval.notify_full_resync()
    return best_pos, best_cost
