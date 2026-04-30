"""Top-K congestion eviction — Rapid Experiment #2.

Mechanism: greedy directed escape from the hot tail of the congestion grid.
At each pass:
    1. Recompute per-cell V + H routing demand on current placement.
    2. Identify the top-K (5%) hottest cells (V or H direction).
    3. For each soft macro whose footprint touches a hot cell, search
       within radius R for the cell with the LOWEST congestion that
       doesn't introduce overlap.
    4. Try the move via the official compute_proxy_cost (full re-routing).
    5. Strict accept on (overlap == 0) AND (cost strictly improves).

Distinct from coordinate descent because:
    - Search is FOCUSED on hot-tail softs, not all softs.
    - Direction is INFORMED by the per-cell congestion gradient (move to
      coolest cell), not random/lattice candidates.
    - Single-pass evicts the entire tail in one batch; CD evicts one at
      a time and allows neighboring softs to drift back into the hot cell.

Why this might work where Adam/basin-hop didn't:
    - The smooth-vs-exact divergence problem doesn't apply: every move
      is validated by the EXACT scorer.
    - The local-minimizer plateau problem doesn't apply: we're not
      running any minimizer. We're directly using the EXACT cost to
      gate moves.
    - The basin barrier doesn't apply: we're not crossing barriers,
      we're directly relocating the cause of the high-cost top-K cells.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def evict_hot_softs(
    incr_eval,
    benchmark,
    plc,
    *,
    top_k_frac: float = 0.05,
    radius_cells: int = 5,
    n_passes: int = 3,
    max_softs_per_pass: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Run greedy top-K eviction for n_passes.

    Returns (best_pos, best_cost, n_accepted_total).
    incr_eval is mutated to the best state.
    """
    from macro_place.objective import compute_proxy_cost

    n_hard = incr_eval.n_hard
    n_total = incr_eval.macro_pos.shape[0]
    n_soft = n_total - n_hard
    grid_col = incr_eval.grid_col
    grid_row = incr_eval.grid_row
    grid_w = incr_eval.grid_width
    grid_h = incr_eval.grid_height
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    initial_pos = np.array(incr_eval.macro_pos).copy()
    initial_tensor = torch.tensor(initial_pos, dtype=torch.float32)
    r0 = compute_proxy_cost(initial_tensor, benchmark, plc)
    initial_cost = float(r0["proxy_cost"])
    initial_overlaps = int(r0["overlap_count"])
    if initial_overlaps != 0:
        if verbose:
            print(f"  [evict] starting state has {initial_overlaps} overlaps; "
                  f"cannot run eviction safely", flush=True)
        return initial_pos, initial_cost, 0

    best_pos = initial_pos.copy()
    best_cost = initial_cost
    n_accepted_total = 0
    t0 = time.time()
    if verbose:
        print(f"  [evict] start cost={best_cost:.6f}, n_passes={n_passes}, "
              f"top_k={top_k_frac:.2%}, R={radius_cells} cells",
              flush=True)

    for pass_i in range(n_passes):
        # ── Refresh per-cell congestion grid via the incr eval ────────
        incr_eval.macro_pos[:] = best_pos
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()

        V = (np.asarray(incr_eval.V_routing_smooth)
             + np.asarray(incr_eval.V_macro_raw)
               / np.maximum(np.asarray(incr_eval.vrouting_alloc), 1e-9))
        H = (np.asarray(incr_eval.H_routing_smooth)
             + np.asarray(incr_eval.H_macro_raw)
               / np.maximum(np.asarray(incr_eval.hrouting_alloc), 1e-9))
        # per-cell, the dominant direction's value
        cong = np.maximum(V, H)

        # ── Top-K threshold ──────────────────────────────────────────
        K = max(1, int(top_k_frac * cong.size))
        threshold = np.partition(cong, -K)[-K]

        # ── Score each soft by max-cong-in-footprint ────────────────
        sw_arr = np.asarray(incr_eval.macro_w)
        sh_arr = np.asarray(incr_eval.macro_h)
        soft_scores = np.zeros(n_soft, dtype=np.float64)
        soft_fp_c0 = np.zeros(n_soft, dtype=np.int64)
        soft_fp_r0 = np.zeros(n_soft, dtype=np.int64)
        soft_fp_c1 = np.zeros(n_soft, dtype=np.int64)
        soft_fp_r1 = np.zeros(n_soft, dtype=np.int64)
        for si in range(n_soft):
            s = n_hard + si
            x, y = best_pos[s, 0], best_pos[s, 1]
            sw, sh = sw_arr[s], sh_arr[s]
            c0 = max(0, int(x / grid_w))
            c1 = min(grid_col - 1, int((x + sw) / grid_w))
            r0_idx = max(0, int(y / grid_h))
            r1_idx = min(grid_row - 1, int((y + sh) / grid_h))
            soft_fp_c0[si] = c0
            soft_fp_c1[si] = c1
            soft_fp_r0[si] = r0_idx
            soft_fp_r1[si] = r1_idx
            max_cong = 0.0
            for c in range(c0, c1 + 1):
                for rrow in range(r0_idx, r1_idx + 1):
                    val = cong[rrow * grid_col + c]
                    if val > max_cong:
                        max_cong = val
            soft_scores[si] = max_cong

        # ── Identify candidates ─────────────────────────────────────
        candidate_softs_idx = np.where(soft_scores >= threshold)[0]
        # Sort by score descending — try the worst ones first
        candidate_softs_idx = candidate_softs_idx[
            np.argsort(-soft_scores[candidate_softs_idx])]
        if max_softs_per_pass is not None:
            candidate_softs_idx = candidate_softs_idx[:max_softs_per_pass]
        if verbose:
            print(f"  [evict] pass {pass_i}: {len(candidate_softs_idx)} "
                  f"hot-tail softs (threshold cong={threshold:.3f}, "
                  f"max={soft_scores.max():.3f})", flush=True)

        moves_accepted = 0
        moves_attempted = 0
        for si in candidate_softs_idx:
            s = n_hard + si
            sw, sh = sw_arr[s], sh_arr[s]
            c_lo, c_hi = soft_fp_c0[si], soft_fp_c1[si]
            r_lo, r_hi = soft_fp_r0[si], soft_fp_r1[si]
            cur_max_cong = soft_scores[si]

            # Find the coolest cell within radius_cells (Manhattan-ish).
            best_target_x = best_pos[s, 0]
            best_target_y = best_pos[s, 1]
            best_target_cong = cur_max_cong
            R = radius_cells
            for dc in range(-R, R + 1):
                for drow in range(-R, R + 1):
                    if dc == 0 and drow == 0:
                        continue
                    new_c = c_lo + dc
                    new_rrow = r_lo + drow
                    if new_c < 0 or new_c >= grid_col:
                        continue
                    if new_rrow < 0 or new_rrow >= grid_row:
                        continue
                    new_cong = cong[new_rrow * grid_col + new_c]
                    if new_cong >= best_target_cong:
                        continue
                    # Tentative target: lower-left corner at this cell
                    tx = new_c * grid_w
                    ty = new_rrow * grid_h
                    # Clip to canvas
                    tx = min(max(tx, 0.0), cw - sw)
                    ty = min(max(ty, 0.0), ch - sh)
                    best_target_cong = new_cong
                    best_target_x = tx
                    best_target_y = ty

            if (best_target_x == best_pos[s, 0]
                    and best_target_y == best_pos[s, 1]):
                continue   # no cooler cell found

            # Try the move via official compute_proxy_cost (full re-routing).
            moves_attempted += 1
            trial_pos = best_pos.copy()
            trial_pos[s, 0] = best_target_x
            trial_pos[s, 1] = best_target_y
            trial_tensor = torch.tensor(trial_pos, dtype=torch.float32)
            r = compute_proxy_cost(trial_tensor, benchmark, plc)
            trial_cost = float(r["proxy_cost"])
            trial_overlaps = int(r["overlap_count"])

            if trial_overlaps == 0 and trial_cost < best_cost - 1e-7:
                best_pos = trial_pos
                best_cost = trial_cost
                moves_accepted += 1

        n_accepted_total += moves_accepted
        if verbose:
            print(f"  [evict] pass {pass_i}: "
                  f"{moves_accepted}/{moves_attempted} accepted; "
                  f"cost {best_cost:.6f}  ({time.time()-t0:.1f}s)",
                  flush=True)
        if moves_accepted == 0:
            break

    # Sync incr_eval to best
    incr_eval.macro_pos[:] = best_pos
    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()

    return best_pos, best_cost, n_accepted_total
