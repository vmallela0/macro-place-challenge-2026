"""GPU mass-coordinate-descent: propose K candidates per macro per pass,
score on GPU via MLXBatchEvaluator (HPWL exact + density exact + congestion
approx), accept GPU-best iff the CPU-exact `IncrementalEvaluator` confirms
improvement.

Why mass-CD wins over 8-direction lattice CD:
- 8-direction lattice with one delta = 8 evals/macro. Most are no-improvement.
- Mass-CD: K=1024 candidates per macro per pass, structured proposal mix.
  Among 1024, the chance of finding a strict improvement is much higher
  per macro-pass, and the search basins overlap less.
- Cost: 1024 GPU evals (~20 ms total at 50k evals/s) + 1 CPU validate
  (~0.3 ms). Same wall-clock per macro as the 8-direction lattice ran on
  CPU, but the proposal density is 128× larger.

Acceptance is strictly against the CPU-exact proxy_cost — the GPU only
ranks. Rejected candidates incur no cost beyond the GPU score.
"""
from __future__ import annotations
import time
import math
import numpy as np
import mlx.core as mx


def gpu_mass_cd(pos_np, benchmark, plc_eval, *, incr_eval, gpu_eval,
                max_time, K=512, top_validate=4, verbose=False, seed=0,
                sa_T0=None, sa_cooling=0.9995,
                stop_on_plateau_passes=None):
    """Multi-scale mass-CD driven by the MLX batch evaluator.

    Per macro per pass: build K candidates spanning ALL proposal scales
    (lattice at 5 deltas + narrow/medium/wide Gaussian + uniform-canvas) in
    a single GPU call. Validate top-`top_validate` on CPU, accept best.

    Compared to a sigma-frac sweep (5 GPU calls per macro per pass), this
    amortizes the GPU dispatch overhead and exploits MLX's per-call
    fixed-cost ceiling.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] / 2.0, sizes[:, 1] / 2.0
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]

    # Connectivity ordering (most-connected first), as in v4.
    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_hard)])
    movable_sorted = sorted(movable_idx.tolist(), key=lambda i: -net_count[i])

    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    gap = 0.05  # match v4 overlap slack

    # sync incremental state to incoming pos
    incr_eval.sync_positions(pos_np)
    gpu_eval.notify_full_resync()

    rng = np.random.default_rng(seed)
    canvas_diag = math.hypot(cw, ch)

    # Multi-scale proposal: ALL deltas + Gaussians in one batch.
    # Lattice 8 dirs × 5 deltas = 40 ; narrow Gaussian (1 sigma_macro) × Q ;
    # medium Gaussian (canvas/8) × Q ; wide Gaussian (canvas/3) × Q ;
    # uniform canvas × Q. Q chosen so total ~= K.
    deltas = np.array([2.0, 1.0, 0.5, 0.2, 0.05], dtype=np.float64)
    axis = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                     (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)], dtype=np.float64)
    K_lat = axis.shape[0] * deltas.shape[0]
    Q = max(16, (K - K_lat) // 4)
    K_total = K_lat + 4 * Q

    current_cost = float(incr_eval.get_proxy_cost())
    best_pos = pos_np.copy()
    best_cost = current_cost

    sa_enabled = sa_T0 is not None and float(sa_T0) > 0.0
    sa_T = float(sa_T0) if sa_enabled else 0.0

    t0 = time.time()
    pass_num = 0
    plateau_passes = 0
    n_total_validates = 0
    n_accepts = 0
    n_gpu_calls = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        any_improve = False
        for i in movable_sorted:
            if time.time() - t0 > max_time:
                break

            old_x = float(pos_np[i, 0])
            old_y = float(pos_np[i, 1])
            macro_max_dim = float(max(sizes[i, 0], sizes[i, 1]))

            # Lattice block: 8 dirs × len(deltas) points around (old_x,old_y)
            lat_d = np.outer(deltas, axis[:, 0]).ravel() * macro_max_dim
            lat_dy_arr = np.outer(deltas, axis[:, 1]).ravel() * macro_max_dim
            lat = np.column_stack([old_x + lat_d, old_y + lat_dy_arr])

            # Narrow / medium / wide Gaussians
            ng = rng.normal(0, macro_max_dim * 0.5, (Q, 2))
            mg = rng.normal(0, canvas_diag * 0.125, (Q, 2))
            wg = rng.normal(0, canvas_diag * 0.33, (Q, 2))
            narrow = np.column_stack([old_x + ng[:, 0], old_y + ng[:, 1]])
            med = np.column_stack([old_x + mg[:, 0], old_y + mg[:, 1]])
            wide_g = np.column_stack([old_x + wg[:, 0], old_y + wg[:, 1]])
            uni = np.column_stack([rng.uniform(half_w[i], cw - half_w[i], Q),
                                   rng.uniform(half_h[i], ch - half_h[i], Q)])

            cands_np = np.concatenate([lat, narrow, med, wide_g, uni], axis=0).astype(np.float32)
            cands_np[:, 0] = np.clip(cands_np[:, 0], half_w[i], cw - half_w[i])
            cands_np[:, 1] = np.clip(cands_np[:, 1], half_h[i], ch - half_h[i])

            cands_mx = mx.array(cands_np)
            scores = gpu_eval.score_candidates(i, cands_mx)
            mx.eval(scores)
            scores_np = np.asarray(scores)
            n_gpu_calls += 1

            # Skip trivial candidates
            trivial = (np.abs(cands_np[:, 0] - old_x) < 1e-3) & \
                      (np.abs(cands_np[:, 1] - old_y) < 1e-3)
            scores_np = np.where(trivial, np.inf, scores_np)

            # Top-T candidates
            T = top_validate
            topk_idx = np.argpartition(scores_np, T)[:T]
            topk_idx = topk_idx[np.argsort(scores_np[topk_idx])]

            for cand_b in topk_idx:
                # If SA is enabled, also consider candidates GPU thinks are
                # *worse* than current — Metropolis will probabilistically
                # accept based on the CPU-exact delta.
                if not sa_enabled and scores_np[cand_b] >= current_cost - 1e-9:
                    break
                nx = float(cands_np[cand_b, 0])
                ny = float(cands_np[cand_b, 1])
                if _check_overlap_static(pos_np, sep_x[i], sep_y[i], i, nx, ny, gap):
                    continue
                new_cost = float(incr_eval.move_macro(i, nx, ny))
                n_total_validates += 1
                delta_c = new_cost - current_cost
                if delta_c < -1e-9:
                    accept = True
                elif sa_enabled and sa_T > 1e-18:
                    try:
                        accept = rng.random() < math.exp(-delta_c / sa_T)
                    except OverflowError:
                        accept = False
                else:
                    accept = False
                if accept:
                    pos_np[i, 0] = nx
                    pos_np[i, 1] = ny
                    current_cost = new_cost
                    gpu_eval.notify_committed_move(i)
                    n_accepts += 1
                    if delta_c < -1e-9:
                        any_improve = True
                    if sa_enabled:
                        sa_T *= sa_cooling
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_pos = pos_np.copy()
                    break
                else:
                    incr_eval.undo_move()

        if verbose:
            print(f"  [gpu_cd pass {pass_num}] cost={best_cost:.6f} "
                  f"validates={n_total_validates} accepts={n_accepts} "
                  f"gpu_calls={n_gpu_calls} T={sa_T:.2e}" if sa_enabled
                  else f"  [gpu_cd pass {pass_num}] cost={best_cost:.6f} "
                  f"validates={n_total_validates} accepts={n_accepts} "
                  f"gpu_calls={n_gpu_calls}", flush=True)
        if not any_improve:
            plateau_passes += 1
            if stop_on_plateau_passes is not None and \
                    plateau_passes >= stop_on_plateau_passes:
                break
        else:
            plateau_passes = 0

    if verbose:
        print(f"  [gpu_cd] done in {time.time()-t0:.1f}s, "
              f"final cost={best_cost:.6f} accepts={n_accepts} "
              f"gpu_calls={n_gpu_calls} validates={n_total_validates}", flush=True)
    # Sync the incremental evaluator to best_pos (might differ from current
    # if best was strictly better than the last accepted state).
    incr_eval.sync_positions(best_pos)
    gpu_eval.notify_full_resync()
    return best_pos, best_cost


def _check_overlap_static(pos_np, sep_x_row, sep_y_row, i, nx, ny, gap):
    """Returns True if moving macro i to (nx,ny) overlaps any other macro."""
    ddx = np.abs(nx - pos_np[:, 0])
    ddy = np.abs(ny - pos_np[:, 1])
    o = (ddx < sep_x_row + gap) & (ddy < sep_y_row + gap)
    o[i] = False
    return bool(o.any())
