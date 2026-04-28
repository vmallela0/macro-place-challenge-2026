"""GPU-driven coordinate descent (cross-macro batched).

T1.1+T1.3 implementation. Per delta level, ONE cross-macro GPU batch covering
all movable macros × K candidates each. Pick GPU-best per macro; CPU validate
in macro order; accept iff CPU-exact proxy improves. Optional Metropolis SA
acceptance.

Why cross-macro batching changes the picture
--------------------------------------------
Previous v6 (single-macro batches): 6330 GPU dispatches in 60 s on ibm01,
each ~3-4 ms latency-bound. Throughput effectively 50 k evals/s. GPU CD lost
to CPU CD at fixed budget.

This version: 1 GPU dispatch per delta level (covers all 246 movable macros
× 32 candidates = 7872 scores in ~100 ms on M5 Pro MPS, ~10 ms on RTX 6000
Ada). Throughput ~80 k evals/s on MPS, ~800 k on CUDA. Per-delta-pass goes
from ~750 ms (single-macro) to ~100 ms — 7-8x faster, with strictly larger
candidate density (32 vs 8 per macro).

The delta schedule mirrors v4's CPU CD lattice [5.0, 3.0, 2.0, 1.5, 1.0,
0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02], so the search
covers both long-range escape moves and sub-cell refinement. Each macro's
candidate set per delta is 8 lattice + 8 narrow Gaussian + 8 medium Gaussian
+ 8 uniform-canvas — a structured proposal mix that explores more basins
than v4's pure 8-direction lattice.
"""
from __future__ import annotations
import time
import math
import numpy as np
import torch


_LATTICE_8 = np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                       (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)],
                      dtype=np.float64)


def gpu_mass_cd(pos_np, benchmark, plc_eval, *, incr_eval, gpu_eval,
                max_time, K=32, top_validate=1, verbose=False, seed=0,
                sa_T0=None, sa_cooling=0.9995,
                delta_schedule=None,
                stop_on_plateau_passes=None):
    """Cross-macro batched coordinate descent.

    Parameters
    ----------
    pos_np : (n_hard, 2) float64
        Current hard positions. Updated in-place on accepts.
    incr_eval, gpu_eval : the CPU IncrementalEvaluator and a
        TorchBatchEvaluator wrapping it. The CPU evaluator is the source of
        truth for accept/reject; the GPU only ranks.
    max_time : seconds budget for this phase.
    K : candidates per macro per delta-pass (default 32 = 8 lattice + 8
        narrow Gaussian + 8 medium Gaussian + 8 uniform-canvas).
    top_validate : how many GPU-top candidates to validate on CPU per macro
        per delta. Default 1; set to 2-3 if the GPU's congestion approx
        diverges from CPU exact (rare for single-macro moves).
    sa_T0 : optional Metropolis temperature; matches v4's PLACER_SA_T0.
    delta_schedule : list of float multipliers of macro_max_dim. None →
        v4-matching default.
    stop_on_plateau_passes : if int, exit after N consecutive plateau passes
        (no improvement). None → run until time budget runs out.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0]
    M = movable_idx.shape[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    gap = 0.05  # match v4 overlap slack

    # Connectivity-ordered macros (same as v4 CPU CD).
    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_hard)])
    movable_sorted = np.array(sorted(movable_idx.tolist(),
                                     key=lambda i: -net_count[i]),
                              dtype=np.int64)
    macro_max_dim = np.maximum(sizes[:, 0], sizes[:, 1])

    if delta_schedule is None:
        delta_schedule = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35,
                          0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]

    # Sync state
    incr_eval.sync_positions(pos_np)
    gpu_eval.notify_full_resync()
    current_cost = float(incr_eval.get_proxy_cost())
    best_pos = pos_np.copy()
    best_cost = current_cost

    sa_enabled = sa_T0 is not None and float(sa_T0) > 0.0
    sa_T = float(sa_T0) if sa_enabled else 0.0

    rng = np.random.default_rng(seed)
    canvas_diag = math.hypot(cw, ch)

    t0 = time.time()
    pass_num = 0
    plateau_passes = 0
    n_total_validates = 0
    n_accepts = 0
    n_gpu_calls = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        any_improve = False
        # One GPU dispatch per delta level → covers all M movable macros × K cands.
        for delta in delta_schedule:
            if time.time() - t0 > max_time:
                break

            # Build the (M*K) candidate batch on CPU then upload.
            # K split: 8 lattice + Q narrow + Q medium + (K-8-2Q) uniform.
            K_lat = 8
            Q = max(1, (K - K_lat) // 3)
            K_uni = K - K_lat - 2 * Q

            # Lattice block (M, K_lat, 2): (delta * macro_max_dim_m * dir) per macro
            scale_per_m = (delta * macro_max_dim[movable_sorted])  # (M,)
            lat_off = scale_per_m[:, None, None] * _LATTICE_8[None, :, :]  # (M, 8, 2)
            lat = pos_np[movable_sorted, None, :] + lat_off  # (M, 8, 2)

            # Narrow Gaussian (sigma = delta * macro_max_dim_m)
            sigma_n = scale_per_m[:, None, None]  # (M, 1, 1)
            ng = rng.normal(0, 1, (M, Q, 2)) * sigma_n
            narrow = pos_np[movable_sorted, None, :] + ng  # (M, Q, 2)

            # Medium Gaussian (sigma = delta * canvas_diag/8) — same scale across macros
            sigma_m_scalar = delta * canvas_diag * 0.125
            mg = rng.normal(0, sigma_m_scalar, (M, Q, 2))
            med = pos_np[movable_sorted, None, :] + mg

            # Uniform canvas
            if K_uni > 0:
                ux = rng.uniform(half_w[movable_sorted, None],
                                 cw - half_w[movable_sorted, None],
                                 (M, K_uni))
                uy = rng.uniform(half_h[movable_sorted, None],
                                 ch - half_h[movable_sorted, None],
                                 (M, K_uni))
                uni = np.stack([ux, uy], axis=-1)  # (M, K_uni, 2)
            else:
                uni = np.zeros((M, 0, 2))

            cands_mk = np.concatenate([lat, narrow, med, uni], axis=1)  # (M, K, 2)
            # Clip to canvas
            cands_mk[:, :, 0] = np.clip(cands_mk[:, :, 0],
                                         half_w[movable_sorted, None],
                                         cw - half_w[movable_sorted, None])
            cands_mk[:, :, 1] = np.clip(cands_mk[:, :, 1],
                                         half_h[movable_sorted, None],
                                         ch - half_h[movable_sorted, None])

            # Flatten to (M*K, 2) and build macro_ids vector
            B = M * K
            cands_flat = cands_mk.reshape(B, 2).astype(np.float32)
            macro_ids = np.repeat(movable_sorted, K)

            scores = gpu_eval.score_candidates_multimacro(macro_ids, cands_flat)
            n_gpu_calls += 1
            scores_np = scores.detach().cpu().numpy().reshape(M, K)

            # Mark candidates equal to current pos as +inf (no-op moves).
            cur_x = pos_np[movable_sorted, 0:1]
            cur_y = pos_np[movable_sorted, 1:2]
            trivial = ((np.abs(cands_mk[:, :, 0] - cur_x) < 1e-3) &
                       (np.abs(cands_mk[:, :, 1] - cur_y) < 1e-3))
            scores_np = np.where(trivial, np.inf, scores_np)

            # For each macro: pick top_validate by GPU score; CPU-validate; accept best.
            # Process in connectivity order (movable_sorted).
            for mi in range(M):
                if time.time() - t0 > max_time:
                    break
                i = int(movable_sorted[mi])
                row = scores_np[mi]
                # Top-T candidates by ascending score
                T = top_validate
                if T == 1:
                    cand_idxs = [int(np.argmin(row))]
                else:
                    cand_idxs = np.argpartition(row, T)[:T]
                    cand_idxs = cand_idxs[np.argsort(row[cand_idxs])]

                for cand_b in cand_idxs:
                    if not sa_enabled and row[cand_b] >= current_cost - 1e-9:
                        break
                    nx = float(cands_mk[mi, cand_b, 0])
                    ny = float(cands_mk[mi, cand_b, 1])
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
            T_str = f"T={sa_T:.2e} " if sa_enabled else ""
            print(f"  [gpu_cd pass {pass_num}] cost={best_cost:.6f} "
                  f"{T_str}validates={n_total_validates} accepts={n_accepts} "
                  f"gpu_calls={n_gpu_calls}", flush=True)
        if not any_improve:
            plateau_passes += 1
            if stop_on_plateau_passes is not None and \
                    plateau_passes >= stop_on_plateau_passes:
                break
        else:
            plateau_passes = 0

    if verbose:
        print(f"  [gpu_cd] done {time.time()-t0:.1f}s "
              f"final={best_cost:.6f} accepts={n_accepts} "
              f"gpu_calls={n_gpu_calls} validates={n_total_validates}",
              flush=True)
    incr_eval.sync_positions(best_pos)
    gpu_eval.notify_full_resync()
    return best_pos, best_cost


def _check_overlap_static(pos_np, sep_x_row, sep_y_row, i, nx, ny, gap):
    ddx = np.abs(nx - pos_np[:, 0])
    ddy = np.abs(ny - pos_np[:, 1])
    o = (ddx < sep_x_row + gap) & (ddy < sep_y_row + gap)
    o[i] = False
    return bool(o.any())
