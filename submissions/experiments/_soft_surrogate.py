"""Surrogate-guided soft CD (idea #1 — MLP probe ranking).

Two-phase loop:
  A. Warm-up (first ~15s): collect real probe results. Train MLP once.
  B. Main: rank 16 candidates per macro with MLP (~1ms); verify top 2 with
     real eval (~100ms). 8× more candidates per macro vs 8-direction
     baseline, at similar wall-clock.
"""
import time
import numpy as np
from _surrogate import (ProbeLogger, train_surrogate, surrogate_rank_candidates,
                        _HAS_TORCH)


def soft_cd_surrogate(pos_np, benchmark, incr_eval, max_time,
                      warmup_frac=0.25, cand_per_macro=20, verify_top_k=2,
                      verbose=False):
    """Surrogate-guided soft CD."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    rng = np.random.RandomState(1234)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost

    t0 = time.time()
    logger = ProbeLogger(max_samples=20000)

    # --- Phase A: warm-up with real 8-direction CD + logging ---
    dirs_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
    deltas_warmup = [1.0, 0.5, 0.25]
    warmup_deadline = t0 + max_time * warmup_frac

    n_warm_moves = 0
    for delta in deltas_warmup:
        if time.time() > warmup_deadline:
            break
        for i in soft_sorted:
            if time.time() > warmup_deadline:
                break
            ox = float(incr_eval.macro_pos[i, 0])
            oy = float(incr_eval.macro_pos[i, 1])
            best_dir_c = current_cost
            best_nx, best_ny = None, None
            for dx, dy in dirs_8:
                nx = float(max(0, min(cw, ox + delta * dx)))
                ny = float(max(0, min(ch, oy + delta * dy)))
                if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                    continue
                feats = logger.build_features(incr_eval, i, ox, oy, nx, ny)
                c = incr_eval.move_macro(i, nx, ny)
                delta_cost = c - current_cost
                logger.add(feats, delta_cost)
                if c < best_dir_c:
                    best_dir_c = c
                    best_nx, best_ny = nx, ny
                incr_eval.undo_move()
            if best_nx is not None:
                incr_eval.move_macro(i, best_nx, best_ny)
                current_cost = best_dir_c
                if best_dir_c < best_cost:
                    best_cost = best_dir_c
                n_warm_moves += 1

    # --- Train MLP ---
    model = None
    if _HAS_TORCH and logger.is_ready(min_samples=1000):
        t_train = time.time()
        model = train_surrogate(logger, epochs=25, batch_size=512)
        train_time = time.time() - t_train
        if verbose:
            print(f"    surrogate: trained on {logger.n} samples in {train_time:.1f}s")

    if verbose:
        print(f"    warmup: {n_warm_moves} moves, {logger.n} probes logged, "
              f"cost {current_cost:.6f}")

    # --- Phase B: surrogate-ranked exploration ---
    n_main_moves = 0
    while time.time() - t0 < max_time:
        improved = False
        for i in soft_sorted:
            if time.time() - t0 > max_time:
                break
            ox = float(incr_eval.macro_pos[i, 0])
            oy = float(incr_eval.macro_pos[i, 1])

            # Generate candidate moves: random offsets in multiple radii
            cand_positions = []
            cand_feats = []
            for _ in range(cand_per_macro):
                r = rng.choice([0.12, 0.25, 0.5, 1.0, 2.0])
                theta = rng.uniform(0, 2 * np.pi)
                nx = float(max(0, min(cw, ox + r * np.cos(theta))))
                ny = float(max(0, min(ch, oy + r * np.sin(theta))))
                if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                    continue
                cand_positions.append((nx, ny))
                cand_feats.append(logger.build_features(incr_eval, i, ox, oy, nx, ny))

            if not cand_positions:
                continue

            if model is not None:
                preds, order = surrogate_rank_candidates(model, np.array(cand_feats))
                top_indices = order[:verify_top_k]
            else:
                top_indices = list(range(min(verify_top_k, len(cand_positions))))

            best_c = current_cost
            best_nx, best_ny = None, None
            for idx in top_indices:
                nx, ny = cand_positions[idx]
                feats = cand_feats[idx]
                c = incr_eval.move_macro(i, nx, ny)
                delta_cost = c - current_cost
                logger.add(feats, delta_cost)  # keep growing training set
                if c < best_c:
                    best_c = c
                    best_nx, best_ny = nx, ny
                incr_eval.undo_move()

            if best_nx is not None:
                incr_eval.move_macro(i, best_nx, best_ny)
                current_cost = best_c
                if best_c < best_cost:
                    best_cost = best_c
                n_main_moves += 1
                improved = True

        if not improved:
            break

    if verbose:
        print(f"    main: {n_main_moves} moves, cost {best_cost:.6f} "
              f"(surrogate={'yes' if model else 'NO'})")

    return pos_np, best_cost
