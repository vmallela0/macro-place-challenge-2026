"""Surrogate-guided soft CD v2 — shared model across cycles.

v1 issue: each soft_macro_cd call was standalone; warmup+train happened
every call, so main phase had <10s. Surrogate couldn't pay off.

v2 fix: share the logger and model across calls. First call does warmup
on fixed budget; subsequent calls use the trained model directly.
Model is periodically retrained with accumulated data.
"""
import time
import numpy as np
from _surrogate import (ProbeLogger, train_surrogate, surrogate_rank_candidates,
                        _HAS_TORCH)


class SurrogateSoftCD:
    """Stateful surrogate soft CD — keep training data + model across calls."""

    def __init__(self):
        self.logger = ProbeLogger(max_samples=30000)
        self.model = None
        self.n_samples_at_last_train = 0
        self.retrain_interval = 3000  # retrain every N new samples

    def run(self, pos_np, benchmark, incr_eval, max_time,
            cand_per_macro=20, verify_top_k=2, verbose=False):
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

        # Train/retrain if needed
        if _HAS_TORCH and self.logger.n >= 1000 and (
                self.model is None or
                self.logger.n - self.n_samples_at_last_train > self.retrain_interval):
            t_train = time.time()
            self.model = train_surrogate(self.logger, epochs=20, batch_size=512)
            self.n_samples_at_last_train = self.logger.n
            if verbose:
                print(f"    surrogate_v2: trained on {self.logger.n}, took {time.time() - t_train:.1f}s")

        warmup_time = 3 if self.model is not None else 8
        dirs_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]
        deltas = [1.0, 0.5, 0.25]

        # Brief warmup to update logger with current cost surface
        t_warmup_end = t0 + min(warmup_time, max_time * 0.3)
        for delta in deltas:
            if time.time() > t_warmup_end:
                break
            for i in soft_sorted:
                if time.time() > t_warmup_end:
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
                    feats = self.logger.build_features(incr_eval, i, ox, oy, nx, ny)
                    c = incr_eval.move_macro(i, nx, ny)
                    self.logger.add(feats, c - current_cost)
                    if c < best_dir_c:
                        best_dir_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c

        # Main surrogate-guided phase
        n_main = 0
        while time.time() - t0 < max_time:
            improved = False
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])

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
                    cand_feats.append(self.logger.build_features(incr_eval, i, ox, oy, nx, ny))

                if not cand_positions:
                    continue

                if self.model is not None:
                    preds, order = surrogate_rank_candidates(self.model, np.array(cand_feats))
                    top_indices = list(order[:verify_top_k])
                else:
                    top_indices = list(range(min(verify_top_k, len(cand_positions))))

                best_c = current_cost
                best_nx, best_ny = None, None
                for idx in top_indices:
                    nx, ny = cand_positions[idx]
                    c = incr_eval.move_macro(i, nx, ny)
                    self.logger.add(cand_feats[idx], c - current_cost)
                    if c < best_c:
                        best_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()

                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_c
                    if best_c < best_cost:
                        best_cost = best_c
                    n_main += 1
                    improved = True
            if not improved:
                break

        if verbose:
            print(f"    surrogate_v2: {n_main} moves (main), cost {best_cost:.6f} "
                  f"(model={'yes' if self.model else 'NO'})")
        return pos_np, best_cost


# Module-level singleton (shared across cycles within one placer.place() call)
_global_state = SurrogateSoftCD()


def soft_cd_surrogate_v2(pos_np, benchmark, incr_eval, max_time, verbose=False):
    return _global_state.run(pos_np, benchmark, incr_eval, max_time, verbose=verbose)


def reset_surrogate_state():
    global _global_state
    _global_state = SurrogateSoftCD()
