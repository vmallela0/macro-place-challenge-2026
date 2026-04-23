"""Periodic soft-macro perturbation restart.

After N cycles of soft optimization plateau, shake N% of softs with
large-variance Gaussian noise, then re-run soft CD. If result improves,
keep; else revert.
"""
import time
import numpy as np


def soft_perturb_and_reopt(pos_np, benchmark, incr_eval,
                           soft_cd_fn, perturb_frac=0.2,
                           perturb_sigma=2.0, reopt_time=15,
                           verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    saved_state = incr_eval.macro_pos.copy()
    cost_before = incr_eval.get_proxy_cost()

    # Perturb
    rng = np.random.RandomState(int(time.time() * 1000) % 100000)
    n_perturb = max(1, int(perturb_frac * len(soft_movable)))
    chosen = rng.choice(len(soft_movable), n_perturb, replace=False)
    for idx in chosen:
        m = soft_movable[idx]
        ox = float(incr_eval.macro_pos[m, 0])
        oy = float(incr_eval.macro_pos[m, 1])
        nx = float(max(0, min(cw, ox + rng.randn() * perturb_sigma)))
        ny = float(max(0, min(ch, oy + rng.randn() * perturb_sigma)))
        incr_eval.macro_pos[m, 0] = np.float32(nx)
        incr_eval.macro_pos[m, 1] = np.float32(ny)

    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()

    # Re-optimize softs
    _, new_cost = soft_cd_fn(pos_np, benchmark, incr_eval, reopt_time)

    if new_cost < cost_before - 1e-6:
        if verbose:
            print(f"    perturb_reopt: IMPROVED {cost_before:.6f} -> {new_cost:.6f}")
        return pos_np, new_cost
    else:
        # Restore
        incr_eval.macro_pos[:] = saved_state
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        if verbose:
            print(f"    perturb_reopt: no gain ({new_cost:.6f} vs {cost_before:.6f})")
        return pos_np, cost_before
