"""(1+1)-Evolution-Strategy on soft macro positions.

Maintain current best. Per iteration:
  - Mutate: perturb K softs with Gaussian noise σ
  - Evaluate proxy cost via full recompute (no incremental — mutations are batch)
  - Accept if improving; adapt σ (1/5 rule: if improve rate > 0.2, grow σ; else shrink)
"""
import time
import numpy as np


def one_plus_one_soft(pos_np, benchmark, incr_eval, max_time,
                      sigma_init=1.0, k_mutate=50, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = np.array([n_hard + i for i in range(n_total - n_hard)
                             if not macro_fixed[n_hard + i]])
    if len(soft_movable) == 0:
        return pos_np, incr_eval.get_proxy_cost()

    sigma = sigma_init
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_state = incr_eval.macro_pos.copy()

    rng = np.random.RandomState(9999)
    t0 = time.time()
    n_iter = 0
    n_accept = 0
    recent_accepts = []

    while time.time() - t0 < max_time:
        n_iter += 1
        # Snapshot
        saved = incr_eval.macro_pos.copy()

        # Mutate K random softs
        idx = rng.choice(len(soft_movable), min(k_mutate, len(soft_movable)), replace=False)
        for j in idx:
            m = int(soft_movable[j])
            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])
            nx = float(max(0, min(cw, ox + rng.randn() * sigma)))
            ny = float(max(0, min(ch, oy + rng.randn() * sigma)))
            incr_eval.macro_pos[m, 0] = np.float32(nx)
            incr_eval.macro_pos[m, 1] = np.float32(ny)

        # Full recompute (since we changed many)
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        new_cost = incr_eval.get_proxy_cost()

        accepted = new_cost < current_cost - 1e-7
        if accepted:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_state = incr_eval.macro_pos.copy()
            n_accept += 1
        else:
            # Revert
            incr_eval.macro_pos[:] = saved
            incr_eval._recompute_pin_positions()
            incr_eval._full_recompute_wl()
            incr_eval._full_recompute_density()
            incr_eval._full_recompute_congestion()

        recent_accepts.append(accepted)
        if len(recent_accepts) > 20:
            recent_accepts.pop(0)

        # 1/5-th rule
        if len(recent_accepts) >= 10:
            accept_rate = sum(recent_accepts) / len(recent_accepts)
            if accept_rate > 0.2:
                sigma *= 1.1
            elif accept_rate < 0.1:
                sigma *= 0.9

    # Restore best
    if best_cost < current_cost - 1e-7:
        incr_eval.macro_pos[:] = best_state
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()

    if verbose:
        print(f"    1+1_ES: {n_iter} iter, {n_accept} accept, sigma_end={sigma:.3f}, "
              f"cost {best_cost:.6f}")
    return pos_np, best_cost
