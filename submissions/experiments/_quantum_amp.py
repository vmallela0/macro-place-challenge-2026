"""Grover-like amplitude amplification for soft macros — classical sim.

Each soft macro m has a probability distribution p_m(j) over K candidate
positions. Start uniform. Each step:
  1. Sample position from each p_m
  2. Apply jointly; measure full proxy cost
  3. Update p_m(j): amplify if sampled j was in a good joint config

Approximates Grover-style amplification classically. Converges via
multiplicative weights (exponentiated gradient).
"""
import time
import numpy as np


def quantum_amp_soft(pos_np, benchmark, incr_eval, max_time,
                     n_candidates=16, n_iters=100, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    rng = np.random.RandomState(31337)
    t0 = time.time()

    # For each soft, define K candidate positions around current
    candidates = {}
    for m in soft_movable:
        ox = float(incr_eval.macro_pos[m, 0])
        oy = float(incr_eval.macro_pos[m, 1])
        cands = [(ox, oy)]  # include current
        for _ in range(n_candidates - 1):
            r = rng.choice([0.2, 0.5, 1.0, 2.0, 4.0])
            theta = rng.uniform(0, 2 * np.pi)
            nx = float(max(0, min(cw, ox + r * np.cos(theta))))
            ny = float(max(0, min(ch, oy + r * np.sin(theta))))
            cands.append((nx, ny))
        candidates[m] = cands

    # Initial weights: uniform over candidates
    weights = {m: np.ones(n_candidates) / n_candidates for m in soft_movable}

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_choice = {m: 0 for m in soft_movable}  # everyone at current

    lr = 0.5  # multiplicative-weight step
    n_evals = 0

    for it in range(n_iters):
        if time.time() - t0 > max_time:
            break

        # Sample one choice per macro
        choices = {}
        for m in soft_movable:
            choices[m] = int(rng.choice(n_candidates, p=weights[m]))

        # Apply
        saved = []
        for m in soft_movable:
            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])
            nx, ny = candidates[m][choices[m]]
            saved.append((m, ox, oy))
            incr_eval.macro_pos[m, 0] = np.float32(nx)
            incr_eval.macro_pos[m, 1] = np.float32(ny)
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        cost = incr_eval.get_proxy_cost()
        n_evals += 1

        # Update weights: if cost improved, amplify chosen; else dampen
        delta = cost - current_cost
        factor = np.exp(-lr * delta / max(1e-6, current_cost))  # relative improvement

        if cost < best_cost - 1e-7:
            best_cost = cost
            best_choice = dict(choices)
            # Keep the position
            current_cost = cost
        else:
            # Revert
            for m, ox, oy in saved:
                incr_eval.macro_pos[m, 0] = np.float32(ox)
                incr_eval.macro_pos[m, 1] = np.float32(oy)
            incr_eval._recompute_pin_positions()
            incr_eval._full_recompute_wl()
            incr_eval._full_recompute_density()
            incr_eval._full_recompute_congestion()

        # Apply factor to each chosen candidate's weight
        for m in soft_movable:
            weights[m][choices[m]] *= factor
            weights[m] = weights[m] / weights[m].sum()  # renormalize

    # Final: set each macro to argmax of its distribution
    final_state = []
    for m in soft_movable:
        best_j = int(np.argmax(weights[m]))
        nx, ny = candidates[m][best_j]
        final_state.append((m, nx, ny))
        incr_eval.macro_pos[m, 0] = np.float32(nx)
        incr_eval.macro_pos[m, 1] = np.float32(ny)
    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()
    final_cost = incr_eval.get_proxy_cost()
    # If final worse than best seen, apply best_choice
    if final_cost > best_cost + 1e-6:
        for m in soft_movable:
            nx, ny = candidates[m][best_choice[m]]
            incr_eval.macro_pos[m, 0] = np.float32(nx)
            incr_eval.macro_pos[m, 1] = np.float32(ny)
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        final_cost = best_cost

    if verbose:
        print(f"    quantum_amp: {n_evals} evals, best {best_cost:.6f}, final {final_cost:.6f}")
    return pos_np, min(best_cost, final_cost)
