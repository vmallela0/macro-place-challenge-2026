"""Per-net optimization — iterate nets by weight, optimize pin positions in each.

For each net, find the HPWL-minimizing positions for its MOVABLE pins
(softs and hards). Subject to canvas bounds and overlap (for hards).
Minimize the net's bounding box.

This is a SINGLE-NET local optimizer. Not "correct" for proxy cost
(doesn't consider density/congestion), but can shrink HPWL dramatically.

Greedy: for each net N, for each movable pin p in N, try moving p to
shrink N's bbox (without growing other nets' bboxes too much).
"""
import time
import numpy as np


def per_net_optimize(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """Per-net HPWL optimizer.

    BUG FIX (exp 0): the caller in placer.py passed `best_pos` (hard positions)
    and read back (_, c) — discarding any updated hard positions. But this
    function moves HARD macros in incr_eval during its line search. The next
    hard-operator (hard_lns, _coord_descent) then called
    `incr_eval.sync_positions(best_pos)` which RESET incr_eval's hard state
    back to pre-per_net, wiping out every hard-macro improvement per_net had
    just found.

    Fix: return `best_pos_hard` (a freshly-copied incr_eval.macro_pos[:n_hard]
    snapshot at the moment best_cost was last beaten) so the caller can
    actually propagate the hard positions.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    gap = 0.05

    # Sort nets by weight, descending
    net_weights = incr_eval.net_weight
    net_order = np.argsort(-net_weights)

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    # Snapshot hard positions at entry — this is what we'd return if no improvement.
    best_pos_hard = incr_eval.macro_pos[:n_hard].copy().astype(np.float64)
    t0 = time.time()
    n_moves = 0

    for nid in net_order:
        if time.time() - t0 > max_time:
            break
        # Find movable macros in this net
        macros = [m for m in incr_eval.net_macros[nid]
                  if 0 <= m < n_total and not macro_fixed[m]]
        if not macros:
            continue

        # For each movable macro in this net, try to move toward the
        # weighted centroid of the net's other pins.
        for m in macros:
            if time.time() - t0 > max_time:
                break
            # Collect other-pin positions in this net
            start = int(incr_eval.net_starts[nid])
            end = int(incr_eval.net_starts[nid + 1])
            xs, ys = [], []
            for pidx in range(start, end):
                pm = int(incr_eval.pin_macro[pidx])
                if pm == m:
                    continue
                if pm < 0:
                    xs.append(float(incr_eval.pin_xoff[pidx]))
                    ys.append(float(incr_eval.pin_yoff[pidx]))
                else:
                    xs.append(float(incr_eval.macro_pos[pm, 0]))
                    ys.append(float(incr_eval.macro_pos[pm, 1]))
            if not xs:
                continue

            # Target: median of other x/y (HPWL-optimal for this net)
            tx = float(np.median(xs))
            ty = float(np.median(ys))

            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])

            # exp 7: line-search env-gated.
            _ls = __import__('os').environ.get('PLACER_EXP7_LINESEARCH', '0') == '1'
            _alphas = [1.0, 0.5, 0.25, 0.125, 0.0625] if _ls else [1.0, 0.5, 0.25, 0.1]
            _ls_best_c, _ls_best_x, _ls_best_y = None, None, None
            for alpha in _alphas:
                nx = ox + alpha * (tx - ox)
                ny = oy + alpha * (ty - oy)
                if m < n_hard:
                    nx = float(np.clip(nx, half_w[m], cw - half_w[m]))
                    ny = float(np.clip(ny, half_h[m], ch - half_h[m]))
                    # Check overlap (hard only)
                    ddx = np.abs(nx - incr_eval.macro_pos[:n_hard, 0])
                    ddy = np.abs(ny - incr_eval.macro_pos[:n_hard, 1])
                    ov = (ddx < sep_x[m] + gap) & (ddy < sep_y[m] + gap)
                    ov[m] = False
                    if ov.any():
                        continue
                else:
                    nx = float(max(0, min(cw, nx)))
                    ny = float(max(0, min(ch, ny)))

                if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                    continue

                c = incr_eval.move_macro(m, nx, ny)
                if c < current_cost - 1e-7:
                    current_cost = c
                    if c < best_cost:
                        best_cost = c
                        # Snapshot hard positions at the new best cost so the
                        # caller can propagate the hard-macro moves (exp 0 fix).
                        best_pos_hard = incr_eval.macro_pos[:n_hard].copy().astype(np.float64)
                    n_moves += 1
                    break  # take first improving alpha
                incr_eval.undo_move()

    if verbose:
        print(f"    per_net: {n_moves} moves, cost {best_cost:.6f}")
    return best_pos_hard, best_cost
