"""Analytical warm-start: LSE-HPWL minimization with soft pairwise repulsion.

Goal: produce a HARD-MACRO position vector that minimizes total weighted
HPWL globally (informed by every net at once) while being approximately
overlap-free, to feed into the legalize tournament as an additional
starting candidate.

Method: gradient descent on
    f(p) = sum_nets w_n * [LSE_x(eta, n) + LSE_y(eta, n)]
         + alpha * sum_{i<j} max(0, sep_ij - dx_ij)^2 * 1[dy_ij<sep_y_ij]
                + max(0, sep_ij - dy_ij)^2 * 1[dx_ij<sep_x_ij]
where
    LSE_x(eta, n) = (1/eta) * [logsumexp(eta * x_pin_n) + logsumexp(-eta * x_pin_n)]
                  -> exact HPWL_x as eta -> infinity
    sep_ij = (size_i + size_j) / 2 + gap

Eta is annealed up (so HPWL approximation tightens over iterations);
step size is annealed down. Repulsion alpha grows so macros spread.

The result is fed into _legalize as a new starting candidate; the
tournament logic in placer.py picks the best of {pushed_positions,
init_pos, warmstart_pos}.

Env-gated: PLACER_WARMSTART=1 enables (default 0 = v4 behavior).
"""
import os
import time
import numpy as np


def analytical_warmstart(benchmark, plc, init_pos, max_time=30.0,
                         eta_start=0.3, eta_end=10.0, eta_steps=12,
                         steps_per_eta=8, lr_start=2.0, lr_end=0.05,
                         alpha_start=0.5, alpha_end=20.0,
                         lambda_init=0.05, verbose=False):
    """Run LSE-HPWL + repulsion warm-start.

    Args:
        benchmark: macro_place.benchmark.Benchmark.
        plc: TILOS plc object (from _load_plc).
        init_pos: float64 (n_hard, 2) initial positions.
        max_time: wall-clock cap (s).
        eta_start/end: LSE smoothing (smaller = smoother / less HPWL-faithful).
        eta_steps: number of annealing levels.
        steps_per_eta: gradient steps per eta level.
        lr_start/end: step size anneal range (multiplicative on grad).
        alpha_start/end: repulsion strength anneal.
        lambda_init: weight on stay-close-to-init regularizer.

    Returns:
        np.ndarray (n_hard, 2) float64. May still have small overlaps —
        downstream `_legalize` will fix.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy().astype(bool)
    movable_idx = np.where(movable)[0]
    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2  # (n,n)
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    gap = 0.05

    # Build pin index: pin -> macro_idx (-1 = port), pin offsets, net→pin CSR.
    # Reuse incr_eval's loader logic by iterating plc directly.
    plc_to_hard = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        plc_to_hard[plc.modules_w_pins[plc_idx].get_name()] = bidx

    pin_offsets = {}
    for plc_idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == 'MACRO_PIN':
            parent = mod.get_name().split('/')[0]
            if parent in plc_to_hard:
                xo, yo = mod.get_offset()
                pin_offsets[mod.get_name()] = (plc_to_hard[parent], xo, yo)

    port_positions = {}
    for plc_idx in plc.port_indices:
        mod = plc.modules_w_pins[plc_idx]
        port_positions[mod.get_name()] = mod.get_pos()

    # Flatten pins per net. We only consider hard-macro pins + ports here
    # (soft macros not in init_pos; they'll be set by the soft phase later).
    pin_macro = []
    pin_xoff = []  # macro pins: offset; ports: absolute x
    pin_yoff = []
    net_starts = [0]
    net_weights = []
    for driver_name, sinks in plc.nets.items():
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = float(plc.modules_w_pins[driver_plc_idx].get_weight())
        all_pins = [driver_name] + sinks
        net_pin_count = 0
        for pin_name in all_pins:
            parent = pin_name.split('/')[0]
            if pin_name in pin_offsets:
                bidx, xo, yo = pin_offsets[pin_name]
                pin_macro.append(bidx)
                pin_xoff.append(float(xo))
                pin_yoff.append(float(yo))
                net_pin_count += 1
            elif parent in plc_to_hard:
                pin_macro.append(plc_to_hard[parent])
                pin_xoff.append(0.0)
                pin_yoff.append(0.0)
                net_pin_count += 1
            elif parent in port_positions:
                px, py = port_positions[parent]
                pin_macro.append(-1)
                pin_xoff.append(float(px))
                pin_yoff.append(float(py))
                net_pin_count += 1
        if net_pin_count >= 2:
            net_starts.append(len(pin_macro))
            net_weights.append(weight)
        else:
            # roll back
            for _ in range(net_pin_count):
                pin_macro.pop()
                pin_xoff.pop()
                pin_yoff.pop()

    pin_macro = np.array(pin_macro, dtype=np.int32)
    pin_xoff = np.array(pin_xoff, dtype=np.float64)
    pin_yoff = np.array(pin_yoff, dtype=np.float64)
    net_starts = np.array(net_starts, dtype=np.int32)
    net_weights = np.array(net_weights, dtype=np.float64)
    n_nets = len(net_weights)
    n_pins = len(pin_macro)
    if verbose:
        print(f"  [warmstart] n_nets={n_nets} n_pins={n_pins}")

    # Initial positions. Movable get init_pos; non-movable stay where init_pos
    # has them. We optimize ALL hards together but only update movable gradient.
    pos = init_pos.copy().astype(np.float64)
    init_pos_save = init_pos.copy().astype(np.float64)

    # Per-pin macro mask (True if macro pin, False if port).
    is_macro_pin = pin_macro >= 0
    macro_pin_idx = np.where(is_macro_pin)[0]
    port_pin_idx = np.where(~is_macro_pin)[0]

    # Eta and lr schedules (geometric).
    etas = np.geomspace(eta_start, eta_end, eta_steps)
    alphas = np.geomspace(alpha_start, alpha_end, eta_steps)
    lrs = np.geomspace(lr_start, lr_end, eta_steps)

    t0 = time.time()
    eta_diag = float(min(cw, ch))  # used for numerical stability scaling

    for ei, (eta, alpha, lr) in enumerate(zip(etas, alphas, lrs)):
        if time.time() - t0 > max_time:
            break

        for inner in range(steps_per_eta):
            if time.time() - t0 > max_time:
                break

            # Refresh pin positions from current macro positions.
            pin_x = np.empty(n_pins, dtype=np.float64)
            pin_y = np.empty(n_pins, dtype=np.float64)
            pin_x[macro_pin_idx] = pos[pin_macro[macro_pin_idx], 0] + pin_xoff[macro_pin_idx]
            pin_y[macro_pin_idx] = pos[pin_macro[macro_pin_idx], 1] + pin_yoff[macro_pin_idx]
            pin_x[port_pin_idx] = pin_xoff[port_pin_idx]
            pin_y[port_pin_idx] = pin_yoff[port_pin_idx]

            # --- HPWL gradient (LSE-smoothed) ---
            # For each net, compute softmax(eta * x) and softmax(-eta * x);
            # contribution to pin grad = w_n * (sm_pos - sm_neg) (in x; same for y).
            # Then scatter-add per-pin grad back to macros via pin_macro.
            grad = np.zeros((n_hard, 2), dtype=np.float64)
            for nid in range(n_nets):
                s = int(net_starts[nid])
                e = int(net_starts[nid + 1])
                w = net_weights[nid]
                xs = pin_x[s:e]
                ys = pin_y[s:e]

                # Stable softmax(eta*xs). Subtract max for numerical safety.
                ex_max = xs.max()
                en_max = (-xs).max()
                e_xs = np.exp(eta * (xs - ex_max))
                e_negxs = np.exp(eta * (-xs - en_max))
                sm_x = e_xs / e_xs.sum()
                sm_negx = e_negxs / e_negxs.sum()
                grad_x_pins = w * (sm_x - sm_negx)

                ey_max = ys.max()
                eyn_max = (-ys).max()
                e_ys = np.exp(eta * (ys - ey_max))
                e_negys = np.exp(eta * (-ys - eyn_max))
                sm_y = e_ys / e_ys.sum()
                sm_negy = e_negys / e_negys.sum()
                grad_y_pins = w * (sm_y - sm_negy)

                # Scatter to macros (only macro pins; ports are fixed).
                for k in range(s, e):
                    m = int(pin_macro[k])
                    if m < 0:
                        continue
                    grad[m, 0] += grad_x_pins[k - s]
                    grad[m, 1] += grad_y_pins[k - s]

            # --- Repulsion gradient (vectorized pair) ---
            # For each pair (i, j) movable, compute overlap; if overlapping,
            # add a force pushing them apart in the direction of LARGER overlap.
            # Implemented O(n^2) — fine for n up to ~600.
            dx = pos[:, 0:1] - pos[:, 0:1].T   # (n,n) signed
            dy = pos[:, 1:2] - pos[:, 1:2].T
            adx = np.abs(dx)
            ady = np.abs(dy)
            ovx = sep_x_mat + gap - adx   # >0 means overlap in x
            ovy = sep_y_mat + gap - ady
            overlap = (ovx > 0.0) & (ovy > 0.0)
            np.fill_diagonal(overlap, False)
            # Quadratic penalty grows as overlap deepens.
            # Push along the dim with smaller overlap (cheaper resolution).
            push_x = np.where((ovx > 0.0) & (ovx <= ovy), 2.0 * ovx * np.sign(dx), 0.0)
            push_y = np.where((ovy > 0.0) & (ovy < ovx),  2.0 * ovy * np.sign(dy), 0.0)
            push_x = np.where(overlap, push_x, 0.0)
            push_y = np.where(overlap, push_y, 0.0)
            # Sum row-wise: row i = total push received by macro i.
            grad[:, 0] -= alpha * push_x.sum(axis=1)
            grad[:, 1] -= alpha * push_y.sum(axis=1)

            # --- Stay-close regularizer (small, prevents drift) ---
            grad[:, 0] += 2.0 * lambda_init * (pos[:, 0] - init_pos_save[:, 0])
            grad[:, 1] += 2.0 * lambda_init * (pos[:, 1] - init_pos_save[:, 1])

            # Zero gradient for fixed (non-movable) macros.
            grad[~movable] = 0.0

            # Step. Normalize so step size is bounded in microns.
            gnorm = np.linalg.norm(grad, axis=1).max()
            if gnorm < 1e-9:
                break
            scale = lr / max(gnorm, 1.0)
            pos -= scale * grad

            # Project onto canvas.
            pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

        if verbose:
            elapsed = time.time() - t0
            print(f"    [warmstart eta={eta:.2f} lr={lr:.2f} alpha={alpha:.2f}] "
                  f"t={elapsed:.1f}s gnorm={gnorm:.4f}")

    if verbose:
        print(f"  [warmstart] DONE t={time.time() - t0:.1f}s")
    return pos
