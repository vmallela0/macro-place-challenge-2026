"""Kahng-Kennings quadratic placement: solve L x = b for each axis.

Clique model: each net with pins {p_1..p_k} contributes w/(k-1) between every
pair. Fixed pins (ports, soft macros) stay on RHS. Only hard macros are solved.
"""
import numpy as np


def _build_lb(ie, axis):
    """Build dense Laplacian L (n_hard x n_hard) and RHS b (n_hard,).

    axis = 0 (x) or 1 (y).
    """
    n = ie.n_hard
    L = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    # pin info
    pin_macro = ie.pin_macro                    # int32[n_pins]; >=0 if macro, -1 if fixed
    pin_off = ie.pin_xoff if axis == 0 else ie.pin_yoff  # pin x/y offset from macro center
    # for fixed pins, we still stored their absolute coords in pin_xoff/yoff with macro=-1
    # macro_pos for current pos of macros (hard and soft)
    mpos = ie.macro_pos[:, axis].astype(np.float64)
    n_hard = ie.n_hard

    for nid in range(ie.n_nets):
        start = ie.net_starts[nid]
        end = ie.net_starts[nid + 1]
        k = end - start
        if k < 2:
            continue
        w = float(ie.net_weight[nid]) / (k - 1)

        # Build per-pin info: (var, alpha) where y_p = z_var + alpha if var>=0 else constant alpha
        var_arr = np.empty(k, dtype=np.int64)
        alpha_arr = np.empty(k, dtype=np.float64)
        for t in range(k):
            p = start + t
            m = pin_macro[p]
            if 0 <= m < n_hard:
                # hard macro: variable = m, alpha = pin offset
                var_arr[t] = m
                alpha_arr[t] = pin_off[p]
            elif m >= n_hard:
                # soft macro: treated as fixed at macro_pos + pin_offset
                var_arr[t] = -1
                alpha_arr[t] = mpos[m] + pin_off[p]
            else:
                # port: fixed abs position stored in pin_off
                var_arr[t] = -1
                alpha_arr[t] = pin_off[p]

        # iterate all pairs
        for i in range(k):
            vi = var_arr[i]; ai = alpha_arr[i]
            for j in range(i + 1, k):
                vj = var_arr[j]; aj = alpha_arr[j]
                if vi >= 0 and vj >= 0:
                    if vi == vj:
                        continue  # same macro, same variable; pair adds constant (aj-ai)^2
                    L[vi, vi] += w
                    L[vj, vj] += w
                    L[vi, vj] -= w
                    L[vj, vi] -= w
                    d = ai - aj
                    b[vi] -= w * d
                    b[vj] += w * d
                elif vi >= 0 and vj < 0:
                    L[vi, vi] += w
                    b[vi] += w * (aj - ai)
                elif vj >= 0 and vi < 0:
                    L[vj, vj] += w
                    b[vj] += w * (ai - aj)
                # both fixed: constant, skip
    return L, b


def qp_init(init_pos, benchmark, plc):
    """Quadratic-CG init. Returns (n_hard, 2) float64 positions clipped to canvas.

    init_pos: (n_hard, 2) float64 initial positions (used for regularization fallback)
    benchmark: Benchmark object
    plc: PlacementCost object

    Uses IncrementalEvaluator to collect net data, builds dense Laplacian per axis,
    solves via numpy.linalg.solve. Adds small Tikhonov regularization toward init_pos
    so that fully-isolated macros (no nets touching them) don't blow up.
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "_v_placer", str(Path(__file__).resolve().parents[1] / "vmallela" / "placer.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    IncrementalEvaluator = mod.IncrementalEvaluator
    ie = IncrementalEvaluator(plc, benchmark)

    n_hard = ie.n_hard
    cw, ch = ie.cw, ie.ch
    sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)

    # Regularization: pull macros softly toward init_pos (strength = median(diag(L)) * 1e-4)
    out = np.zeros((n_hard, 2), dtype=np.float64)
    for axis in range(2):
        L, b = _build_lb(ie, axis)
        diag = np.diag(L).copy()
        pos_diag = diag[diag > 0]
        lam = (np.median(pos_diag) if pos_diag.size > 0 else 1.0) * 1e-4
        # Also handle rows with zero diagonal (isolated macros): set to lam
        for i in range(n_hard):
            if L[i, i] <= 0.0:
                L[i, i] = lam
                b[i] = lam * init_pos[i, axis]
            else:
                L[i, i] += lam
                b[i] += lam * init_pos[i, axis]
        try:
            x = np.linalg.solve(L, b)
        except np.linalg.LinAlgError:
            # fallback lstsq
            x, *_ = np.linalg.lstsq(L, b, rcond=None)
        out[:, axis] = x

    # Clip to canvas
    half_w = sizes_np[:, 0] * 0.5
    half_h = sizes_np[:, 1] * 0.5
    out[:, 0] = np.clip(out[:, 0], half_w, cw - half_w)
    out[:, 1] = np.clip(out[:, 1], half_h, ch - half_h)
    return out
