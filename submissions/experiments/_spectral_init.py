"""Spectral bisection warm-start using pure numpy (no scipy).

For N < 1000 macros, dense eigendecomposition is fast enough. Use
numpy.linalg.eigh on (D - W) — symmetric positive semi-definite.
Smallest eigenvalues via .eigh return sorted ascending.
"""
import numpy as np


def spectral_init(benchmark, plc, damping_to_init=0.3):
    """Return hard-macro init positions from 2nd/3rd Laplacian eigenvectors."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    init = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()

    if n_hard > 1500:
        # Too big for dense eig
        return init

    # Build name-to-bidx map
    if not hasattr(plc, '_name_to_hard_bidx'):
        plc._name_to_hard_bidx = {}
        for bidx, plc_idx in enumerate(plc.hard_macro_indices):
            plc._name_to_hard_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx

    # Dense adjacency matrix W
    W = np.zeros((n_hard, n_hard), dtype=np.float64)
    for driver_name, sinks in plc.nets.items():
        names = [driver_name.split("/")[0]] + [s.split("/")[0] for s in sinks]
        hard_indices = [plc._name_to_hard_bidx[nm] for nm in names
                        if nm in plc._name_to_hard_bidx]
        hard_indices = list(set(hard_indices))
        if len(hard_indices) < 2:
            continue
        try:
            w = plc.modules_w_pins[plc.mod_name_to_indices[driver_name]].get_weight()
        except Exception:
            w = 1.0
        k = len(hard_indices)
        edge_w = 2.0 * w / (k * (k - 1))
        for a in hard_indices:
            for b in hard_indices:
                if a != b:
                    W[a, b] += edge_w

    # Symmetrize
    W = (W + W.T) / 2

    # Laplacian
    degree = W.sum(axis=1)
    L = np.diag(degree) - W

    # Add tiny jitter to help conditioning
    L += 1e-8 * np.eye(n_hard)

    try:
        vals, vecs = np.linalg.eigh(L)
    except Exception:
        return init

    # vecs[:, 0] is trivial (zero eigenvalue for connected graph)
    # Use vecs[:, 1] and vecs[:, 2]
    fiedler = vecs[:, 1]
    f2 = vecs[:, 2]

    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)

    def _scale(v, lo, hi):
        v_lo, v_hi = v.min(), v.max()
        if v_hi - v_lo < 1e-9:
            return np.full_like(v, (lo + hi) / 2)
        return lo + (v - v_lo) / (v_hi - v_lo) * (hi - lo)

    x_new = _scale(fiedler, sizes[:, 0].max() / 2, cw - sizes[:, 0].max() / 2)
    y_new = _scale(f2, sizes[:, 1].max() / 2, ch - sizes[:, 1].max() / 2)

    result = init.copy()
    result[:, 0] = damping_to_init * init[:, 0] + (1 - damping_to_init) * x_new
    result[:, 1] = damping_to_init * init[:, 1] + (1 - damping_to_init) * y_new
    return result
