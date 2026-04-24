"""Alternate initial-placement generators for the legalize tournament.

Each function returns an (n_hard, 2) float64 array of hard-macro positions
on the canvas. Intended to be added as extra seeds to the push-apart /
legalize tournament in OptimalPlacer.place().
"""
import numpy as np


def random_init(init_pos, benchmark, seed=42):
    """IP1: uniform random on canvas, respecting macro half-sizes."""
    rng = np.random.RandomState(seed)
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    out = np.zeros_like(init_pos)
    out[:, 0] = rng.uniform(sizes[:, 0] / 2, cw - sizes[:, 0] / 2)
    out[:, 1] = rng.uniform(sizes[:, 1] / 2, ch - sizes[:, 1] / 2)
    return out


def jacobi_init(init_pos, benchmark, n_iter=100, damping=0.5):
    """IP3: Jacobi centroid iteration.
    x_i <- damping * x_i + (1 - damping) * weighted mean of connected macros.
    """
    n_hard = benchmark.num_hard_macros
    pos = init_pos.copy()

    # Build macro→connected-macros + weights, using net hyperedges via adj_list if present
    # Fall back: use benchmark.net_indices / net_weights if exposed.
    # For robustness here, rely on benchmark.net_pins if available, else skip.
    if not hasattr(benchmark, 'net_indices'):
        return pos  # can't do it without net info; return init

    # Build adj in (macro_idx, weight) lists
    adj = [[] for _ in range(n_hard)]
    net_indices = benchmark.net_indices  # list of lists of (macro_idx, pin_offset_x, pin_offset_y)
    net_weights = getattr(benchmark, 'net_weights', None)
    for nid, pins in enumerate(net_indices):
        w = float(net_weights[nid]) if net_weights is not None else 1.0
        macros_on_net = [p[0] for p in pins if p[0] < n_hard]
        if len(macros_on_net) < 2:
            continue
        # Clique approximation
        for a in macros_on_net:
            for b in macros_on_net:
                if a != b:
                    adj[a].append((b, w / (len(macros_on_net) - 1)))

    for it in range(n_iter):
        new = pos.copy()
        for i in range(n_hard):
            if not adj[i]:
                continue
            wsum = 0.0
            x = 0.0; y = 0.0
            for (j, w) in adj[i]:
                x += w * pos[j, 0]
                y += w * pos[j, 1]
                wsum += w
            if wsum > 0:
                new[i, 0] = damping * pos[i, 0] + (1 - damping) * x / wsum
                new[i, 1] = damping * pos[i, 1] + (1 - damping) * y / wsum
        max_move = float(np.max(np.abs(new - pos)))
        pos = new
        if max_move < 1e-4:
            break
    return pos


def center_init(init_pos, benchmark, jitter=0.5):
    """IP7: collapse all macros to canvas center + small jitter."""
    rng = np.random.RandomState(42)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    out = init_pos.copy()
    out[:, 0] = cw / 2 + rng.uniform(-jitter, jitter, size=out.shape[0])
    out[:, 1] = ch / 2 + rng.uniform(-jitter, jitter, size=out.shape[0])
    return out


def grid_init(init_pos, benchmark):
    """IP6: regular grid, largest macros first."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    order = np.argsort(-(sizes[:, 0] * sizes[:, 1]))
    ncol = int(np.ceil(np.sqrt(n_hard)))
    nrow = int(np.ceil(n_hard / ncol))
    dx = cw / ncol
    dy = ch / nrow
    out = init_pos.copy()
    for rank, i in enumerate(order):
        r = rank // ncol
        c = rank % ncol
        out[i, 0] = float(np.clip(dx * (c + 0.5), sizes[i, 0] / 2, cw - sizes[i, 0] / 2))
        out[i, 1] = float(np.clip(dy * (r + 0.5), sizes[i, 1] / 2, ch - sizes[i, 1] / 2))
    return out
