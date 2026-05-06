"""Fiedler-vector recursive bisection warm-start.

Hypothesis: every albania1 run shows v4-baseline variance ~0.005-0.010
across "same code, same seed" runs. This variance dominates the
algorithmic signal of every Hessian variant we've tested. The cause
is v4's stochastic SA — same seed, same wall budget, but different
rounding accumulations produce different post-v4 placements.

Recursive bisection fixes this by giving v4 a DETERMINISTIC, globally-
structured initial placement based on netlist TOPOLOGY (not on SA
randomness). The Fiedler vector of the netlist Laplacian gives the
optimal 2-way bipartition (Hall 1970, Pothen-Simon-Liou 1990). Sorting
macros by Fiedler value and median-splitting recursively gives a
hierarchical placement that respects netlist clusters.

Algorithm:
1. Build clique-Laplacian L of the netlist (macro-to-macro edges with
   weight w_n / (k_n - 1) per net).
2. Compute Fiedler vector v_2 via Lanczos (smallest non-zero eigval).
3. Sort macros by Fiedler value.
4. Median split: lower half → left canvas region, upper half → right.
5. Recurse within each region, alternating x/y axis at each level.
6. Stop when cluster has ≤ MIN_CLUSTER macros; place those at cluster
   centroid (uniformly distributed within their canvas region).

Output: (n_macros, 2) numpy array of initial positions. Replaces the
.plc init for v4.

Cost: O(N · log N) Lanczos + O(N log N) recursion = O(N log² N).
For ibm15 (N=1531): ~5s on CPU.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, eye
from scipy.sparse.linalg import eigsh


def build_clique_laplacian(pin_macro, net_starts, net_weight, n_macros):
    """Build sparse clique Laplacian over macros only (ports excluded).

    For each net of k macro-pins (port pins ignored here), add edges
    between every pair with weight w_n / (k - 1). Pure macro Laplacian.
    """
    rows, cols, data = [], [], []
    n_nets = int(net_starts.shape[0] - 1)
    for nid in range(n_nets):
        s, e = int(net_starts[nid]), int(net_starts[nid + 1])
        macros_in_net = [int(pin_macro[p]) for p in range(s, e)
                          if pin_macro[p] >= 0]
        # Dedup macros that appear multiple times in the same net.
        macros_in_net = list(set(macros_in_net))
        k = len(macros_in_net)
        if k < 2:
            continue
        w = float(net_weight[nid]) / (k - 1)
        for i, mi in enumerate(macros_in_net):
            for j_idx in range(i + 1, k):
                mj = macros_in_net[j_idx]
                rows += [mi, mj, mi, mj]
                cols += [mj, mi, mi, mj]
                data += [-w, -w, w, w]
    L = csr_matrix((data, (rows, cols)),
                    shape=(n_macros, n_macros), dtype=np.float64).tocsc()
    return L


def fiedler_vector(L, n_lanczos=200):
    """Compute the Fiedler eigenvector (smallest non-zero eigenvalue).

    For a connected graph, λ_1 = 0 with eigenvector all-ones. We use
    shift-invert mode with sigma=0 to find λ_2, the algebraic
    connectivity. Returns a (N,) numpy array of unit-norm eigenvector.
    """
    n = L.shape[0]
    try:
        # Add tiny Tikhonov to avoid singular shift-invert
        L_reg = L + 1e-9 * eye(n)
        eigvals, eigvecs = eigsh(L_reg, k=2, which="SA",
                                   sigma=0.0, mode="normal",
                                   maxiter=n_lanczos, tol=1e-4)
    except Exception:
        # Fallback: simple smallest-2 algebraic
        eigvals, eigvecs = eigsh(L + 1e-6 * eye(n),
                                   k=2, which="SA",
                                   maxiter=n_lanczos, tol=1e-4)
    # Sort by eigenvalue ascending; take the second one (λ_1 ≈ 0).
    order = np.argsort(eigvals)
    fiedler = eigvecs[:, order[1]]
    # Normalize (should already be unit-norm but double-check)
    n_v = np.linalg.norm(fiedler)
    if n_v > 1e-12:
        fiedler = fiedler / n_v
    return fiedler


def recursive_bisect_positions(macro_indices, fiedler_value, region,
                                  axis, depth, max_depth, min_cluster):
    """Recursively place macros within a canvas region.

    Args:
        macro_indices: list of macro indices in this cluster
        fiedler_value: (n_macros,) — full Fiedler vector
        region: (x_min, y_min, x_max, y_max) canvas region for this cluster
        axis: 0 (x-bisect) or 1 (y-bisect) for THIS level's split
        depth: current recursion depth
        max_depth: stop recursing
        min_cluster: stop when cluster size ≤ this

    Returns: dict {macro_idx: (x, y)} of positions
    """
    x_min, y_min, x_max, y_max = region
    n = len(macro_indices)
    out = {}

    if n == 0:
        return out

    if n <= min_cluster or depth >= max_depth:
        # Spread macros uniformly across the region.
        # For 1 macro, place at center. For more, spread on a small grid.
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        if n == 1:
            out[macro_indices[0]] = (cx, cy)
        else:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            dx = (x_max - x_min) / max(cols, 1)
            dy = (y_max - y_min) / max(rows, 1)
            for i, m in enumerate(macro_indices):
                r, c = i // cols, i % cols
                out[m] = (x_min + (c + 0.5) * dx, y_min + (r + 0.5) * dy)
        return out

    # Sort macros in this cluster by Fiedler value. Median split.
    sorted_macros = sorted(macro_indices, key=lambda m: fiedler_value[m])
    half = n // 2
    left_macros = sorted_macros[:half]
    right_macros = sorted_macros[half:]

    # Bisect canvas region along current axis.
    if axis == 0:
        mid = (x_min + x_max) / 2
        left_region = (x_min, y_min, mid, y_max)
        right_region = (mid, y_min, x_max, y_max)
    else:
        mid = (y_min + y_max) / 2
        left_region = (x_min, y_min, x_max, mid)
        right_region = (x_min, mid, x_max, y_max)

    out.update(recursive_bisect_positions(
        left_macros, fiedler_value, left_region, 1 - axis,
        depth + 1, max_depth, min_cluster))
    out.update(recursive_bisect_positions(
        right_macros, fiedler_value, right_region, 1 - axis,
        depth + 1, max_depth, min_cluster))
    return out


def compute_recursive_bisect_init(pin_macro, net_starts, net_weight,
                                     n_macros, n_hard, canvas_w, canvas_h,
                                     macro_w, macro_h,
                                     max_depth=12, min_cluster=8,
                                     verbose=False):
    """Top-level: compute recursive-bisection initial placement.

    Uses ALL macros (hard + soft) for Fiedler-vector partition, since
    they're connected via shared nets. Returns (n_macros, 2) array of
    positions in canvas units, clamped to canvas-interior (so each
    macro is fully inside the canvas).
    """
    import time
    t0 = time.time()

    L = build_clique_laplacian(pin_macro, net_starts, net_weight, n_macros)
    if verbose:
        print(f"  [bisect] Laplacian built: shape {L.shape}, nnz {L.nnz} "
              f"({time.time()-t0:.2f}s)", flush=True)

    fiedler = fiedler_vector(L, n_lanczos=200)
    if verbose:
        print(f"  [bisect] Fiedler vector ({time.time()-t0:.2f}s)", flush=True)

    # Recursive split. Initial region = full canvas, x-axis first.
    region = (0.0, 0.0, float(canvas_w), float(canvas_h))
    pos_dict = recursive_bisect_positions(
        list(range(n_macros)), fiedler, region,
        axis=0, depth=0, max_depth=max_depth, min_cluster=min_cluster)

    # Build positions array; clamp by macro half-extent.
    positions = np.zeros((n_macros, 2), dtype=np.float64)
    for m in range(n_macros):
        x, y = pos_dict.get(m, (canvas_w / 2, canvas_h / 2))
        half_w = float(macro_w[m]) / 2
        half_h = float(macro_h[m]) / 2
        x = float(np.clip(x, half_w, max(half_w, canvas_w - half_w)))
        y = float(np.clip(y, half_h, max(half_h, canvas_h - half_h)))
        positions[m] = (x, y)

    if verbose:
        print(f"  [bisect] {n_macros} positions computed "
              f"(total {time.time()-t0:.2f}s)", flush=True)
    return positions
