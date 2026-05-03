"""Hyperbolic-embedding warm-start for chip placement.

THE INSIGHT
-----------
Chip netlists are inherently hierarchical: a CPU contains caches contain
cells contain transistors. Each level fans out exponentially.

Hyperbolic geometry is the *intrinsic* metric for hierarchy. In hyperbolic
space the volume of a ball grows exponentially with radius — matching
how hierarchical structures fan out. Hyperbolic distance compactly encodes
genealogical distance in trees and tree-like graphs (Krioukov et al. 2010,
"Hyperbolic Geometry of Complex Networks", Phys. Rev. E 82).

Standard placement (SA, eplace, RePlAce) optimizes in EUCLIDEAN canvas
space, fighting the metric mismatch: the cost function lives on a
connectivity manifold whose intrinsic geometry is hyperbolic, but the
search proceeds in flat space and has to discover the hierarchy via
random walks.

THE WARM-START
--------------
Embed the netlist hypergraph in hyperbolic space via the popularity-
similarity model:

  - Each macro gets a (radius r, angle θ) where:
    * r ∝ -log(degree)        — high-degree macros near origin (popular)
    * θ from spectral embedding — angular position by similarity

  - Project Poincaré disk to canvas:
    * r_canvas = R · tanh(α · r_hyp)   for canvas radius R, compress α

  - Hard macros take new positions; soft macros stay at their .plc
    init (preserves the v7 laplacian phase's downstream behavior).

The resulting placement has clusters of co-connected macros geometrically
close ON THE CANVAS, before SA even runs. SA's job reduces from "find
the hierarchy" to "polish locally."

Never been applied to chip placement (search literature: hyperbolic
embeddings exist for Internet topology, social networks, biological
networks — none for EDA placement that I can find).
"""
from __future__ import annotations
import numpy as np


def _build_macro_macro_adjacency(benchmark, incr):
    """Build hard-macro × hard-macro adjacency from netlist via
    the IncrementalEvaluator's pin_macro + net_starts arrays.

    Returns (A, deg) where A is symmetric n_hard×n_hard ndarray and
    deg is per-macro degree.
    """
    from scipy.sparse import csr_matrix, diags

    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    pin_macro = np.asarray(incr.pin_macro, dtype=np.int64)
    net_starts = np.asarray(incr.net_starts, dtype=np.int64)
    n_nets = len(net_starts) - 1
    n_pins = len(pin_macro)

    # Net index per pin
    pin_net = np.zeros(n_pins, dtype=np.int64)
    for j in range(n_nets):
        pin_net[net_starts[j]:net_starts[j + 1]] = j

    # Macro-net incidence: M[i,j] = 1 if macro i has any pin in net j
    valid = pin_macro >= 0  # ports have pin_macro = -1
    rows = pin_macro[valid]
    cols = pin_net[valid]
    data = np.ones(len(rows), dtype=np.float64)
    M = csr_matrix((data, (rows, cols)), shape=(n_total, n_nets))
    M.sum_duplicates()
    M.data = np.minimum(M.data, 1.0)  # boolean

    # Net-size weighting: pairs sharing a 2-pin net get full weight,
    # pairs sharing a 100-pin "clock" net get nearly nothing.
    net_size = np.asarray(M.sum(axis=0)).flatten()
    net_weight = 1.0 / np.maximum(net_size - 1, 1.0)
    W = diags(net_weight)

    # Hard-only adjacency: A = M_hard · W · M_hard^T
    M_hard = M[:n_hard]
    A_sparse = M_hard @ W @ M_hard.T
    A = np.asarray(A_sparse.todense())
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    return A, deg


def _spectral_angles(A, deg):
    """Spectral embedding into 2D angular coordinates.

    Computes the 2nd and 3rd smallest eigenvectors of the normalized
    Laplacian L_norm = I - D^(-1/2) A D^(-1/2). The 1st eigenvector is
    the trivial constant; 2nd and 3rd give the principal directions in
    the connectivity space.

    Maps (eigvec_2, eigvec_3) to angle θ = atan2(v3, v2).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    n = A.shape[0]
    if n < 4:
        # tiny graph — use uniform circle
        return np.linspace(0, 2 * np.pi, n, endpoint=False)

    eps = 1e-9
    D_inv_sqrt = diags(1.0 / np.sqrt(deg + eps))
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L_norm = np.eye(n) - A_sym
    try:
        eigvals, eigvecs = eigsh(L_norm, k=3, which="SM",
                                  maxiter=200, tol=1e-4)
        # Ensure eigvecs are sorted by eigvalue ascending
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        u = eigvecs[:, 1]
        v = eigvecs[:, 2]
    except Exception as e:
        # Fallback: spread macros uniformly on a circle
        return np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Normalize to avoid scale issues
    u_norm = u / (np.abs(u).max() + 1e-12)
    v_norm = v / (np.abs(v).max() + 1e-12)
    return np.arctan2(v_norm, u_norm)


def _hyperbolic_radii(deg):
    """Hyperbolic radius from degree: r ~ -log(deg / max_deg).

    High-degree (popular) macros → small radius (near origin).
    Low-degree macros → large radius (periphery).
    Output normalized to [0, 1].
    """
    deg_norm = deg / (deg.max() + 1e-9)
    r_hyp = -np.log(deg_norm + 1e-3)
    r_hyp /= (r_hyp.max() + 1e-9)
    return r_hyp


def hyperbolic_warm_start_positions(
    benchmark,
    plc,
    *,
    incr=None,
    canvas_margin_frac: float = 0.10,
    radius_compress: float = 1.5,
    verbose: bool = False,
):
    """Compute hyperbolic-embedding-based positions for HARD macros.

    Returns (n_total, 2) numpy array; soft macros unchanged.
    """
    if incr is None:
        # Lazy-build IncrementalEvaluator. Slow but only happens once
        # per place() call; the hyperbolic warm-start runs before the
        # heavy SA phase so the cost is negligible.
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[2]
        v1_dir = str(ROOT / "submissions" / "vmallela")
        if v1_dir not in sys.path:
            sys.path.insert(0, v1_dir)
        from placer import IncrementalEvaluator
        incr = IncrementalEvaluator(plc, benchmark)

    n_hard = benchmark.num_hard_macros
    A, deg = _build_macro_macro_adjacency(benchmark, incr)

    if (deg == 0).all():
        if verbose:
            print(f"  [hyperbolic] all-zero adjacency; skipping warm-start",
                  flush=True)
        return benchmark.macro_positions.numpy().copy().astype(np.float64)

    angles = _spectral_angles(A, deg)
    r_hyp = _hyperbolic_radii(deg)

    # Project to canvas
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    R = (1.0 - canvas_margin_frac) * 0.5 * min(canvas_w, canvas_h)
    r_canvas = R * np.tanh(radius_compress * r_hyp)

    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    new_x = cx + r_canvas * np.cos(angles)
    new_y = cy + r_canvas * np.sin(angles)

    # Clip to canvas accounting for half-macro-size (can't place center
    # at canvas edge or macro hangs off).
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    new_x = np.clip(new_x, half_w, canvas_w - half_w)
    new_y = np.clip(new_y, half_h, canvas_h - half_h)

    # Output: hard macros at hyperbolic positions, softs unchanged
    new_pos = benchmark.macro_positions.numpy().copy().astype(np.float64)
    new_pos[:n_hard, 0] = new_x
    new_pos[:n_hard, 1] = new_y

    if verbose:
        print(f"  [hyperbolic] n_hard={n_hard} deg=[{deg.min():.1f},{deg.max():.1f}] "
              f"r_canvas=[{r_canvas.min():.2f},{r_canvas.max():.2f}] "
              f"compress={radius_compress}", flush=True)
    return new_pos
