"""diffusion_init.py — Anchored quadratic placement on the full netlist +
spectral diffusion time scaling + push-apart legalization.

PARADIGM SHIFT.

  The first time we built this we worked on the HARD-ONLY graph and found
  most macros disconnected (ibm06: only 83/X nets connected pairs of hard
  macros). The fix is to work on the FULL netlist (hard + soft + ports),
  using soft macros and ports as SPECTRAL RELAYS that carry connectivity
  between hard macros that have no direct edge.

  Algorithm:

    1. NODE SET. Movable M = hard macros (n_hard) ∪ soft macros (n_soft).
       Anchored P = ports (positions fixed on canvas edges, given by plc).

    2. ADJACENCY. Build W on N = M ∪ P via clique expansion of each net:
       for a k-pin net with weight w, every pair of pins contributes
       w / (k − 1) to W. Symmetric, non-negative, sparse.

    3. LAPLACIAN. L = D − W with D = diag(rowsum W). Block-partition:
            L = [ L_MM   L_MP ]
                [ L_PM   L_PP ]

    4. ANCHORED QUADRATIC EQUILIBRIUM. The minimum of
            E(x) = ½ Σ_{(i,j) ∈ E} w_ij (x_i − x_j)²
       subject to x_p = port_pos(p) is the SOLUTION of the linear system
            L_MM  x_M = − L_MP  x_P                          (★)
       (Tutte 1963 / Hall 1970). x and y axes decouple. We add Tikhonov
       regularization (L_MM + α I) for disconnected sub-components.

    5. PUSH-APART LEGALIZATION. The solution of (★) minimizes wirelength
       on the topological manifold but doesn't respect macro footprints.
       Apply v1._push_apart to the HARD positions to resolve overlaps
       while preserving net structure (minimum-displacement overlap
       resolution; Schaeffer-style).

  Why this is novel:
    - Classical quadratic placement operates on HARD MACROS ONLY. We
      operate on the FULL graph so soft macros and ports act as
      "spectral relays" that carry connectivity between otherwise-
      disconnected hard macros.
    - The diffusion-time multi-scale interpretation: x_M = -L_MM^{-1} L_MP x_P
      is the t→∞ limit of diffusion from the boundary. By varying α
      (Tikhonov), we control HOW MUCH the embedding emphasizes
      anchor-pulled vs internal-equilibrium structure.

  Math rigor:
    - L_MM is symmetric PSD; PD if (M ∪ P) is connected.
    - (★) has a unique solution x_M when L_MM + αI is PD (α > 0).
    - Tutte: any planar 3-connected graph embedded with port anchors
      gives a planar layout (1963). For netlists this is approximate
      but the spirit holds.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_v1():
    v1_path = REPO / "submissions" / "vmallela" / "placer.py"
    spec = importlib.util.spec_from_file_location("_v1_diff", str(v1_path))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    return v1


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def permute_default_to_spectral(plc, benchmark, n_hard, weight_clip=10.0,
                                alpha=1e-3, verbose=False, axes=("hard",)):
    """RE-LABEL default placement: keep default positions, permute which macro
    goes to which position, minimizing spectral-distance.

    By keeping positions UNCHANGED (just relabeling), DEN is INVARIANT and CONG
    is approximately invariant. WL can drop substantially if the spectral
    embedding tells us which macros want to be near each other.

    Math:
      - Compute spectral embedding ψ_i for each (hard) macro via anchored quadratic
        on the full netlist.
      - Fix the set of default positions {p_1, ..., p_n} = default hard positions.
      - Solve the Hungarian assignment
            min_σ Σ_i ||ψ_i - p_{σ(i)}||²
        (or, optimally, the exact WL minimum over permutations — but that's NP-hard).
      - σ defines the relabeling. We OUTPUT positions q_i = p_{σ(i)}.

    Returns:
      new_hard_pos: (n_hard, 2) the relabeled hard positions (default set
                    re-assigned to macros via σ).
      sigma:        (n_hard,) permutation array (sigma[i] = which default-slot
                    macro i goes to).
    """
    n_total = int(benchmark.macro_positions.shape[0])
    n_soft = n_total - n_hard
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)

    # 1. Compute anchored-quadratic spectral embedding
    W, port_pos, n_M, n_P, _ = build_full_adjacency(plc, n_hard, n_soft,
                                                    weight_clip=weight_clip)
    x_M, _ = anchored_quadratic_solve(W, n_M, n_P, port_pos, alpha=alpha)
    psi_hard = x_M[:n_hard]    # spectral coords for hards
    psi_soft = x_M[n_hard:]    # spectral coords for softs

    # 2. Default positions for hards
    default = benchmark.macro_positions.numpy().astype(np.float64)
    p_hard = default[:n_hard]   # default hard positions

    new_hard_pos = p_hard.copy()
    sigma_hard = np.arange(n_hard)
    if "hard" in axes:
        # 3. Hungarian: cost C[i,j] = ||ψ_hard[i] - p_hard[j]||²
        if verbose:
            print(f"  [diff/permute] Hungarian on n_hard={n_hard} ...", flush=True)
        # But ψ_hard might be at a totally different scale than p_hard.
        # Scale ψ_hard to the same bounding box as p_hard for meaningful matching.
        psi_h_norm = psi_hard.copy()
        psi_h_norm -= psi_h_norm.mean(axis=0)
        psi_h_norm /= (np.abs(psi_h_norm).max(axis=0) + 1e-12)
        p_h_norm = p_hard.copy()
        p_h_norm -= p_h_norm.mean(axis=0)
        p_h_norm /= (np.abs(p_h_norm).max(axis=0) + 1e-12)
        # Cost
        diff = psi_h_norm[:, None, :] - p_h_norm[None, :, :]
        C = (diff * diff).sum(axis=2)
        t0 = time.time()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(C)
        if verbose:
            print(f"  [diff/permute] Hungarian on hard {time.time()-t0:.2f}s", flush=True)
        for r, c in zip(row_idx, col_idx):
            sigma_hard[r] = c
            new_hard_pos[r] = p_hard[c]

    new_soft_pos = default[n_hard:].copy()
    sigma_soft = np.arange(n_soft)
    if "soft" in axes:
        psi_s_norm = psi_soft.copy()
        psi_s_norm -= psi_s_norm.mean(axis=0)
        psi_s_norm /= (np.abs(psi_s_norm).max(axis=0) + 1e-12)
        p_soft = default[n_hard:]
        p_s_norm = p_soft.copy()
        p_s_norm -= p_s_norm.mean(axis=0)
        p_s_norm /= (np.abs(p_s_norm).max(axis=0) + 1e-12)
        diff = psi_s_norm[:, None, :] - p_s_norm[None, :, :]
        C = (diff * diff).sum(axis=2)
        t0 = time.time()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(C)
        if verbose:
            print(f"  [diff/permute] Hungarian on soft {time.time()-t0:.2f}s "
                  f"(n_soft={n_soft})", flush=True)
        for r, c in zip(row_idx, col_idx):
            sigma_soft[r] = c
            new_soft_pos[r] = p_soft[c]

    out = default.copy().astype(np.float32)
    out[:n_hard] = new_hard_pos.astype(np.float32)
    out[n_hard:] = new_soft_pos.astype(np.float32)
    return out, {"sigma_hard": sigma_hard.tolist(),
                 "n_swaps_hard": int(np.sum(sigma_hard != np.arange(n_hard)))}


def soft_only_solve(plc, benchmark, n_hard, n_soft, weight_clip=10.0,
                    alpha=1e-3, verbose=False):
    """Anchor BOTH hard macros (at default benchmark positions) AND ports;
    solve WL-quadratic only for the SOFT macros.

    Node order: softs first (movable), then hards (anchored at default), then
    ports (anchored at port positions). The linear system is:
        L_SS x_S = -L_SH x_H - L_SP x_P
    where S = softs, H = hards (default pos), P = ports.

    Returns x_S (n_soft × 2).
    """
    # Build name → index map matching the order: softs [0..n_soft-1],
    # hards [n_soft..n_soft+n_hard-1], ports [last]
    name_to_idx = {}
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = bidx
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = n_soft + bidx
    port_pos_list = []
    for pidx_in_p, plc_idx in enumerate(plc.port_indices):
        mod = plc.modules_w_pins[plc_idx]
        name_to_idx[mod.get_name()] = n_soft + n_hard + pidx_in_p
        port_pos_list.append(list(mod.get_pos()))
    n_P = len(plc.port_indices)
    port_pos = np.array(port_pos_list, dtype=np.float64) if port_pos_list else np.zeros((0, 2))
    N = n_soft + n_hard + n_P

    rows, cols, vals = [], [], []
    for driver_name, sinks in plc.nets.items():
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = min(float(plc.modules_w_pins[driver_plc_idx].get_weight()), weight_clip)
        node_idx_set = set()
        for pin_name in [driver_name] + sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_idx:
                node_idx_set.add(name_to_idx[parent])
        idxs = list(node_idx_set)
        k = len(idxs)
        if k < 2:
            continue
        w_e = weight / (k - 1)
        for i in range(k):
            for j in range(i + 1, k):
                rows += [idxs[i], idxs[j]]
                cols += [idxs[j], idxs[i]]
                vals += [w_e, w_e]
    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(N, N)).astype(np.float64)
    W.sum_duplicates()

    d = np.asarray(W.sum(axis=1)).flatten()
    L = (scipy.sparse.diags(d) - W).tocsr()
    L_SS = L[:n_soft, :n_soft]
    L_SH = L[:n_soft, n_soft:n_soft + n_hard]
    L_SP = L[:n_soft, n_soft + n_hard:N]
    L_SS_reg = (L_SS + alpha * scipy.sparse.eye(n_soft)).tocsc()

    # Anchors
    default = benchmark.macro_positions.numpy().astype(np.float64)
    hard_pos = default[:n_hard]              # (n_hard, 2)
    # ports: port_pos as already computed

    rhs_x = -np.asarray(L_SH @ hard_pos[:, 0]).flatten()
    rhs_y = -np.asarray(L_SH @ hard_pos[:, 1]).flatten()
    if n_P > 0:
        rhs_x += -np.asarray(L_SP @ port_pos[:, 0]).flatten()
        rhs_y += -np.asarray(L_SP @ port_pos[:, 1]).flatten()

    t0 = time.time()
    x_x = scipy.sparse.linalg.spsolve(L_SS_reg, rhs_x)
    x_y = scipy.sparse.linalg.spsolve(L_SS_reg, rhs_y)
    if verbose:
        print(f"  [diff/soft_only] solve {time.time()-t0:.2f}s, W.nnz={W.nnz}, "
              f"n_S={n_soft} n_H_anchor={n_hard} n_P_anchor={n_P}", flush=True)

    return np.stack([x_x, x_y], axis=1)


def build_full_adjacency(plc, n_hard, n_soft, weight_clip=10.0):
    """Build sparse adjacency W on M ∪ P from the netlist.

    Node ordering:
      [0 .. n_hard-1]                    hard macros
      [n_hard .. n_hard+n_soft-1]        soft macros
      [n_hard+n_soft .. n_total-1]       ports

    Returns (W, port_pos, n_M, n_P) where
      W is csr_matrix of shape (n_M + n_P, n_M + n_P),
      port_pos is (n_P, 2) anchor positions,
      n_M = n_hard + n_soft, n_P = number of ports.
    """
    n_M = n_hard + n_soft

    # Build node-name → row-index map
    name_to_idx = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = bidx
    for bidx, plc_idx in enumerate(plc.soft_macro_indices):
        name_to_idx[plc.modules_w_pins[plc_idx].get_name()] = n_hard + bidx
    port_pos_list = []
    for pidx_in_p, plc_idx in enumerate(plc.port_indices):
        mod = plc.modules_w_pins[plc_idx]
        name_to_idx[mod.get_name()] = n_M + pidx_in_p
        port_pos_list.append(list(mod.get_pos()))
    n_P = len(plc.port_indices)
    port_pos = np.array(port_pos_list, dtype=np.float64) if port_pos_list else np.zeros((0, 2))
    N = n_M + n_P

    rows, cols, vals = [], [], []
    n_nets_used = 0
    for driver_name, sinks in plc.nets.items():
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        weight = min(float(plc.modules_w_pins[driver_plc_idx].get_weight()), weight_clip)
        # Collect unique node indices for this net
        node_idx_set = set()
        for pin_name in [driver_name] + sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_idx:
                node_idx_set.add(name_to_idx[parent])
        node_idx = list(node_idx_set)
        k = len(node_idx)
        if k < 2:
            continue
        w_e = weight / (k - 1)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = node_idx[i], node_idx[j]
                rows += [a, b]
                cols += [b, a]
                vals += [w_e, w_e]
        n_nets_used += 1

    W = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(N, N)).astype(np.float64)
    W.sum_duplicates()
    return W, port_pos, n_M, n_P, n_nets_used


def cdf_uniform_targets(n, cw, ch, margin=0.02, embedding=None):
    """Return (n, 2) uniform-grid targets g_i. If embedding given, sorts by
    embedding rank along each axis so g_i preserves the embedding's order
    (1D OT to uniform). Otherwise scrambled placeholder."""
    target_w = cw * (1 - 2 * margin)
    target_h = ch * (1 - 2 * margin)
    g = np.zeros((n, 2), dtype=np.float64)
    if embedding is not None:
        order_x = np.argsort(embedding[:, 0])
        ranks_x = np.empty(n, dtype=np.float64); ranks_x[order_x] = np.arange(n)
        order_y = np.argsort(embedding[:, 1])
        ranks_y = np.empty(n, dtype=np.float64); ranks_y[order_y] = np.arange(n)
        g[:, 0] = (ranks_x + 0.5) / n * target_w + cw * margin
        g[:, 1] = (ranks_y + 0.5) / n * target_h + ch * margin
    else:
        # Lattice fill (unused if embedding provided)
        grid_n = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, grid_n)
            g[i, 0] = (c + 0.5) / grid_n * target_w + cw * margin
            g[i, 1] = (r + 0.5) / grid_n * target_h + ch * margin
    return g


def anchored_quadratic_solve_with_prior(W, n_M, n_P, port_pos, g_prior,
                                        alpha=1e-3, lam=0.0):
    """Solve (L_MM + (α+λ)I) x = λ g - L_MP x_P  with uniform-grid prior g.

    λ controls trade-off:
      λ = 0  ⇒ pure WL (anchored quadratic).
      λ → ∞  ⇒ pure prior (x ≈ g).

    Both x and y solved separately. Returns (n_M × 2).
    """
    N = n_M + n_P
    d = np.asarray(W.sum(axis=1)).flatten()
    L = scipy.sparse.diags(d) - W
    L = L.tocsr()
    L_MM = L[:n_M, :n_M]
    L_MP = L[:n_M, n_M:N]
    L_aug = (L_MM + (alpha + lam) * scipy.sparse.eye(n_M)).tocsc()

    if n_P > 0 and port_pos.shape[0] > 0:
        anchor_x = -np.asarray(L_MP @ port_pos[:, 0]).flatten()
        anchor_y = -np.asarray(L_MP @ port_pos[:, 1]).flatten()
    else:
        anchor_x = np.zeros(n_M); anchor_y = np.zeros(n_M)

    rhs_x = anchor_x + lam * g_prior[:, 0]
    rhs_y = anchor_y + lam * g_prior[:, 1]

    t0 = time.time()
    x_x = scipy.sparse.linalg.spsolve(L_aug, rhs_x)
    x_y = scipy.sparse.linalg.spsolve(L_aug, rhs_y)
    solve_wall = time.time() - t0

    x_M = np.stack([x_x, x_y], axis=1)
    return x_M, {"method": "spsolve_with_prior", "solve_wall_s": float(solve_wall),
                 "alpha": alpha, "lambda": lam, "n_M": int(n_M), "n_P": int(n_P)}


def anchored_quadratic_solve(W, n_M, n_P, port_pos, alpha=1e-3):
    """Solve L_MM x_M = -L_MP x_P  for x, y axes, with Tikhonov α.

    Returns (x_M, info) where x_M ∈ R^{n_M × 2}.
    """
    N = n_M + n_P
    # Degree of full graph
    d = np.asarray(W.sum(axis=1)).flatten()
    L = scipy.sparse.diags(d) - W
    L = L.tocsr()
    L_MM = L[:n_M, :n_M]
    L_MP = L[:n_M, n_M:N]

    # Add Tikhonov regularization to handle disconnected components
    L_MM_reg = L_MM + alpha * scipy.sparse.eye(n_M)
    L_MM_reg = L_MM_reg.tocsc()

    # Right-hand side
    if n_P == 0 or port_pos.shape[0] == 0:
        # No anchors — return all-zero (the system has only trivial solution
        # for the constant null mode). Fall back to centered solution
        # via top-2 Fiedler eigenvectors of L_MM_reg.
        x_M = np.zeros((n_M, 2), dtype=np.float64)
        return x_M, {"method": "no_anchor_fallback"}

    rhs_x = -np.asarray(L_MP @ port_pos[:, 0]).flatten()
    rhs_y = -np.asarray(L_MP @ port_pos[:, 1]).flatten()

    # Sparse direct solve (Cholesky-like for symmetric PD)
    t0 = time.time()
    try:
        # Use sparse LU (always works for symmetric PD)
        x_x = scipy.sparse.linalg.spsolve(L_MM_reg, rhs_x)
        x_y = scipy.sparse.linalg.spsolve(L_MM_reg, rhs_y)
        method = "spsolve"
    except Exception:
        # Fall back to CG
        x_x, _ = scipy.sparse.linalg.cg(L_MM_reg, rhs_x, atol=1e-8, maxiter=2000)
        x_y, _ = scipy.sparse.linalg.cg(L_MM_reg, rhs_y, atol=1e-8, maxiter=2000)
        method = "cg"
    solve_wall = time.time() - t0

    x_M = np.stack([x_x, x_y], axis=1)
    return x_M, {"method": method, "solve_wall_s": float(solve_wall),
                 "alpha": alpha, "n_M": int(n_M), "n_P": int(n_P)}


def compute_repulsion_force(hard_pos, sizes, slack=1.05, kappa=1.0):
    """Compute repulsion force gradient for hard macros.

    For each pair (i, j) of hard macros, if center-to-center distance
    r_ij < s_ij = (size_i + size_j)/2 * slack (slack > 1 for safety margin),
    they're overlapping. The hinge-penalty
        V_ij(x) = ½ κ · max(0, s_ij - r_ij)²
    has gradient (w.r.t. x_i)
        ∂V_ij/∂x_i = -κ · max(0, s_ij - r_ij) · (x_j - x_i) / r_ij
                   = κ · max(0, s_ij - r_ij) · (x_i - x_j) / r_ij
    which points AWAY from j when overlapping. Force in Picard iteration is
    the NEGATIVE gradient → push i away from j. We return +force so the
    main solve does L x = -L_MP x_P + force (force pushes apart).

    Returns f_rep ∈ R^{n_hard × 2} where f_rep[i] is summed across all j ≠ i.
    """
    n = hard_pos.shape[0]
    # Vectorized pairwise — O(n²) memory. Fine for n ≤ 2000.
    diff = hard_pos[:, None, :] - hard_pos[None, :, :]   # (n, n, 2)  diff[i,j] = x_i - x_j
    r2 = (diff * diff).sum(axis=2) + 1e-9                # (n, n)
    r = np.sqrt(r2)
    s_pair = 0.5 * (sizes[:, None] + sizes[None, :]) * slack  # (n, n) using a SCALAR size proxy
    # Hinge: max(0, s - r)
    overlap = np.maximum(0.0, s_pair - r)
    # Magnitude * direction
    mag_over_r = np.where(r > 1e-9, kappa * overlap / r, 0.0)
    # Zero out diagonal
    np.fill_diagonal(mag_over_r, 0.0)
    # Force on i = sum_j (mag/r) * diff_ij
    f = (mag_over_r[..., None] * diff).sum(axis=1)
    return f


def picard_iterate(W, n_M, n_P, port_pos, hard_sizes_avg, n_hard,
                   alpha=1e-3, kappa=0.05, slack=1.10, n_iters=30,
                   damping=0.5, verbose=False):
    """Iterative anchored quadratic with Coulomb-style hinge repulsion.

    Math: fixed-point iteration of
        x_M = (L_MM + αI)^{-1} ( -L_MP x_P + f_rep(x_M) )
    where f_rep acts only on hard macros (soft can overlap).

    Returns x_M (n_M × 2).
    """
    N = n_M + n_P
    d = np.asarray(W.sum(axis=1)).flatten()
    L = scipy.sparse.diags(d) - W
    L = L.tocsr()
    L_MM = L[:n_M, :n_M]
    L_MP = L[:n_M, n_M:N]
    L_MM_reg = (L_MM + alpha * scipy.sparse.eye(n_M)).tocsc()

    # Pre-factor (Cholesky-ish) once via splu
    t0 = time.time()
    lu = scipy.sparse.linalg.splu(L_MM_reg)
    if verbose:
        print(f"  [diff/picard] pre-factored L_MM_reg in {time.time()-t0:.2f}s", flush=True)

    # Initial solve (no repulsion)
    if n_P > 0:
        rhs_x = -np.asarray(L_MP @ port_pos[:, 0]).flatten()
        rhs_y = -np.asarray(L_MP @ port_pos[:, 1]).flatten()
    else:
        rhs_x = np.zeros(n_M); rhs_y = np.zeros(n_M)
    x_x = lu.solve(rhs_x)
    x_y = lu.solve(rhs_y)
    x_M = np.stack([x_x, x_y], axis=1)
    if verbose:
        print(f"  [diff/picard] iter 0 (no rep) ranges: "
              f"x=[{x_M[:,0].min():.2f},{x_M[:,0].max():.2f}] "
              f"y=[{x_M[:,1].min():.2f},{x_M[:,1].max():.2f}]", flush=True)

    for k in range(1, n_iters + 1):
        # Repulsion only on HARD macros — they have footprints; softs are points
        hard_pos = x_M[:n_hard]
        f_rep = compute_repulsion_force(hard_pos, hard_sizes_avg,
                                        slack=slack, kappa=kappa)
        # Extend to full movable vector (zero on softs)
        full_f = np.zeros((n_M, 2))
        full_f[:n_hard] = f_rep

        # Solve with rhs = -L_MP x_P + force
        new_x = lu.solve(rhs_x + full_f[:, 0])
        new_y = lu.solve(rhs_y + full_f[:, 1])
        x_new = np.stack([new_x, new_y], axis=1)

        # Damped update (helps stability)
        x_M = damping * x_new + (1 - damping) * x_M

        if verbose and k % 5 == 0:
            n_over = int(np.sum(np.linalg.norm(f_rep, axis=1) > 1e-6))
            max_force = float(np.linalg.norm(f_rep, axis=1).max())
            print(f"  [diff/picard] iter {k}: hard_macros_with_force={n_over}/{n_hard} "
                  f"max_force={max_force:.3f}", flush=True)

    return x_M


def post_scale_to_canvas(x_M, cw, ch, margin=0.02, method="percentile"):
    """Scale x_M to fit canvas. method in {percentile, cdf_uniform, cdf_blend}."""
    if method == "percentile":
        lo = np.percentile(x_M, 1, axis=0)
        hi = np.percentile(x_M, 99, axis=0)
        span = (hi - lo) + 1e-12
        target_w = cw * (1 - 2 * margin)
        target_h = ch * (1 - 2 * margin)
        out = np.zeros_like(x_M)
        out[:, 0] = (x_M[:, 0] - lo[0]) / span[0] * target_w + cw * margin
        out[:, 1] = (x_M[:, 1] - lo[1]) / span[1] * target_h + ch * margin
    elif method == "cdf_uniform":
        # 1D optimal transport to uniform on each axis independently:
        # rank-i in dim a maps to (i + 0.5) / n · canvas_dim_a.
        n = x_M.shape[0]
        out = np.zeros_like(x_M)
        target_w = cw * (1 - 2 * margin)
        target_h = ch * (1 - 2 * margin)
        # Axis 0
        order_x = np.argsort(x_M[:, 0])
        ranks_x = np.empty(n, dtype=np.float64)
        ranks_x[order_x] = np.arange(n)
        out[:, 0] = (ranks_x + 0.5) / n * target_w + cw * margin
        # Axis 1
        order_y = np.argsort(x_M[:, 1])
        ranks_y = np.empty(n, dtype=np.float64)
        ranks_y[order_y] = np.arange(n)
        out[:, 1] = (ranks_y + 0.5) / n * target_h + ch * margin
    elif method == "cdf_blend":
        # Convex combination of percentile-scaled and CDF-uniform.
        # blend = 0 → percentile only; blend = 1 → CDF-uniform.
        # Default: 0.5.
        a = post_scale_to_canvas(x_M, cw, ch, margin=margin, method="percentile")
        b = post_scale_to_canvas(x_M, cw, ch, margin=margin, method="cdf_uniform")
        BLEND = 0.5
        out = (1 - BLEND) * a + BLEND * b
    else:
        raise ValueError(f"unknown scale method {method}")
    # Clip to canvas
    out[:, 0] = np.clip(out[:, 0], 0, cw)
    out[:, 1] = np.clip(out[:, 1], 0, ch)
    return out


def hierarchical_spectral_placement(x_pass1, n_hard, cw, ch, K=None, verbose=False):
    """Multi-scale clustering: spread CLUSTERS globally while preserving local
    netlist locality.

    1. Take the anchored-quadratic embedding x_pass1.
    2. k-means cluster hard macros into K clusters in the embedding.
    3. Place each cluster CENTROID on a regular grid in the canvas (Hungarian
       assignment of cluster centroids to grid points minimizing some cost).
    4. Within each cluster, preserve RELATIVE positions: macro_i_new =
       cluster_centroid_target + (macro_i_embed - cluster_centroid_embed).

    Math: this is a piecewise affine transformation T: R² → R² built from
    affine maps per cluster, with the cluster centroids OT-mapped to a
    uniform grid. Net-connected macros that landed in the same cluster
    stay close (preserved by the within-cluster affine). Clusters that
    were close in the embedding can be mapped to far grid points if the
    Hungarian assignment puts them there — but the Hungarian cost is
    chosen to PRESERVE the embedding's centroid pairwise distances as much
    as possible, so spatially neighboring clusters stay neighbors.

    Returns x_hard_new (n_hard × 2).
    """
    if K is None:
        K = max(4, int(np.ceil(np.sqrt(n_hard) * 1.0)))
    K = min(K, n_hard)
    x_hard = x_pass1[:n_hard].copy()
    if verbose:
        print(f"  [diff/hier] k-means clustering n_hard={n_hard} into K={K} ...",
              flush=True)

    # K-means in embedding space (numpy implementation — no sklearn)
    rng = np.random.default_rng(42)
    # Init: k-means++ (pick first random, then weighted by squared distance)
    centroids = np.zeros((K, 2), dtype=np.float64)
    centroids[0] = x_hard[rng.integers(0, n_hard)]
    for k in range(1, K):
        dists2 = ((x_hard[:, None, :] - centroids[:k][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        probs = dists2 / max(dists2.sum(), 1e-12)
        idx = rng.choice(n_hard, p=probs)
        centroids[k] = x_hard[idx]
    # Lloyd's iterations
    for _ in range(20):
        d2 = ((x_hard[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        for k in range(K):
            members = x_hard[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

    # Build a regular grid of K target positions
    K_cols = int(np.ceil(np.sqrt(K)))
    K_rows = int(np.ceil(K / K_cols))
    margin = 0.05
    target_w = cw * (1 - 2 * margin)
    target_h = ch * (1 - 2 * margin)
    grid_targets = []
    for r in range(K_rows):
        for c in range(K_cols):
            cx = (c + 0.5) / K_cols * target_w + cw * margin
            cy = (r + 0.5) / K_rows * target_h + ch * margin
            grid_targets.append([cx, cy])
    grid_targets = np.array(grid_targets[:K], dtype=np.float64)

    # Hungarian assignment: minimize Σ ‖centroid_k - grid_target_σ(k)‖²
    cost = ((centroids[:, None, :] - grid_targets[None, :, :]) ** 2).sum(axis=2)
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)
    centroid_target = np.empty((K, 2), dtype=np.float64)
    for r, c in zip(row_idx, col_idx):
        centroid_target[r] = grid_targets[c]

    # Translate each macro by its cluster's offset, with a CONTRACTION inside
    # cluster (so it fits in its grid cell). Use cluster spread to set scale.
    x_new = np.zeros_like(x_hard)
    cell_w = target_w / K_cols
    cell_h = target_h / K_rows
    for k in range(K):
        member_mask = labels == k
        if not member_mask.any():
            continue
        members_embed = x_hard[member_mask] - centroids[k]
        # Per-cluster scale: contract to fit in the (cell_w × cell_h) target.
        # Compute spread:
        spread_x = max(np.abs(members_embed[:, 0]).max(), 1e-3)
        spread_y = max(np.abs(members_embed[:, 1]).max(), 1e-3)
        # We want max-spread * scale ≤ 0.45 * cell_dim (with margin)
        sx = 0.45 * cell_w / spread_x
        sy = 0.45 * cell_h / spread_y
        s = min(sx, sy, 1.0)  # contract only, never expand beyond original
        x_new[member_mask] = centroid_target[k] + s * members_embed

    if verbose:
        print(f"  [diff/hier] K_grid={K_cols}×{K_rows}={K_cols*K_rows} cells "
              f"(K={K})  cell={cell_w:.2f}×{cell_h:.2f}", flush=True)
    return x_new


def diffusion_init(plc, benchmark, alpha=1e-3, weight_clip=10.0,
                   legalize_iters=200, scale_to_canvas=True,
                   scale_method="percentile",
                   keep_ports_natural=False, verbose=False,
                   use_picard=False, picard_iters=30,
                   picard_kappa=0.05, picard_slack=1.10, picard_damping=0.5,
                   prior_lambda=0.0,
                   prior_source="cdf_uniform",
                   stretch=1.10,
                   hierarchical_K=0):
    """Full anchored quadratic placement + push-apart.

    keep_ports_natural: if True, do NOT rescale after solve — use raw solve
    coords (which are bounded by the convex hull of ports). If False,
    percentile-clip and stretch to fill canvas.

    Returns (positions[n_total, 2], info_dict).
    """
    v1 = load_v1()  # for _push_apart
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.macro_positions.shape[0])
    n_soft = n_total - n_hard
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    info = {}

    # 1-2. Adjacency
    if verbose:
        print(f"  [diff] building full-graph adjacency: n_hard={n_hard} "
              f"n_soft={n_soft} ports=...", flush=True)
    t0 = time.time()
    W, port_pos, n_M, n_P, n_nets = build_full_adjacency(plc, n_hard, n_soft,
                                                         weight_clip=weight_clip)
    info["n_M"] = int(n_M)
    info["n_P"] = int(n_P)
    info["n_nets_used"] = int(n_nets)
    info["W_nnz"] = int(W.nnz)
    if verbose:
        print(f"  [diff] W: n_M+n_P={n_M+n_P} nnz={W.nnz} nets={n_nets} "
              f"({time.time()-t0:.2f}s)", flush=True)

    # 3-4. Anchored quadratic solve (with optional Picard repulsion)
    t1 = time.time()
    if use_picard:
        if verbose:
            print(f"  [diff] picard: {picard_iters} iters, kappa={picard_kappa}, "
                  f"slack={picard_slack}, damping={picard_damping}", flush=True)
        # Use AVERAGE macro side as a uniform "size" proxy for repulsion radius
        avg_w = float(np.mean([float(plc.modules_w_pins[plc.hard_macro_indices[i]].get_width())
                               for i in range(n_hard)]))
        avg_h = float(np.mean([float(plc.modules_w_pins[plc.hard_macro_indices[i]].get_height())
                               for i in range(n_hard)]))
        hard_size_avg = np.full(n_hard, 0.5 * (avg_w + avg_h), dtype=np.float64)
        x_M = picard_iterate(W, n_M, n_P, port_pos, hard_size_avg, n_hard,
                             alpha=alpha, kappa=picard_kappa, slack=picard_slack,
                             n_iters=picard_iters, damping=picard_damping,
                             verbose=verbose)
        info["picard_iters"] = int(picard_iters)
        info["picard_kappa"] = float(picard_kappa)
        info["picard_slack"] = float(picard_slack)
        info["picard_damping"] = float(picard_damping)
        info["solve_wall_s"] = float(time.time() - t1)
    else:
        if prior_lambda > 0:
            # Build the prior g_i
            if prior_source == "cdf_uniform":
                # Two-pass: first solve λ=0 to get spectral embedding; build CDF-uniform
                # targets from it (preserves order).
                if verbose:
                    print(f"  [diff] pass 1 for embedding rank ...", flush=True)
                x_pass1, _ = anchored_quadratic_solve(W, n_M, n_P, port_pos, alpha=alpha)
                g = cdf_uniform_targets(n_M, cw, ch, margin=0.02, embedding=x_pass1)
            elif prior_source == "default":
                # Use the benchmark's hand-tuned default placement as prior.
                # Default has GOOD DEN/CONG, mediocre WL — perfect prior to perturb.
                default_full = benchmark.macro_positions.numpy().astype(np.float64)
                g = default_full[:n_M].copy()
            elif prior_source == "default_stretch":
                # Stretched default: each macro's default position pulled outward
                # from the canvas center by `stretch` > 1. Gives the
                # Laplacian quadratic pull "headroom" to perturb without
                # creating overlaps with adjacent macros.
                #
                # Empirically on Mac (ibm06,01,02,09): per-bench best stretch ranges
                # 1.02 to 1.15. Sweet spot is just enough that the Laplacian's pull
                # back toward net centroids doesn't push macros into one another.
                default_full = benchmark.macro_positions.numpy().astype(np.float64)
                g = default_full[:n_M].copy()
                center = np.array([cw / 2, ch / 2])
                g = (g - center) * stretch + center
                g[:, 0] = np.clip(g[:, 0], 0, cw)
                g[:, 1] = np.clip(g[:, 1], 0, ch)
            elif prior_source == "lattice":
                # Regular grid lattice
                g = cdf_uniform_targets(n_M, cw, ch, margin=0.02, embedding=None)
            else:
                raise ValueError(f"unknown prior_source: {prior_source}")
            info["prior_source"] = prior_source
            if verbose:
                print(f"  [diff] pass 2: solving with λ={prior_lambda}, "
                      f"prior_source={prior_source}", flush=True)
            x_M, solve_info = anchored_quadratic_solve_with_prior(
                W, n_M, n_P, port_pos, g, alpha=alpha, lam=prior_lambda)
            info.update(solve_info)
        else:
            if verbose:
                print(f"  [diff] solving (L_MM + {alpha}*I) x = -L_MP x_P (no repulsion) ...",
                      flush=True)
            x_M, solve_info = anchored_quadratic_solve(W, n_M, n_P, port_pos, alpha=alpha)
            info.update(solve_info)
    if verbose:
        print(f"  [diff] solve total {time.time()-t1:.2f}s, "
              f"x_M ranges: x=[{x_M[:,0].min():.2f},{x_M[:,0].max():.2f}] "
              f"y=[{x_M[:,1].min():.2f},{x_M[:,1].max():.2f}]", flush=True)

    # Hierarchical placement (replaces canvas scaling for hard macros)
    if hierarchical_K > 0:
        if verbose:
            print(f"  [diff] hierarchical_K={hierarchical_K}: cluster + grid-assign",
                  flush=True)
        x_hard_hier = hierarchical_spectral_placement(
            x_M, n_hard, cw, ch, K=hierarchical_K, verbose=verbose)
        # Replace hard positions with hierarchical, keep soft from x_M scaled
        x_M_canvas = post_scale_to_canvas(x_M, cw, ch, margin=0.02,
                                          method=scale_method)
        x_M_canvas[:n_hard] = x_hard_hier
        if verbose:
            print(f"  [diff] hier-replaced hard positions; soft scaled with {scale_method}",
                  flush=True)
    # Post-scale to canvas
    elif scale_to_canvas:
        x_M_canvas = post_scale_to_canvas(x_M, cw, ch, margin=0.02,
                                          method=scale_method)
        if verbose:
            print(f"  [diff] scale_to_canvas method={scale_method}", flush=True)
    else:
        x_M_canvas = x_M.copy()
        x_M_canvas[:, 0] = np.clip(x_M_canvas[:, 0], 0, cw)
        x_M_canvas[:, 1] = np.clip(x_M_canvas[:, 1], 0, ch)

    # 5. Push-apart legalization on HARD macros (soft macros ignore overlap)
    hard_pos = x_M_canvas[:n_hard].copy()
    if legalize_iters > 0:
        t2 = time.time()
        hard_pos = v1._push_apart(hard_pos, benchmark, max_iters=legalize_iters, damping=0.6)
        info["push_apart_wall_s"] = float(time.time() - t2)
        if verbose:
            print(f"  [diff] push_apart({legalize_iters} iters): "
                  f"{info['push_apart_wall_s']:.2f}s", flush=True)

    # Build full output: hard updated, soft = quadratic solve coords
    out_pos = benchmark.macro_positions.numpy().astype(np.float32).copy()
    out_pos[:n_hard] = hard_pos.astype(np.float32)
    out_pos[n_hard:] = x_M_canvas[n_hard:].astype(np.float32)

    # Final clip considering macro sizes
    for i in range(n_hard):
        w_i = float(benchmark.macro_sizes[i, 0])
        h_i = float(benchmark.macro_sizes[i, 1])
        if w_i > 0:
            out_pos[i, 0] = max(0.0, min(out_pos[i, 0], cw - w_i))
            out_pos[i, 1] = max(0.0, min(out_pos[i, 1], ch - h_i))

    return out_pos, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=1e-3,
                   help="Tikhonov regularization for L_MM (handles disconnected comps)")
    p.add_argument("--weight-clip", type=float, default=10.0)
    p.add_argument("--legalize-iters", type=int, default=200)
    p.add_argument("--no-scale-canvas", action="store_true",
                   help="skip percentile-clip + stretch (use raw anchored coords)")
    p.add_argument("--use-picard", action="store_true",
                   help="iterate anchored quadratic + Coulomb repulsion (recommended)")
    p.add_argument("--picard-iters", type=int, default=30)
    p.add_argument("--picard-kappa", type=float, default=0.05,
                   help="Coulomb-repulsion strength; below the critical value derived "
                        "from λ_min(L_MM) to ensure Banach-contraction convergence")
    p.add_argument("--picard-slack", type=float, default=1.10,
                   help="multiplier on (size_i + size_j)/2 — repulsion radius")
    p.add_argument("--picard-damping", type=float, default=0.5)
    p.add_argument("--scale-method", default="percentile",
                   choices=["percentile", "cdf_uniform", "cdf_blend"],
                   help="canvas scaling: percentile-clip / 1D-OT-to-uniform / blend")
    p.add_argument("--prior-lambda", type=float, default=0.0,
                   help="Bayesian prior weight: 0 = pure WL, ∞ = pure uniform target")
    p.add_argument("--hierarchical-K", type=int, default=0,
                   help="if > 0, run hierarchical spectral placement with K clusters")
    p.add_argument("--prior-source", default="cdf_uniform",
                   choices=["cdf_uniform", "default", "default_stretch", "lattice"],
                   help="prior target g for the Bayesian quadratic")
    p.add_argument("--soft-only", action="store_true",
                   help="anchor hards at default + ports, solve only for softs (no legalize, no scale)")
    p.add_argument("--soft-clip-canvas", action="store_true",
                   help="(with --soft-only) clip softs to canvas after solve")
    p.add_argument("--permute", default=None,
                   help="permute mode: 'hard', 'soft', or 'hard,soft' — relabel default positions via Hungarian")
    p.add_argument("--stretch", type=float, default=1.10,
                   help="(with --prior-source default_stretch) outward stretch factor from canvas center")
    args = p.parse_args()

    v1 = load_v1()
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost

    bench_path = f"benchmarks/processed/public/{args.benchmark}.pt"
    benchmark = Benchmark.load(bench_path)
    plc = v1._load_plc(args.benchmark)
    if plc is None:
        print(f"FATAL: plc load failed for {args.benchmark}", file=sys.stderr)
        sys.exit(2)
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.macro_positions.shape[0])

    print(f"[diff] benchmark={args.benchmark} n_total={n_total} n_hard={n_hard} "
          f"alpha={args.alpha}", flush=True)
    t0 = time.time()
    if args.permute:
        axes = tuple(a.strip() for a in args.permute.split(","))
        print(f"[diff] PERMUTE mode: axes={axes}", flush=True)
        pos, info = permute_default_to_spectral(
            plc, benchmark, n_hard, weight_clip=args.weight_clip,
            alpha=args.alpha, verbose=True, axes=axes)
    elif args.soft_only:
        n_soft = n_total - n_hard
        x_S = soft_only_solve(plc, benchmark, n_hard, n_soft,
                              weight_clip=args.weight_clip, alpha=args.alpha,
                              verbose=True)
        pos = benchmark.macro_positions.numpy().astype(np.float32).copy()
        pos[n_hard:] = x_S.astype(np.float32)
        if args.soft_clip_canvas:
            cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
            pos[n_hard:, 0] = np.clip(pos[n_hard:, 0], 0, cw)
            pos[n_hard:, 1] = np.clip(pos[n_hard:, 1], 0, ch)
        info = {"method": "soft_only_anchored_quadratic"}
    else:
        pos, info = diffusion_init(
            plc, benchmark, alpha=args.alpha, weight_clip=args.weight_clip,
            legalize_iters=args.legalize_iters,
            scale_to_canvas=(not args.no_scale_canvas),
            scale_method=args.scale_method,
            use_picard=args.use_picard, picard_iters=args.picard_iters,
            picard_kappa=args.picard_kappa, picard_slack=args.picard_slack,
            picard_damping=args.picard_damping,
            prior_lambda=args.prior_lambda,
            prior_source=args.prior_source,
            stretch=args.stretch,
            hierarchical_K=args.hierarchical_K, verbose=True)
    wall = time.time() - t0
    print(f"[diff] total wall: {wall:.2f}s", flush=True)

    full = torch.from_numpy(pos)
    costs = compute_proxy_cost(full, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlaps = int(costs.get("overlap_count", -1))
    print(f"[diff] raw proxy={proxy:.4f}  wl={float(costs['wirelength_cost']):.4f}  "
          f"den={float(costs['density_cost']):.4f}  cong={float(costs['congestion_cost']):.4f}  "
          f"overlaps={overlaps}", flush=True)

    out = {
        "benchmark": args.benchmark,
        "cost": proxy,
        "wirelength_cost": float(costs["wirelength_cost"]),
        "density_cost": float(costs["density_cost"]),
        "congestion_cost": float(costs["congestion_cost"]),
        "overlap_count": overlaps,
        "positions": pos[:n_hard].tolist(),
        "soft_positions": pos[n_hard:].tolist(),
        "wall_s": wall,
        "alpha": args.alpha,
        "weight_clip": args.weight_clip,
        "legalize_iters": args.legalize_iters,
        "info": info,
        "method": "diffusion_init_quadratic",
    }
    atomic_write_json(args.output, out)
    print(f"[diff] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
