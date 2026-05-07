"""Spectral net criticality from Hessian eigenvectors.

Hypothesis: at a saddle of the smooth proxy, the negative-eigenvalue
eigenvectors of the Hessian point in directions where the cost drops.
Each eigvec is a 2N-dim signed displacement field over macros. The
*per-net energy* of an eigvec — variance of its displacement field
restricted to the macros connected by that net — measures how much that
net's bbox is being "stretched" along this escape direction.

Aggregating per-net energy across the negative-eigvalue eigvecs
(weighted by |λ|) gives a self-referential criticality score: the
optimizer's own landscape geometry tells us which nets are at the
bottleneck. This is distinct from STA-based or fanout-based
criticality — it emerges from the cost surface itself.

Use this score to amplify net_weight in the next Hessian iteration.
The next saddle escape is then biased toward shortening the bottleneck
nets — precisely the nets whose geometry the previous escape was
fighting against.
"""
from __future__ import annotations
import numpy as np


def eigvec_net_criticality(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    pin_macro: np.ndarray,
    net_starts: np.ndarray,
    *,
    n_total: int,
    only_negative: bool = True,
    eps_neg: float = -1e-6,
) -> np.ndarray:
    """Compute per-net spectral criticality from Hessian eigenpairs.

    Args:
        eigvals: shape (K,), eigenvalues from Lanczos.
        eigvecs: shape (2N, K), eigenvector basis (columns).
        pin_macro: shape (P,), macro index for each pin (-1 for ports).
        net_starts: shape (n_nets+1,), CSR start offsets for nets.
        n_total: number of macros (= N).
        only_negative: only use eigvecs with eigval < eps_neg.
        eps_neg: threshold for "negative enough to count".

    Returns:
        criticality: shape (n_nets,) float64, normalized to [0, 1].

    The criticality of net n is:

        crit[n] = Σ_{j: λ_j < eps} |λ_j| · var_n(v_j)

    where var_n(v_j) is the variance of the eigvec's macro
    displacements restricted to macros touching net n. Variance is
    used because HPWL responds to *spread* of pins, not their absolute
    position; a constant translation of a net leaves HPWL unchanged.
    """
    K = int(eigvals.shape[0])
    if K == 0 or eigvecs.size == 0:
        n_nets = int(net_starts.shape[0]) - 1
        return np.zeros(max(n_nets, 0), dtype=np.float64)
    n_nets = int(net_starts.shape[0]) - 1
    crit = np.zeros(n_nets, dtype=np.float64)

    # Mask to filter eigvecs by sign
    if only_negative:
        keep = eigvals < eps_neg
    else:
        keep = np.ones(K, dtype=bool)
    if not keep.any():
        return crit

    sel_eigvals = eigvals[keep]
    sel_eigvecs = eigvecs[:, keep]                    # (2N, K')
    K_neg = int(sel_eigvecs.shape[1])

    # Reshape each eigvec to (N, 2). For each coord (x, y) separately,
    # compute the variance across pins in a net — that variance is the
    # first-order rate at which the net's bbox grows under that eigvec
    # direction. Doing x and y separately (then summing) captures the
    # OPPOSITE-DIRECTION-STRETCH case (macros A and B moving in
    # opposite x at equal magnitude) — the most stressed case — which
    # the magnitude-squared variance would erroneously zero out.
    vj_xy = sel_eigvecs.reshape(n_total, 2, K_neg)    # (N, 2, K')
    n_pins = int(pin_macro.shape[0])
    pin_to_net = np.zeros(n_pins, dtype=np.int64)
    for nid in range(n_nets):
        s = int(net_starts[nid])
        e = int(net_starts[nid + 1])
        pin_to_net[s:e] = nid
    non_port = pin_macro >= 0

    abs_lams = np.abs(sel_eigvals)                     # (K',)
    for j in range(K_neg):
        v_x = vj_xy[:, 0, j]                            # (N,)
        v_y = vj_xy[:, 1, j]
        per_pin_x = np.zeros(n_pins, dtype=np.float64)
        per_pin_y = np.zeros(n_pins, dtype=np.float64)
        per_pin_x[non_port] = v_x[pin_macro[non_port]]
        per_pin_y[non_port] = v_y[pin_macro[non_port]]

        # Per-net variance via scatter (one pass for x, one for y).
        sum_x = np.zeros(n_nets, dtype=np.float64)
        sum_x2 = np.zeros(n_nets, dtype=np.float64)
        sum_y = np.zeros(n_nets, dtype=np.float64)
        sum_y2 = np.zeros(n_nets, dtype=np.float64)
        cnt = np.zeros(n_nets, dtype=np.float64)
        np.add.at(sum_x, pin_to_net, per_pin_x)
        np.add.at(sum_x2, pin_to_net, per_pin_x ** 2)
        np.add.at(sum_y, pin_to_net, per_pin_y)
        np.add.at(sum_y2, pin_to_net, per_pin_y ** 2)
        np.add.at(cnt, pin_to_net, 1.0)
        cnt_safe = np.maximum(cnt, 1.0)
        var_x = sum_x2 / cnt_safe - (sum_x / cnt_safe) ** 2
        var_y = sum_y2 / cnt_safe - (sum_y / cnt_safe) ** 2
        var = np.maximum(var_x, 0.0) + np.maximum(var_y, 0.0)

        crit += abs_lams[j] * var

    # Normalize to [0, 1] for stable downstream weighting
    cmax = float(crit.max())
    if cmax > 1e-12:
        crit /= cmax
    return crit


def apply_criticality_to_weights(
    base_weight: np.ndarray,
    criticality: np.ndarray,
    *,
    gain: float = 0.5,
) -> np.ndarray:
    """Multiplicative reweight: w_new = w_old · (1 + gain · criticality).

    With criticality ∈ [0, 1] and gain=0.5, the most critical net gets
    50% boost; uncritical nets unchanged.
    """
    if base_weight.shape[0] != criticality.shape[0]:
        return base_weight.copy()
    return base_weight * (1.0 + float(gain) * criticality)
