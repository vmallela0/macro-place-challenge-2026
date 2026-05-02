"""Windowed legalisation: short-range push-apart constrained to within
`radius` of the moved macros.

This is the retraction operator for Riemannian descent on the
no-overlap manifold. A full re-legalize would destroy the gradient
signal (the new positions would be totally different from x_k - η g_T);
we only resolve overlaps that are actually local.

Algorithm
---------
1. Identify the set S of macros within radius of any macro that moved
   in this step (S includes the movers themselves).
2. Iteratively resolve pairwise overlaps in S using a constant-rate
   damped push (Chen-Sherwani-Lou 2008-style sequential push). Hard
   macros never move (they are constraints).
3. Stop when no overlaps remain in S OR after `max_iters` (caller
   re-runs with smaller step on failure).

Cost: O(|S|² + |S| · max_iters · 1) per call. For our scale
(|S| ≤ 50 typical, max_iters ≤ 30), trivial.

Cross-platform
--------------
Pure NumPy. No torch, no CUDA, no MPS. Identical results across
platforms because operations are float64 deterministic.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def overlap_pairs(
    pos: NDArray, w: NDArray, h: NDArray, indices: NDArray | None = None
) -> list[tuple[int, int, float, float]]:
    """Return list of (i, j, ox, oy) overlapping macro pairs.

    Pos is (n, 2) macro centers; w/h are sizes. ox/oy = overlap extent
    in each axis (positive when overlapping).

    If `indices` given, only pairs within that index subset are checked.
    """
    if indices is None:
        idx = np.arange(pos.shape[0])
    else:
        idx = np.asarray(indices)
    out: list[tuple[int, int, float, float]] = []
    for ii in range(len(idx)):
        i = int(idx[ii])
        for jj in range(ii + 1, len(idx)):
            j = int(idx[jj])
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            ox = (w[i] + w[j]) / 2.0 - dx
            oy = (h[i] + h[j]) / 2.0 - dy
            if ox > 1e-9 and oy > 1e-9:
                out.append((i, j, float(ox), float(oy)))
    return out


def short_pushapart(
    pos_old: NDArray,
    pos_new: NDArray,
    w: NDArray,
    h: NDArray,
    *,
    n_hard: int,
    radius: float,
    canvas_w: float,
    canvas_h: float,
    damping: float = 0.5,
    max_iters: int = 30,
) -> tuple[NDArray, dict]:
    """Short-range push-apart: only resolve overlaps within `radius` of
    macros that moved between pos_old and pos_new.

    Hard macros (indices [0, n_hard)) never move.

    Returns (pos_legal, info) where info has keys:
        moved_macros : indices considered (within radius)
        n_iters      : iterations used
        n_overlaps   : remaining overlaps after stop (0 = clean)
    """
    pos = pos_new.astype(np.float64).copy()
    delta = pos_new - pos_old
    moved_mask = (np.linalg.norm(delta, axis=1) > 1e-12)

    # Macros within `radius` of any mover (Euclidean center distance).
    movers = np.where(moved_mask)[0]
    if movers.size == 0:
        return pos, {"moved_macros": np.array([], dtype=np.int64),
                     "n_iters": 0, "n_overlaps": 0}
    dists = np.linalg.norm(
        pos_new[None, :, :] - pos_new[movers, None, :], axis=2
    )    # (n_movers, n_total)
    near = (dists.min(axis=0) <= radius)
    indices = np.where(near)[0]

    n_iters = 0
    for it in range(max_iters):
        pairs = overlap_pairs(pos, w, h, indices)
        if not pairs:
            n_iters = it
            break
        # Resolve each pair: push apart along the shorter overlap axis.
        for i, j, ox, oy in pairs:
            push = damping * (ox if ox < oy else oy)
            axis = 0 if ox < oy else 1
            # Direction: from j to i (i moves +, j moves -)
            sign = 1.0 if pos[i, axis] >= pos[j, axis] else -1.0
            i_movable = i >= n_hard
            j_movable = j >= n_hard
            if i_movable and j_movable:
                pos[i, axis] += 0.5 * sign * push
                pos[j, axis] -= 0.5 * sign * push
            elif i_movable:
                pos[i, axis] += sign * push
            elif j_movable:
                pos[j, axis] -= sign * push
            # else: both hard, can't fix this overlap here
        # Clip to canvas (centers stay within [size/2, canvas - size/2]).
        pos[indices, 0] = np.clip(pos[indices, 0],
                                    w[indices] / 2.0,
                                    canvas_w - w[indices] / 2.0)
        pos[indices, 1] = np.clip(pos[indices, 1],
                                    h[indices] / 2.0,
                                    canvas_h - h[indices] / 2.0)
        n_iters = it + 1

    remaining = overlap_pairs(pos, w, h, indices)
    return pos, {
        "moved_macros": indices,
        "n_iters": n_iters,
        "n_overlaps": len(remaining),
    }
