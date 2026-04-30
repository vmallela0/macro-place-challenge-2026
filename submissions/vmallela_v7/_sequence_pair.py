"""Sequence-pair representation of hard-macro topology.

Reference: Murata, Fujiyoshi, Nakatake, Kajitani 1996, "VLSI Module
Placement Based on Rectangle-Packing by the Sequence-Pair." IEEE TCAD.

A sequence pair (α, β) is two permutations of n block indices that
encodes the topological relationships between rectangular blocks.
Decoding (α, β) into (x, y) coordinates produces a non-overlapping
packing — guaranteed by construction. This is a *discrete* representation
of placement topology: a single adjacent swap in α or β changes the
relative order of two blocks (e.g., flips "A is left of B" into
"A is below B"), which spatial Gaussian noise cannot reliably do.

Math
====

Given α = (a₁, ..., aₙ) and β = (b₁, ..., bₙ), define:
    pos_α(i) = position of block i in α
    pos_β(i) = position of block i in β

Constraint relations:
    block i is LEFT of block j   iff  pos_α(i) < pos_α(j) AND pos_β(i) < pos_β(j)
    block i is BELOW block j     iff  pos_α(i) < pos_α(j) AND pos_β(i) > pos_β(j)
    block i is RIGHT of block j  iff  pos_α(i) > pos_α(j) AND pos_β(i) > pos_β(j)
    block i is ABOVE block j     iff  pos_α(i) > pos_α(j) AND pos_β(i) < pos_β(j)

Build constraint DAGs:
    G_h: edge i→j with weight w_i  iff i is LEFT of j
    G_v: edge i→j with weight h_i  iff i is BELOW j

Decode by longest-path on each DAG:
    x_j = max over predecessors i of (x_i + w_i)
    y_j = max over predecessors i of (y_i + h_i)
    (with x_source = y_source = 0)

The longest-path DAG decoding produces the *compact* placement (no
wasted whitespace beyond constraint requirements). Different (α, β)
yield different placements; the space of all (α, β) covers the space
of all rectangular packings.

Encoding (positions → (α, β)) uses a Murata-style heuristic:
    α: sort blocks by (x + y) ascending
    β: sort blocks by (y - x) ascending
This is one valid SP for the given placement; it doesn't roundtrip
exactly (decoded SP coords will be compacted to canvas origin, not at
original positions) but the *topology* it captures is consistent with
the input placement.

For our use case (basin-hop perturbation), exact roundtrip isn't
required — we use the encoded SP as a starting topology, swap k pairs
to perturb the topology, and decode to a new packing.

Complexity
==========
- encode_sp: O(n log n)
- sp_swap: O(k)
- decode_sp: O(n²) for the pair-relation enumeration; O(n log n)
  is achievable with interval trees but n_hard is typically small
  (≤ 300) so O(n²) is fine.
"""
from __future__ import annotations
import math
import random
from typing import Sequence, Tuple, List
import numpy as np


def encode_sp(positions: np.ndarray) -> Tuple[List[int], List[int]]:
    """Heuristic encoding: (α, β) consistent with current positions.

    Parameters
    ----------
    positions : (n, 2) array of (x, y) lower-left corners. Widths/heights
        are not needed for encoding — only the centroid-like ordering
        in (x+y) and (y-x) directions.

    Returns
    -------
    (alpha, beta) : two permutations of [0, n).
        alpha[i] = j means block j is the i-th in the α sequence.
    """
    n = positions.shape[0]
    sums = positions[:, 0] + positions[:, 1]
    diffs = positions[:, 1] - positions[:, 0]
    # argsort with stable=True: ties broken by index
    alpha = list(np.argsort(sums, kind="stable").tolist())
    beta = list(np.argsort(diffs, kind="stable").tolist())
    return alpha, beta


def sp_swap(
    alpha: Sequence[int],
    beta: Sequence[int],
    n_swaps: int = 1,
    *,
    rng: random.Random | None = None,
    swap_target: str = "random",
) -> Tuple[List[int], List[int]]:
    """Apply n_swaps random adjacent swaps to α and/or β.

    Parameters
    ----------
    alpha, beta : current sequence pair.
    n_swaps : number of single-adjacent swaps to apply (each swap is
        independent; can be in α or β based on swap_target).
    rng : random.Random or None.
    swap_target : "random", "alpha", or "beta". "random" picks each
        swap independently from {α, β} with prob 0.5.

    Returns
    -------
    (alpha', beta') : new sequences after k adjacent swaps. Caller
        decides whether to also encode the original placement and
        merge — we return raw swapped sequences.
    """
    if rng is None:
        rng = random.Random()
    a = list(alpha)
    b = list(beta)
    n = len(a)
    if n < 2:
        return a, b
    for _ in range(n_swaps):
        if swap_target == "random":
            seq = a if rng.random() < 0.5 else b
        elif swap_target == "alpha":
            seq = a
        elif swap_target == "beta":
            seq = b
        else:
            raise ValueError(f"swap_target={swap_target!r}")
        i = rng.randint(0, n - 2)
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return a, b


def decode_sp(
    alpha: Sequence[int],
    beta: Sequence[int],
    widths: np.ndarray,
    heights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode (α, β) to compact non-overlapping placement.

    Returns (x, y) arrays of lower-left corners (length n). x[i] and
    y[i] are computed via longest-path DP on the constraint DAGs:
        G_h: edge i→j iff i is LEFT of j (i.e., pos_α(i)<pos_α(j) AND pos_β(i)<pos_β(j))
        G_v: edge i→j iff i is BELOW of j (i.e., pos_α(i)<pos_α(j) AND pos_β(i)>pos_β(j))

    Processing in α order is a valid topological order for both DAGs:
    every edge i→j in either G_h or G_v has pos_α(i) < pos_α(j),
    so processing in α order ensures predecessors are computed first.
    """
    n = len(alpha)
    pos_a = [0] * n
    pos_b = [0] * n
    for k in range(n):
        pos_a[alpha[k]] = k
        pos_b[beta[k]] = k

    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)

    for k in range(n):
        i = alpha[k]
        best_x = 0.0
        best_y = 0.0
        for j in range(n):
            if j == i:
                continue
            if pos_a[j] < pos_a[i] and pos_b[j] < pos_b[i]:
                # j is LEFT of i  →  x_i ≥ x_j + w_j
                cand = x[j] + widths[j]
                if cand > best_x:
                    best_x = cand
            elif pos_a[j] < pos_a[i] and pos_b[j] > pos_b[i]:
                # j is BELOW i  →  y_i ≥ y_j + h_j
                cand = y[j] + heights[j]
                if cand > best_y:
                    best_y = cand
        x[i] = best_x
        y[i] = best_y
    return x, y


def fit_to_canvas(
    x: np.ndarray, y: np.ndarray,
    widths: np.ndarray, heights: np.ndarray,
    canvas_w: float, canvas_h: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale a decoded (x, y) packing to fit inside a canvas.

    SP-decode produces a compact packing whose bounding box is
    (max(x_i + w_i), max(y_i + h_i)). If this exceeds the canvas, we
    isotropic-scale to fit, then center.

    Returns (x', y', scale). scale=1.0 means the packing fits as-is.
    """
    bbox_w = float(np.max(x + widths))
    bbox_h = float(np.max(y + heights))
    sx = canvas_w / bbox_w if bbox_w > 0 else 1.0
    sy = canvas_h / bbox_h if bbox_h > 0 else 1.0
    s = min(1.0, sx, sy)
    if s < 1.0:
        x = x * s
        y = y * s
        widths_s = widths * s
        heights_s = heights * s
    else:
        widths_s = widths
        heights_s = heights
        s = 1.0
    # Center inside canvas
    bbox_w2 = float(np.max(x + widths_s))
    bbox_h2 = float(np.max(y + heights_s))
    pad_x = (canvas_w - bbox_w2) / 2.0
    pad_y = (canvas_h - bbox_h2) / 2.0
    return x + pad_x, y + pad_y, s


def check_no_overlap(
    x: np.ndarray, y: np.ndarray,
    widths: np.ndarray, heights: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[int, float]:
    """Sanity check on decoded placement. Returns (count, total_area).
    If decode is correct, count == 0.
    O(n²); fine for small n_hard."""
    n = len(x)
    count = 0
    area = 0.0
    for i in range(n):
        xi1, yi1 = x[i], y[i]
        xi2, yi2 = xi1 + widths[i], yi1 + heights[i]
        for j in range(i + 1, n):
            xj1, yj1 = x[j], y[j]
            xj2, yj2 = xj1 + widths[j], yj1 + heights[j]
            ow = min(xi2, xj2) - max(xi1, xj1)
            oh = min(yi2, yj2) - max(yi1, yj1)
            if ow > eps and oh > eps:
                count += 1
                area += ow * oh
    return count, area
