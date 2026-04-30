"""Validate sequence-pair encode / swap / decode math.

Asserts:
1. decode_sp on an arbitrary (α, β) produces a non-overlapping packing.
2. encode_sp on a non-overlapping placement produces a valid SP whose
   decode yields a non-overlapping packing (not necessarily identical
   to the input — SP is a topology, not exact coords).
3. Single adjacent swaps change the decoded placement (i.e., the swap
   is not a no-op except for accidental cases).
4. Identical (α, β) decodes to a left-to-right strip arrangement
   (sanity check on the longest-path DP).
"""
import sys
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

from _sequence_pair import (encode_sp, sp_swap, decode_sp,
                              fit_to_canvas, check_no_overlap)


def test_decode_no_overlap():
    """Random (α, β) decode produces a non-overlapping packing."""
    rng = np.random.RandomState(42)
    for trial in range(10):
        n = rng.randint(5, 20)
        widths = rng.uniform(0.5, 2.0, n).astype(np.float64)
        heights = rng.uniform(0.5, 2.0, n).astype(np.float64)
        alpha = list(rng.permutation(n))
        beta = list(rng.permutation(n))
        x, y = decode_sp(alpha, beta, widths, heights)
        ov_count, ov_area = check_no_overlap(x, y, widths, heights)
        assert ov_count == 0, (
            f"trial {trial}: random SP decoded with {ov_count} overlaps "
            f"(total area {ov_area:.4e})\nα={alpha}\nβ={beta}")
        bbox_w = float(np.max(x + widths))
        bbox_h = float(np.max(y + heights))
        assert bbox_w > 0 and bbox_h > 0, "degenerate packing"
    print(f"  ✓ decode produces non-overlapping packing (10 random trials)")


def test_encode_then_decode():
    """encode→decode produces a non-overlapping packing whose topology
    is consistent with the input. Coords need not match input exactly."""
    rng = np.random.RandomState(0)
    n = 12
    widths = rng.uniform(0.5, 1.5, n).astype(np.float64)
    heights = rng.uniform(0.5, 1.5, n).astype(np.float64)
    # Build a random non-overlapping placement by stripe-packing
    positions = np.zeros((n, 2), dtype=np.float64)
    cur_x, cur_y = 0.0, 0.0
    row_h = 0.0
    for i in range(n):
        if cur_x + widths[i] > 5.0:
            cur_x = 0.0
            cur_y += row_h
            row_h = 0.0
        positions[i] = [cur_x, cur_y]
        cur_x += widths[i]
        row_h = max(row_h, heights[i])
    # Verify input is overlap-free
    ov_count_input, _ = check_no_overlap(
        positions[:, 0], positions[:, 1], widths, heights)
    assert ov_count_input == 0, "test input has overlaps"

    alpha, beta = encode_sp(positions)
    assert sorted(alpha) == list(range(n)), "alpha is not a permutation"
    assert sorted(beta) == list(range(n)), "beta is not a permutation"

    x_d, y_d = decode_sp(alpha, beta, widths, heights)
    ov_count, ov_area = check_no_overlap(x_d, y_d, widths, heights)
    assert ov_count == 0, (
        f"encode+decode produced {ov_count} overlaps "
        f"(area {ov_area:.4e})")
    print(f"  ✓ encode→decode round-trip preserves no-overlap (n={n})")


def test_swap_changes_placement():
    """A single adjacent swap changes the decoded placement."""
    rng = np.random.RandomState(7)
    rnd = random.Random(7)
    n = 10
    widths = rng.uniform(0.5, 1.5, n).astype(np.float64)
    heights = rng.uniform(0.5, 1.5, n).astype(np.float64)
    alpha = list(range(n))
    beta = list(range(n))
    x0, y0 = decode_sp(alpha, beta, widths, heights)

    n_changed = 0
    for trial in range(10):
        a2, b2 = sp_swap(alpha, beta, n_swaps=1, rng=rnd)
        x1, y1 = decode_sp(a2, b2, widths, heights)
        if not (np.allclose(x0, x1, atol=1e-9)
                and np.allclose(y0, y1, atol=1e-9)):
            n_changed += 1
    # Most swaps should change the placement
    assert n_changed >= 7, (
        f"only {n_changed}/10 single swaps changed the decoded placement "
        f"(expected ≥ 7; SP swap might be no-op)")
    print(f"  ✓ {n_changed}/10 random adjacent swaps changed the placement")


def test_identity_alpha_beta():
    """α = β = identity → blocks pack left-to-right in a single row.
    (Each block is LEFT of every block after it, since both pos_α and
    pos_β agree.)"""
    n = 5
    widths = np.array([1.0, 2.0, 0.5, 1.5, 1.0])
    heights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    alpha = list(range(n))
    beta = list(range(n))
    x, y = decode_sp(alpha, beta, widths, heights)
    expected_x = np.cumsum(np.concatenate([[0.0], widths[:-1]]))
    assert np.allclose(x, expected_x), (
        f"identity SP didn't strip-pack: x={x}, expected {expected_x}")
    assert np.allclose(y, np.zeros(n)), \
        f"identity SP shouldn't stack vertically: y={y}"
    print(f"  ✓ identity (α=β) packs left-to-right: x={x}")


def test_reverse_alpha():
    """α = reverse, β = identity → blocks stack VERTICALLY.
    (Each block is BELOW the next, since pos_α(i) > pos_α(i+1) but
    pos_β(i) < pos_β(i+1)... wait that says i is ABOVE i+1, i.e., i+1
    is BELOW i. Let's check.)"""
    # α = [4, 3, 2, 1, 0], β = [0, 1, 2, 3, 4]
    # For pair (i=0, j=1): pos_α(0)=4, pos_α(1)=3 → pos_α(0) > pos_α(1).
    #                       pos_β(0)=0, pos_β(1)=1 → pos_β(0) < pos_β(1).
    # So 0 is ABOVE 1 (block 0 is positioned higher). Equivalently 1 is BELOW 0.
    # So decode: 1 must be at y=0; 0 must be at y >= y[1] + h[1].
    # Sequence: each block i is above block i+1 → block 4 at bottom (y=0),
    # block 0 at top.
    n = 5
    widths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    heights = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    alpha = [4, 3, 2, 1, 0]
    beta = [0, 1, 2, 3, 4]
    x, y = decode_sp(alpha, beta, widths, heights)
    # Should stack at x=0
    assert np.allclose(x, np.zeros(n)), \
        f"reverse-α should stack at x=0; got x={x}"
    # Block 4 at y=0; block 3 at y=0.5; ... block 0 at y=2.0
    assert np.isclose(y[4], 0.0), f"block 4 not at bottom: y[4]={y[4]}"
    assert np.isclose(y[0], 2.0), f"block 0 not at top: y[0]={y[0]}"
    print(f"  ✓ reverse-α packs vertically: y[4]={y[4]:.1f}, y[0]={y[0]:.1f}")


if __name__ == "__main__":
    test_decode_no_overlap()
    test_encode_then_decode()
    test_swap_changes_placement()
    test_identity_alpha_beta()
    test_reverse_alpha()
    print("ALL OK")
