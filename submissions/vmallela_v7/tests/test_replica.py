"""Smoke test for B2: replica_diverse_select.

Math claim: greedy farthest-point selection picks subset maximizing
the minimum pairwise distance.

Verification: place 5 candidate placements on a 1D line at positions
(0, 1, 2, 5, 10) — clearly the most diverse subset of 3 is {0, 5, 10}
(min-pairwise = 5), NOT {0, 1, 2} or {2, 5, 10}.

After greedy farthest-point (seed = best U, here taken as first):
  start = pos 0
  next: argmax (distance to {0}) = 10  → subset {0, 10}
  next: argmax (min-dist to {0, 10}) = 5 (dist 5 each) — vs 1 (dist 1, 9), vs 2 (dist 2, 8)
        → subset {0, 5, 10} ✓
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
from _subspace_hmc import replica_diverse_select


def test_farthest_point_picks_spread():
    # Each candidate is a 1-macro placement at (x, 0).
    base = np.array([[0.0, 0.0]])
    cand_positions = [0.0, 1.0, 2.0, 5.0, 10.0]
    candidates = [
        (f"c{i}", np.array([[x, 0.0]])) for i, x in enumerate(cand_positions)
    ]
    # No diagnostics → seed = first.
    selected, diag = replica_diverse_select(
        candidates, base_pos=base, n_select=3, candidate_diagnostics=None)
    labels = [c[0] for c in selected]
    print(f"  selected labels: {labels}")
    print(f"  subset min pairwise: {diag['subset_pairwise_min_microns']:.3f} (expect 5.0)")
    # Verify that {c0, c3, c4} (positions 0, 5, 10) is what we get.
    sel_xs = sorted([float(c[1][0, 0]) for c in selected])
    assert sel_xs == [0.0, 5.0, 10.0], f"got {sel_xs}"


def test_pass_through_if_already_small():
    """If n_select >= n_cand, return all."""
    base = np.array([[0.0, 0.0]])
    candidates = [
        (f"c{i}", np.array([[float(i), 0.0]])) for i in range(3)
    ]
    sel, diag = replica_diverse_select(
        candidates, base, n_select=5, candidate_diagnostics=None)
    assert len(sel) == 3
    assert "warn" in diag


if __name__ == "__main__":
    test_farthest_point_picks_spread()
    test_pass_through_if_already_small()
    print("REPLICA TEST PASSED")
