"""Sanity tests for _langevin_mala.mala_search.

Run on a tiny synthetic problem where the optimum is known.
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

from _langevin_mala import mala_search


def make_2d_quadratic():
    """Synthetic problem: minimize ||x - x*||² on a 4-macro / 1-pin-per-macro grid.

    Smooth proxy = ||x - x*||² (a quadratic bowl).
    Exact proxy = same (so the strict-improvement gate is honest).
    Optimum is at x*.
    """
    n_macros = 4
    x_star_np = np.array([
        [10.0, 10.0], [50.0, 50.0],
        [10.0, 50.0], [50.0, 10.0],
    ], dtype=np.float64)
    x_star = torch.tensor(x_star_np, dtype=torch.float32)
    x0 = torch.tensor([
        [30.0, 30.0], [30.0, 30.0],
        [30.0, 30.0], [30.0, 30.0],
    ], dtype=torch.float32)

    def smooth_proxy(x):
        return ((x - x_star) ** 2).sum()

    eval_count = [0]
    def exact_proxy_np(x_np):
        eval_count[0] += 1
        cost = float(((x_np - x_star_np) ** 2).sum())
        return cost, 0  # always 0 overlaps for this toy

    return x0, smooth_proxy, exact_proxy_np, x_star_np, eval_count


def test_mala_descends():
    x0, smooth, exact, x_star, evc = make_2d_quadratic()
    canvas_diag = 100.0
    best_pos, best_cost, diag = mala_search(
        x0, smooth, exact, canvas_diag,
        n_steps=200, step_size_frac=0.005, temp_init_frac=0.0005,
        temp_decay=0.99, soft_only=False, n_hard=0,
        seed=42, n_burn=0, verbose=False)
    init_cost, _ = exact(x0.cpu().numpy().astype(np.float64))
    assert best_cost < init_cost - 1e-3, (
        f"MALA failed to descend: best={best_cost} init={init_cost}")
    print(f"  toy quadratic PASS: init={init_cost:.3f} -> best={best_cost:.3f} "
          f"({diag['accepted']}/{diag['n_steps']} accepted; "
          f"{evc[0]} exact evals)")


def test_mala_no_regression_on_at_minimum():
    """If we start AT the minimum, MALA should not regress."""
    x0, smooth, exact, x_star, evc = make_2d_quadratic()
    x_at_min = torch.tensor(x_star, dtype=torch.float32)
    canvas_diag = 100.0
    best_pos, best_cost, diag = mala_search(
        x_at_min, smooth, exact, canvas_diag,
        n_steps=100, step_size_frac=0.005, temp_init_frac=0.0001,
        temp_decay=0.99, soft_only=False, n_hard=0,
        seed=42, n_burn=0, verbose=False)
    # We started at min (cost=0). The best_cost reported is the gate
    # baseline (=0). Even if random kicks find slightly negative numerical
    # noise, the gate prevents accepting positive deltas, so best_cost
    # cannot be worse than 0.
    assert best_cost <= 1e-6, (
        f"MALA regressed at minimum: best={best_cost} (expected ~0)")
    print(f"  at-minimum stability PASS: best={best_cost:.3e}, "
          f"accepted={diag['accepted']}")


if __name__ == "__main__":
    test_mala_descends()
    test_mala_no_regression_on_at_minimum()
    print("\nLANGEVIN MALA TESTS PASSED")
