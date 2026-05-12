"""Smoke test for B5: Diffusion Monte Carlo walker module.

Math claims:
  1. Walker population stays roughly bounded under E_T adaptation.
  2. After evolution on a quadratic well, walkers concentrate near min.
  3. No-walker collapse: at least 1 walker survives.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
import torch
from _dmc_walker import diffusion_monte_carlo_candidates


def test_dmc_concentrates_on_quadratic_well():
    """U(x) = ½ x^T x. Walkers initialized around x_0 = 0 with jitter
    should stay near 0 throughout evolution."""
    n_total = 1
    x0 = torch.zeros(n_total, 2)

    def U(x_tensor):
        return 0.5 * (x_tensor[0, 0] ** 2 + x_tensor[0, 1] ** 2)

    cands, diag = diffusion_monte_carlo_candidates(
        x0, U,
        n_walkers=16, n_steps=20, tau=0.3, beta=1.0,
        init_jitter=5.0,
        canvas_w=50.0, canvas_h=50.0,
        n_hard=0,
        step_cap_microns=10.0,
        seed=0, verbose=False)
    # The lowest-U walker should be very close to origin (the well minimum).
    if len(cands) == 0:
        # If all walkers died, the test fails. But check diag for collapse warning.
        print(f"  warning: {diag.get('warn')}")
    assert len(cands) > 0, "DMC produced no candidates"
    best = cands[0]
    norm = float(np.linalg.norm(best[1]))
    print(f"  DMC quadratic well: {len(cands)} cands, best ||x|| = {norm:.4f}")
    assert norm < 5.0, f"DMC best walker not near min: ||x||={norm}"


def test_dmc_population_does_not_explode():
    """U(x) = ½ x², make sure walker count stays bounded."""
    n_total = 1
    x0 = torch.zeros(n_total, 2)

    def U(x_tensor):
        return 0.5 * (x_tensor[0, 0] ** 2 + x_tensor[0, 1] ** 2)

    cands, diag = diffusion_monte_carlo_candidates(
        x0, U,
        n_walkers=16, n_steps=30, tau=0.2, beta=1.0,
        init_jitter=2.0,
        canvas_w=50.0, canvas_h=50.0,
        n_hard=0,
        step_cap_microns=5.0,
        seed=1, verbose=False)
    max_pop = max(diag["history_pop"])
    final_pop = diag["history_pop"][-1]
    print(f"  population history: max={max_pop}, final={final_pop}")
    assert max_pop < 4 * 16 + 5, f"population exploded: max={max_pop}"


if __name__ == "__main__":
    test_dmc_concentrates_on_quadratic_well()
    test_dmc_population_does_not_explode()
    print("DMC TEST PASSED")
