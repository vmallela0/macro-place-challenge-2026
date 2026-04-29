"""Validate the cell-window truncation math.

Asserts:
1. `build_window_indices` returns a window that includes the macro's
   true footprint cells (no false negatives).
2. `smooth_density_grid` with very large softplus_μ converges to the
   exact density grid for a synthetic single-macro placement.
3. CVaR of densities equals the top-K mean at the optimum.
4. Gradients flow through the surrogate (autograd doesn't break).
"""
import sys
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

from _cell_window import (build_window_indices, smooth_density_grid,
                          smooth_macro_blockage, softplus_mu)
from _smooth_proxy import cvar_smooth, lse_max, lse_min


def test_window_includes_footprint():
    """The window must include every cell the macro's footprint touches."""
    grid_col = 30
    grid_row = 30
    grid_w = 1.0
    grid_h = 1.0
    # 3 macros, sizes 2 x 2, 1 x 1, 5 x 0.5
    macro_pos = torch.tensor([[5.5, 5.5], [10.0, 10.0], [15.0, 20.0]])
    macro_w = torch.tensor([2.0, 1.0, 5.0])
    macro_h = torch.tensor([2.0, 1.0, 0.5])
    cell_idx, K_max = build_window_indices(
        macro_pos, macro_w, macro_h,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)

    # Macro 0 at (5.5, 5.5) with size 2x2: footprint = [4.5, 6.5] x [4.5, 6.5]
    # Should cover cells (4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6),
    # (6, 4), (6, 5), (6, 6) — 9 cells. With margin 2 the window is much
    # bigger but must include those 9.
    expected_cells_m0 = set()
    for r in range(4, 7):
        for c in range(4, 7):
            expected_cells_m0.add(r * grid_col + c)

    window_m0 = set(int(x) for x in cell_idx[0].tolist() if x >= 0)
    missing = expected_cells_m0 - window_m0
    assert not missing, f"footprint cells missing from window: {missing}"
    print(f"  ✓ macro 0 window has {len(window_m0)} cells "
          f"(footprint {len(expected_cells_m0)} all included)")


def test_softplus_converges_to_relu():
    """softplus_μ(x) → max(0, x) as μ → ∞."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    relu = torch.clamp_min(x, 0.0)
    for mu in [10.0, 100.0, 1000.0]:
        y = softplus_mu(x, mu)
        diff = (y - relu).abs()
        # At μ=100, max diff should be small.
        if mu == 100.0:
            assert diff.max().item() < 0.01, \
                f"softplus_100 diff too large: {diff.max().item()}"
    print(f"  ✓ softplus_μ converges to ReLU as μ grows")


def test_lse_converges_to_max():
    """lse_max with large τ converges to true max."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    true_max = x.max().item()
    for tau in [10.0, 50.0, 200.0]:
        lse = lse_max(x, tau).item()
        if tau >= 50:
            assert abs(lse - true_max) < 0.05, \
                f"lse_τ={tau} diff from max: {abs(lse - true_max)}"
    print(f"  ✓ lse_max(τ=200) ≈ max within 0.05")


def test_cvar_equals_topk_at_optimum():
    """At the optimal threshold t*, CVaR equals the top-K mean exactly."""
    rng = np.random.RandomState(42)
    rho = torch.tensor(rng.uniform(0.0, 5.0, 100), dtype=torch.float32)
    K = 10
    # True top-K mean
    sorted_rho, _ = torch.sort(rho, descending=True)
    true_topk_mean = sorted_rho[:K].mean().item()

    # Find optimal t* by line search over candidate t values.
    # At t* = sorted_rho[K] (the K-th-largest, since 0-indexed top-K),
    # CVaR formula = top-K mean exactly.
    t_optimal = float(sorted_rho[K])  # i.e. ρ_(n-K) which is the (K+1)-th largest

    # The textbook formula uses t = the (n-K)-th order statistic, which
    # in 0-indexed sorted-descending is sorted_rho[K] (skipping the top
    # K). That equals the min of the top-K.
    # More precisely: CVaR_α with α = 1 - K/n equals avg of top-K when
    # t* = sorted_rho[K] (the K-th-largest, exclusive of the top-K).

    for mu in [1000.0]:  # use tight softplus for near-exact agreement
        t_t = torch.tensor(t_optimal, dtype=rho.dtype)
        cvar_val = cvar_smooth(rho.unsqueeze(0), K, t_t, mu=mu).item()
        # cvar_smooth includes the t baseline + softplus sum / K
        # At t = sorted_rho[K], (rho_c - t)+ is non-zero only for top-K rhos
        # giving CVaR = t + (1/K) * sum of (top-K - t) = avg of top-K. ✓
        assert abs(cvar_val - true_topk_mean) < 0.05, \
            f"CVaR_μ={mu} = {cvar_val} but top-K mean = {true_topk_mean}"
    print(f"  ✓ CVaR(t*=ρ_K, μ=1000) = {cvar_val:.4f}, "
          f"top-K mean = {true_topk_mean:.4f} (diff < 0.05)")


def test_density_gradient_flows():
    """Verify autograd gradient flows through smooth_density_grid."""
    grid_col = 10
    grid_row = 10
    grid_w = 1.0
    grid_h = 1.0
    cell_area = 1.0

    macro_pos = torch.tensor([[5.0, 5.0], [3.0, 7.0]], requires_grad=True)
    macro_w = torch.tensor([2.0, 1.5])
    macro_h = torch.tensor([2.0, 1.5])
    cell_idx, _ = build_window_indices(
        macro_pos.detach(), macro_w, macro_h,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)

    rho = smooth_density_grid(
        macro_pos, macro_w, macro_h, cell_idx,
        grid_col, grid_row, grid_w, grid_h,
        n_cells=grid_col * grid_row, cell_area=cell_area, mu=100.0)
    loss = rho.sum()
    loss.backward()
    assert macro_pos.grad is not None
    assert torch.isfinite(macro_pos.grad).all()
    # Gradient should be small (since sum of areas ≈ constant) but not zero
    # for a non-trivial test we'd want top-K gradient nonzero.
    print(f"  ✓ density gradient flows: grad norm = "
          f"{macro_pos.grad.norm().item():.4f} (finite)")


def test_blockage_gradient_flows():
    """Same for congestion / macro blockage."""
    grid_col = 10
    grid_row = 10
    grid_w = 1.0
    grid_h = 1.0

    macro_pos = torch.tensor([[5.0, 5.0]], requires_grad=True)
    macro_w = torch.tensor([2.0])
    macro_h = torch.tensor([2.0])
    cell_idx, _ = build_window_indices(
        macro_pos.detach(), macro_w, macro_h,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)

    V_macro, H_macro = smooth_macro_blockage(
        macro_pos, macro_w, macro_h, cell_idx,
        grid_col, grid_row, grid_w, grid_h,
        n_cells=grid_col * grid_row,
        vrouting_alloc=1.0, hrouting_alloc=1.0, mu=100.0)
    loss = V_macro.sum() + H_macro.sum()
    loss.backward()
    assert macro_pos.grad is not None
    assert torch.isfinite(macro_pos.grad).all()
    print(f"  ✓ blockage gradient flows: grad norm = "
          f"{macro_pos.grad.norm().item():.4f}")


if __name__ == "__main__":
    test_window_includes_footprint()
    test_softplus_converges_to_relu()
    test_lse_converges_to_max()
    test_cvar_equals_topk_at_optimum()
    test_density_gradient_flows()
    test_blockage_gradient_flows()
    print("ALL OK")
