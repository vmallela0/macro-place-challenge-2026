"""Sanity tests for _rudy_smooth.smooth_rudy_routing.

Validate that the differentiable RUDY surrogate:
  1. Produces finite outputs on a 3-macro / 5-net toy.
  2. Gradient flows from V/H demand back to macro_pos.
  3. Behaves correctly in physical limits:
     - When all pins of a net coincide, contributions to cells far away are ~0.
     - When a net spans the grid, V_demand sums approx to (net_weight × bbox_x).
  4. Matches the expected RUDY scaling for a 2-pin net spanning a known bbox.
"""
from __future__ import annotations

import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import torch

from _rudy_smooth import (
    lse_bbox_per_net,
    build_net_window_indices,
    build_net_window_indices_sparse,
    smooth_rudy_routing,
    smooth_rudy_routing_sparse,
)


def test_sparse_matches_dense():
    """Sparse and dense RUDY should give bit-equal output for valid window cells.

    They differ on cells outside any net's window: dense pads with -1 (no
    contribution); sparse omits them (same zero contribution). For the
    'valid window' cells both populate, the (V, H) tensors should match.
    """
    (macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
     net_weight, n_nets, grid_col, grid_row, grid_w, grid_h, n_cells) = make_toy()
    is_port = (pin_macro < 0)
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)

    cell_idx_dense, _ = build_net_window_indices(
        pin_x, pin_y, pin_to_net, n_nets,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)
    V_d, H_d = smooth_rudy_routing(
        macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
        net_weight, n_nets, cell_idx_dense,
        grid_col, grid_row, grid_w, grid_h, n_cells)

    pair_net, pair_cell, n_pairs, _ = build_net_window_indices_sparse(
        pin_x, pin_y, pin_to_net, n_nets,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)
    V_s, H_s = smooth_rudy_routing_sparse(
        pin_x, pin_y, pin_to_net, net_weight, n_nets,
        pair_net, pair_cell,
        grid_col, grid_row, grid_w, grid_h, n_cells)
    # The dense and sparse versions use the SAME windows (margin=2 for
    # both), so they should produce identical contributions on every cell.
    rel = (V_d - V_s).abs().max() / max(float(V_d.abs().max()), 1e-9)
    assert rel < 1e-5, f"V mismatch rel={float(rel):.2e}"
    rel = (H_d - H_s).abs().max() / max(float(H_d.abs().max()), 1e-9)
    assert rel < 1e-5, f"H mismatch rel={float(rel):.2e}"
    print(f"  sparse==dense PASS: n_pairs={n_pairs} V.max rel.err={float(rel):.2e}")


def make_toy():
    """3 macros + 2 ports, 3 nets, on a 8×8 grid (cell size 10×10)."""
    torch.manual_seed(0)
    n_total = 3
    macro_pos = torch.tensor([
        [10.0, 10.0],   # macro 0 in lower-left
        [50.0, 50.0],   # macro 1 in middle
        [70.0, 70.0],   # macro 2 in upper-right
    ], requires_grad=True)
    # Pin layout: 6 macro pins (2/macro) + 2 ports.
    pin_macro = torch.tensor([0, 0, 1, 1, 2, 2, -1, -1], dtype=torch.long)
    pin_xoff = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 75.0])
    pin_yoff = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 75.0])
    # 3 nets:
    #   net 0 = pins 0 (macro 0), 6 (port at (5,5))                  — small bbox
    #   net 1 = pins 1 (macro 0), 2 (macro 1), 7 (port at (75,75))   — spans canvas
    #   net 2 = pins 3 (macro 1), 4 (macro 2)                        — middle/upper
    pin_to_net = torch.tensor([0, 1, 1, 2, 2, -1, 0, 1], dtype=torch.long)
    # Note: pin 5 is unused (pin_to_net = -1 → no scatter); we'll
    # just skip it via filter.
    valid = (pin_to_net >= 0)
    pin_macro_f = pin_macro[valid]
    pin_xoff_f = pin_xoff[valid]
    pin_yoff_f = pin_yoff[valid]
    pin_to_net_f = pin_to_net[valid]
    n_nets = 3
    net_weight = torch.tensor([1.0, 1.0, 2.0])
    grid_col, grid_row = 8, 8
    grid_w, grid_h = 10.0, 10.0
    n_cells = grid_col * grid_row
    return (macro_pos, pin_macro_f, pin_xoff_f, pin_yoff_f, pin_to_net_f,
            net_weight, n_nets, grid_col, grid_row, grid_w, grid_h, n_cells)


def test_finite_and_gradflow():
    (macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
     net_weight, n_nets, grid_col, grid_row, grid_w, grid_h, n_cells) = make_toy()
    is_port = (pin_macro < 0)
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)
    net_cell_idx, K_max = build_net_window_indices(
        pin_x, pin_y, pin_to_net, n_nets,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)
    assert K_max > 0
    V, H = smooth_rudy_routing(
        macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
        net_weight, n_nets, net_cell_idx,
        grid_col, grid_row, grid_w, grid_h, n_cells)
    assert torch.isfinite(V).all(), "V_demand has non-finite entries"
    assert torch.isfinite(H).all(), "H_demand has non-finite entries"
    assert (V >= 0).all(), "V_demand should be non-negative"
    assert (H >= 0).all(), "H_demand should be non-negative"
    loss = V.sum() + H.sum()
    loss.backward()
    assert macro_pos.grad is not None
    assert torch.isfinite(macro_pos.grad).all()
    print(f"  finite/gradflow PASS: V.sum={V.sum():.3f} H.sum={H.sum():.3f} "
          f"||∇x|| = {macro_pos.grad.norm():.3e}")


def test_far_cell_negligible():
    """A net entirely in the lower-left should not contribute to upper-right cells."""
    (macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
     net_weight, n_nets, grid_col, grid_row, grid_w, grid_h, n_cells) = make_toy()
    # Restrict to net 0 only (lower-left), so far cells should be ~0.
    is_port = (pin_macro < 0)
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)
    # Filter pins to net 0 only.
    mask = (pin_to_net == 0)
    net_cell_idx, _ = build_net_window_indices(
        pin_x[mask], pin_y[mask], pin_to_net[mask], 1,
        grid_col, grid_row, grid_w, grid_h, margin_cells=1)
    V, H = smooth_rudy_routing(
        macro_pos, pin_macro[mask], pin_xoff[mask], pin_yoff[mask],
        pin_to_net[mask], net_weight[:1], 1, net_cell_idx,
        grid_col, grid_row, grid_w, grid_h, n_cells)
    # Upper-right cell index (col=7, row=7)
    ur_cell = 7 * grid_col + 7
    assert V[ur_cell].item() < 1e-3, f"V[ur]={V[ur_cell].item()} should be ~0"
    assert H[ur_cell].item() < 1e-3, f"H[ur]={H[ur_cell].item()} should be ~0"
    print(f"  far-cell PASS: V[ur]={V[ur_cell].item():.2e} "
          f"H[ur]={H[ur_cell].item():.2e}")


def test_rudy_scaling_2pin():
    """For a single 2-pin net at fixed positions, sum_c V_demand_c · cell_area
    should approximately equal net_weight · bbox_x (the net's horizontal
    wirelength), since RUDY allocates exactly Δx wires distributed uniformly.
    """
    grid_col, grid_row = 8, 8
    grid_w, grid_h = 10.0, 10.0
    cell_area = grid_w * grid_h
    n_cells = grid_col * grid_row

    # Single net, 2 pins at (10, 10) and (50, 50) → bbox 40×40.
    macro_pos = torch.tensor([[10.0, 10.0], [50.0, 50.0]], requires_grad=True)
    pin_macro = torch.tensor([0, 1], dtype=torch.long)
    pin_xoff = torch.tensor([0.0, 0.0])
    pin_yoff = torch.tensor([0.0, 0.0])
    pin_to_net = torch.tensor([0, 0], dtype=torch.long)
    n_nets = 1
    net_weight = torch.tensor([1.0])

    is_port = (pin_macro < 0)
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)
    net_cell_idx, _ = build_net_window_indices(
        pin_x, pin_y, pin_to_net, n_nets,
        grid_col, grid_row, grid_w, grid_h, margin_cells=2)
    V, H = smooth_rudy_routing(
        macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net,
        net_weight, n_nets, net_cell_idx,
        grid_col, grid_row, grid_w, grid_h, n_cells)
    # V_demand_c has units of [length], not [length/area]:
    #     V_c  +=  w_n · overlap_area(cell, bbox) / Δy_n
    # Summing over cells gives w_n · A_bbox / Δy = w_n · Δx_n.
    # For our 40×40 bbox at w=1: sum_c V_c ≈ 40 (small bias from eps_bbox).
    expected = float(net_weight.item()) * 40.0
    got = float(V.sum().item())
    rel_err = abs(got - expected) / max(expected, 1e-9)
    # softplus_μ smoothing inflates the bbox by ~1/μ on each side; with μ=100
    # that's ~0.01 micron — negligible. LSE bbox with τ=50 inflates extents by
    # ~log(K)/τ ≈ 0.014 micron for K=2 pins — also negligible. We allow ~5 %
    # tolerance to cover joint smoothing artefacts plus the eps_bbox term.
    assert rel_err < 0.05, (
        f"sum(V)·cell_area = {got:.3f}, expected ≈ {expected:.3f}, "
        f"rel_err = {rel_err:.3f}")
    print(f"  RUDY scaling PASS: ∫V dA = {got:.3f} (expected {expected:.3f}, "
          f"rel_err {rel_err:.3f})")


if __name__ == "__main__":
    test_finite_and_gradflow()
    test_far_cell_negligible()
    test_rudy_scaling_2pin()
    test_sparse_matches_dense()
    print("ALL TESTS PASSED")
