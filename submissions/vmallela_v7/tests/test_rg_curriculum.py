"""Smoke test for RG curriculum module.

Math claims being verified:
  1. γ_n = exp(-L_n² / (2σ²)) correctly downweights long nets.
     At L = σ: γ = exp(-0.5) ≈ 0.607.
     At L = 3σ: γ = exp(-4.5) ≈ 0.011.
     At L = 0:   γ = 1.
  2. The schedule σ(t) goes monotonically from σ_0·canvas to σ_∞·canvas.
  3. Per-net bbox diagonal computation matches simple manual calc.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import math
import numpy as np
import torch
from _rg_curriculum import (
    apply_rg_curriculum_weights, schedule_sigma,
    compute_per_net_bbox_diag,
)


def test_curriculum_weight_form():
    """γ_n = exp(-L²/(2σ²)). Hand-check three L values."""
    base = torch.ones(3)
    L = torch.tensor([0.0, 100.0, 300.0])
    sigma = 100.0
    w = apply_rg_curriculum_weights(base, L, sigma)
    expected = torch.tensor([1.0, math.exp(-0.5), math.exp(-4.5)])
    print(f"  γ values: {w.tolist()}")
    print(f"  expected: {expected.tolist()}")
    assert torch.allclose(w, expected, atol=1e-6), \
        f"γ form mismatch: {w} vs {expected}"


def test_schedule_monotone():
    """σ(t) increases monotonically; endpoints match prescription."""
    canvas_diag = 500.0
    sigma_0 = 0.05    # 0.05 · canvas
    sigma_inf = 10.0   # 10 · canvas
    sigmas = [schedule_sigma(t / 10.0, sigma_0, canvas_diag, sigma_inf)
              for t in range(11)]
    print(f"  σ(0)={sigmas[0]:.3f} (expect {sigma_0*canvas_diag:.3f})")
    print(f"  σ(1)={sigmas[-1]:.3f} (expect {sigma_inf*canvas_diag:.3f})")
    # Endpoints.
    assert abs(sigmas[0] - sigma_0 * canvas_diag) < 1e-6
    assert abs(sigmas[-1] - sigma_inf * canvas_diag) < 1e-6
    # Monotone increasing.
    for i in range(len(sigmas) - 1):
        assert sigmas[i] <= sigmas[i + 1] + 1e-9, \
            f"σ not monotone at i={i}: {sigmas[i]} → {sigmas[i+1]}"


def test_bbox_computation():
    """3 macros + 2 ports, 2 nets — verify net bbox diagonals."""
    # Net 0: macros (0, 1) with pin offsets (0,0) and (0,0). Macro 0 at (10, 10), Macro 1 at (60, 50).
    #   Pin positions: (10, 10) and (60, 50). bbox = 50x40. diag = sqrt(2500+1600) = 64.03.
    # Net 1: macros (0, 2) + port at (0, 0). Macro 2 at (30, 100). Pins: (10,10), (30,100), (0,0).
    #   bbox = 30x100. diag = sqrt(900+10000) = 104.4.
    macro_pos = torch.tensor([[10.0, 10.0], [60.0, 50.0], [30.0, 100.0]])
    pin_macro = torch.tensor([0, 1, 0, 2, -1])      # 5 pins
    pin_xoff = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    pin_yoff = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    pin_to_net = torch.tensor([0, 0, 1, 1, 1])
    n_nets = 2
    diag = compute_per_net_bbox_diag(
        macro_pos, pin_macro, pin_xoff, pin_yoff, pin_to_net, n_nets)
    diag_np = diag.numpy()
    print(f"  bbox diags: {diag_np.tolist()}")
    print(f"  expected:   [{math.sqrt(50**2 + 40**2):.3f}, "
          f"{math.sqrt(30**2 + 100**2):.3f}]")
    expected = [math.sqrt(50**2 + 40**2), math.sqrt(30**2 + 100**2)]
    assert abs(diag_np[0] - expected[0]) < 1e-3, f"net 0: {diag_np[0]}"
    assert abs(diag_np[1] - expected[1]) < 1e-3, f"net 1: {diag_np[1]}"


if __name__ == "__main__":
    test_curriculum_weight_form()
    test_schedule_monotone()
    test_bbox_computation()
    print("ALL RG_CURRICULUM TESTS PASSED")
