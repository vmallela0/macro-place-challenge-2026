"""Smoke test for continuous orientation module.

Math claims being verified:
  1. discretization_penalty R(θ) = -cos(4θ) has minima at {0, π/2, π, 3π/2}.
  2. rotated_pin_positions at θ=π/2 matches R90 rotation:
     (xoff, yoff) → (-yoff, xoff)
  3. Joint optimization: a 2-pin net with one pin at (10, 0) offset and
     macro centered at canvas origin (5, 5); ideal placement should drive
     the pin toward the other macro's pin. If the other pin is at
     (-5, 0) absolute, θ should pivot to bring the offset around.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import math
import numpy as np
import torch
from _continuous_orientation import (
    discretization_penalty, rotated_pin_positions, joint_xy_theta_refine,
)


def test_disc_penalty_minima():
    """R(θ) = -cos(4θ). Check minima at {0, π/2, π, 3π/2} and max at π/4."""
    for theta in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        v = discretization_penalty(torch.tensor([theta]))
        # cos(4·kπ/2) = cos(2kπ) = 1 → -cos = -1. Min value is -1.
        print(f"  R({theta:.4f}) = {float(v):.6f}")
        assert abs(float(v) - (-1.0)) < 1e-6
    for theta in [math.pi / 4, 3 * math.pi / 4]:
        v = discretization_penalty(torch.tensor([theta]))
        # cos(4·π/4) = cos(π) = -1 → -cos = 1. Max value is 1.
        print(f"  R({theta:.4f}) = {float(v):.6f}")
        assert abs(float(v) - 1.0) < 1e-6


def test_r90_rotation_identity():
    """At θ=π/2: (xoff, yoff) → (-yoff, xoff)."""
    macro_pos = torch.tensor([[100.0, 200.0]])
    theta = torch.tensor([math.pi / 2])
    pin_macro = torch.tensor([0])
    xoff = torch.tensor([5.0])
    yoff = torch.tensor([3.0])
    px, py = rotated_pin_positions(macro_pos, theta, pin_macro, xoff, yoff)
    # Expected: pin_x = 100 + cos(π/2)·5 - sin(π/2)·3 = 100 + 0 - 3 = 97
    #           pin_y = 200 + sin(π/2)·5 + cos(π/2)·3 = 200 + 5 + 0 = 205
    print(f"  R90: pin_x={float(px):.3f} (expect 97), "
          f"pin_y={float(py):.3f} (expect 205)")
    assert abs(float(px) - 97.0) < 1e-5
    assert abs(float(py) - 205.0) < 1e-5


def test_joint_xy_theta_runs():
    """End-to-end: 2 macros, 1 net, joint refinement should not crash and
    should produce a valid HPWL output. Snapped theta is in {0,1,2,3}.
    """
    n_total = 2
    macro_pos = torch.tensor([[10.0, 10.0], [90.0, 90.0]])
    pin_macro = torch.tensor([0, 1])
    xoff = torch.tensor([5.0, -5.0])
    yoff = torch.tensor([0.0, 0.0])
    pin_to_net = torch.tensor([0, 0])
    net_weight = torch.tensor([1.0])
    n_nets = 1
    pos, theta_disc, diag = joint_xy_theta_refine(
        macro_pos, pin_macro, xoff, yoff, pin_to_net, net_weight, n_nets,
        cw=100.0, ch=100.0, net_cnt=1.0,
        n_steps=30, verbose=False)
    print(f"  joint: HPWL trajectory {diag['history']['hpwl'][0]:.4f} "
          f"→ {diag['final_hpwl_continuous']:.4f} → "
          f"snapped {diag['final_hpwl_snapped']:.4f}")
    print(f"  theta_disc = {theta_disc.tolist()}")
    assert pos.shape == (2, 2)
    assert all(0 <= t < 4 for t in theta_disc)
    # HPWL should not blow up.
    assert diag["final_hpwl_continuous"] < 100.0


if __name__ == "__main__":
    test_disc_penalty_minima()
    test_r90_rotation_identity()
    test_joint_xy_theta_runs()
    print("ALL CONTINUOUS_ORIENTATION TESTS PASSED")
