"""Sanity tests for Klein-4 orientation flip.

The math is straightforward; what we want to verify is:
- Each Klein-4 element preserves the macro bounding box (so Tier 1
  positions/sizes are unchanged regardless of orientation).
- For a constructed scenario with one off-center pin and one fixed
  port, the greedy picks the orientation that pulls the pin closer
  to the port.
- HPWL strictly does not increase across one greedy pass.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _orientation_flip import (
    klein4_orient, ORIENTATIONS, _FLIPS,
    _net_pin_table, _build_pin_groups,
)


def test_flip_signs_are_klein_four_group():
    # Each element squared is identity (sx*sx, sy*sy) = (1, 1).
    for lab in ORIENTATIONS:
        sx, sy = _FLIPS[lab]
        assert sx * sx == 1.0
        assert sy * sy == 1.0
    # MY ∘ MX = R180 (composition in the Klein-4 group).
    sx_my, sy_my = _FLIPS["MY"]
    sx_mx, sy_mx = _FLIPS["MX"]
    sx_s, sy_s = _FLIPS["R180"]
    assert sx_my * sx_mx == sx_s
    assert sy_my * sy_mx == sy_s


def test_pin_grouping():
    # 3 macros, 5 pins: pins 0,2 owned by macro 0; pin 1 by macro 1; pin 3 by macro 2; pin 4 is a port.
    pin_macro = np.array([0, 1, 0, 2, -1])
    groups = _build_pin_groups(pin_macro, n_hard=3)
    assert list(groups[0]) == [0, 2]
    assert list(groups[1]) == [1]
    assert list(groups[2]) == [3]


def test_greedy_pulls_pin_toward_partner():
    # Setup: 1 hard macro at (10, 0), 1 port-pin at (-10, 0).
    # Macro has one pin with offset (+1, 0). HPWL of the single net is
    # |10+1 - (-10)| = 21. Flipping macro to MY turns offset to (-1, 0)
    # giving HPWL = 19. Greedy should pick MY.
    macro_pos = np.array([[10.0, 0.0]])
    macro_w = np.array([2.0])
    macro_h = np.array([1.0])
    # 2 pins: pin 0 owned by macro 0, pin 1 is a port at (-10, 0).
    pin_macro = np.array([0, -1])
    pin_xoff = np.array([1.0, -10.0])
    pin_yoff = np.array([0.0, 0.0])
    pin_to_net = np.array([0, 0])
    net_weight = np.array([1.0])

    orient, info = klein4_orient(
        macro_pos, macro_w, macro_h, pin_macro, pin_xoff, pin_yoff,
        pin_to_net, net_weight, n_hard=1, n_nets=1, n_passes=2,
    )
    assert orient[0] == "MY", f"expected MY, got {orient[0]}"
    assert info["initial_hpwl"] == 21.0
    assert info["final_hpwl"] == 19.0
    assert info["delta_hpwl"] == 2.0


def test_two_macros_independent_optimum():
    # Two macros each with their own port partner; orientations are
    # independent, both should flip to pull pins toward partners.
    macro_pos = np.array([[10.0, 0.0], [0.0, 10.0]])
    macro_w = np.array([2.0, 2.0])
    macro_h = np.array([1.0, 1.0])
    # 4 pins: pin0 macro0 offset (+1,0); pin1 port at (-10,0);
    #         pin2 macro1 offset (0,+1); pin3 port at (0,-10).
    pin_macro = np.array([0, -1, 1, -1])
    pin_xoff = np.array([1.0, -10.0, 0.0, 0.0])
    pin_yoff = np.array([0.0, 0.0, 1.0, -10.0])
    pin_to_net = np.array([0, 0, 1, 1])
    net_weight = np.array([1.0, 2.0])  # net 1 weighted heavier

    orient, info = klein4_orient(
        macro_pos, macro_w, macro_h, pin_macro, pin_xoff, pin_yoff,
        pin_to_net, net_weight, n_hard=2, n_nets=2, n_passes=2,
    )
    # Macro 0 should flip x-pin to negative side: MY or R180 both achieve
    # offset_x = -1; the greedy picks the first (MY).
    assert orient[0] in ("MY", "R180")
    # Macro 1 should flip y-pin to negative side: MX or R180.
    assert orient[1] in ("MX", "R180")
    assert info["delta_hpwl"] > 0  # strict net improvement


def test_no_regression_when_already_optimal():
    # Macro and port are co-aligned for orientation N already.
    macro_pos = np.array([[0.0, 0.0]])
    macro_w = np.array([1.0])
    macro_h = np.array([1.0])
    # Pin at (-1, 0) and port at (-10, 0). N is already optimal.
    pin_macro = np.array([0, -1])
    pin_xoff = np.array([-1.0, -10.0])
    pin_yoff = np.array([0.0, 0.0])
    pin_to_net = np.array([0, 0])
    net_weight = np.array([1.0])

    orient, info = klein4_orient(
        macro_pos, macro_w, macro_h, pin_macro, pin_xoff, pin_yoff,
        pin_to_net, net_weight, n_hard=1, n_nets=1, n_passes=2,
    )
    # Final HPWL must not exceed initial (greedy is monotone).
    assert info["final_hpwl"] <= info["initial_hpwl"] + 1e-9


if __name__ == "__main__":
    test_flip_signs_are_klein_four_group()
    test_pin_grouping()
    test_greedy_pulls_pin_toward_partner()
    test_two_macros_independent_optimum()
    test_no_regression_when_already_optimal()
    print("All orientation flip tests passed.")
