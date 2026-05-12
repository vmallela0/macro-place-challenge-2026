"""Smoke test for NEB module on known landscapes.

Claim 1: For the 1D double-well U(x) = (x² - 1)², the MEP between
basins x=-1 and x=+1 passes through the saddle at x=0 with U=1.
NEB should find this saddle within finite-image discretization error.

Claim 2: For a single quadratic well, the band relaxes onto the
straight line (no saddle), and the "barrier" is small / negative.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
from _neb import neb_relax


def test_double_well_finds_saddle():
    """Double-well U(x) = (x²-1)² → saddle at x=0 with U=1."""
    n_total = 1
    x_A = np.array([[-1.0, 0.0]])
    x_B = np.array([[+1.0, 0.0]])

    def U_grad_eval(x_np):
        x = x_np[0, 0]
        U = (x ** 2 - 1.0) ** 2
        # dU/dx = 4x(x² - 1); dU/dy = 0
        gx = 4.0 * x * (x ** 2 - 1.0)
        return float(U), np.array([[gx, 0.0]])

    images, Us, diag = neb_relax(
        x_A, x_B, U_grad_eval,
        n_images=11, n_iters=80, lr=0.05, spring_k=0.5,
        n_hard=0, verbose=False)
    saddle_idx = diag["U_max_idx"]
    saddle_x = images[saddle_idx][0, 0]
    saddle_U = diag["U_max"]
    print(f"  double-well: saddle at x={saddle_x:+.3f} (true 0.000), "
          f"U={saddle_U:.3f} (true 1.000), barrier={diag['barrier']:.3f}")
    assert abs(saddle_x) < 0.15, f"saddle off: x={saddle_x}"
    assert abs(saddle_U - 1.0) < 0.05, f"saddle U off: {saddle_U}"


def test_single_well_no_saddle():
    """Single quadratic U(x) = x² → no barrier between any two points."""
    n_total = 1
    x_A = np.array([[-1.0, 0.0]])
    x_B = np.array([[+1.0, 0.0]])

    def U_grad_eval(x_np):
        x = x_np[0, 0]
        return float(x ** 2), np.array([[2.0 * x, 0.0]])

    images, Us, diag = neb_relax(
        x_A, x_B, U_grad_eval,
        n_images=9, n_iters=50, lr=0.1, spring_k=0.5,
        n_hard=0, verbose=False)
    # Endpoints both at U=1; interior images should be at U<1 (along
    # the straight line midpoints are at x=0, U=0).
    print(f"  single-well: barrier={diag['barrier']:.4f} "
          f"(should be ≤ 0; endpoints U={diag['U_endpoints']})")
    # Endpoints have U=1; interior min should be near 0; barrier may be
    # slightly positive due to image-discretization noise.
    assert diag["barrier"] < 0.05, f"single-well has spurious barrier: {diag['barrier']}"


if __name__ == "__main__":
    test_double_well_finds_saddle()
    test_single_well_no_saddle()
    print("ALL NEB TESTS PASSED")
