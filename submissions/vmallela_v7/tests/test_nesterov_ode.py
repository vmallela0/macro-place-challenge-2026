"""Smoke test for B4: Nesterov-ODE-RK4 optimizer.

Math claim: RK4 of ẍ + (3/(t+t_init))·ẋ + ∇U = 0 minimizes a convex
quadratic. With restart, even non-convex quadratics descend monotonically.

Verification: U(x) = ½ x^T A x with A = diag(1, 10) (anisotropic well).
NesterovODE_RK4 should drive ||x|| → 0 over ~100 steps.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import torch
from _nesterov_ode import NesterovODE_RK4


def test_convex_quadratic_descent():
    """Run NesterovODE_RK4 on ½ x^T diag(1, 10) x from x_0 = (1, 1)."""
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    A = torch.diag(torch.tensor([1.0, 10.0]))
    opt = NesterovODE_RK4([x], lr=0.1, t_init=5.0,
                            restart=True, max_step_norm=None)

    def closure():
        opt.zero_grad()
        loss = 0.5 * (x @ A @ x)
        loss.backward()
        return loss

    initial_loss = float(closure())
    for _ in range(200):
        opt.step(closure)
    final_loss = float(closure())
    print(f"  loss: {initial_loss:.6f} → {final_loss:.6f}")
    print(f"  ||x||: {float(x.norm()):.6e}")
    assert final_loss < 0.1 * initial_loss, \
        f"NesterovODE_RK4 did not descend: {initial_loss} → {final_loss}"


def test_handles_negative_curvature():
    """U(x) = ½ x^T diag(1, -0.5) x has saddle at origin.

    With restart, optimizer should NOT diverge to infinity in the
    unstable direction (restart zeros v when v · ∇U > 0).

    We start at x_0 = (0.1, 0.0) — small positive y means small gradient
    in -y direction, but the restart catches when momentum aligns with
    gradient (escape direction).
    """
    x = torch.tensor([0.1, 0.01], requires_grad=True)
    A = torch.diag(torch.tensor([1.0, -0.5]))
    opt = NesterovODE_RK4([x], lr=0.05, t_init=5.0,
                            restart=True, max_step_norm=1.0)

    def closure():
        opt.zero_grad()
        loss = 0.5 * (x @ A @ x)
        loss.backward()
        return loss

    losses = []
    for _ in range(100):
        opt.step(closure)
        losses.append(float(closure()))
    # With restart, we don't expect convergence (it's a saddle), but
    # the trajectory should not blow up unboundedly.
    print(f"  saddle: loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} "
          f"||x||={float(x.norm()):.4f}")
    assert float(x.norm()) < 100.0, "Nesterov diverged on saddle"


if __name__ == "__main__":
    test_convex_quadratic_descent()
    test_handles_negative_curvature()
    print("NESTEROV_ODE TEST PASSED")
