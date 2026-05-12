"""Smoke test for B6: JKO/Wasserstein step.

Math claims:
  1. Log-stabilized Sinkhorn-Knopp converges to a doubly-stochastic plan
     when a = b = uniform.
  2. Marginals are approximately preserved: Σ_j π_ij ≈ a_i, Σ_i π_ij ≈ b_j.
  3. The barycentric projection (π @ y)/Σπ recovers y when π is identity.
  4. JKO step produces a placement closer to the gradient target than
     the original.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
import torch
from _jko_step import sinkhorn_log_stabilized, jko_proximal_step


def test_sinkhorn_marginals():
    """With a = b = uniform and a symmetric cost matrix, the transport
    plan must satisfy Σ_j π_ij ≈ a_i for all i (and similarly for cols).
    """
    n = 5
    rng = np.random.RandomState(0)
    C_np = rng.exponential(1.0, size=(n, n))
    C = torch.tensor(C_np)
    a = torch.full((n,), 1.0 / n)
    b = torch.full((n,), 1.0 / n)
    pi, log_u, log_v = sinkhorn_log_stabilized(C, a, b, epsilon=1.0, n_iters=80)
    row_sums = pi.sum(dim=1).numpy()
    col_sums = pi.sum(dim=0).numpy()
    print(f"  row sums: {row_sums.tolist()} (expect ≈ {1/n:.3f})")
    print(f"  col sums: {col_sums.tolist()} (expect ≈ {1/n:.3f})")
    assert np.allclose(row_sums, 1.0 / n, atol=1e-3)
    assert np.allclose(col_sums, 1.0 / n, atol=1e-3)


def test_jko_moves_toward_gradient_target():
    """Take macro at (0, 0), gradient pointing to (10, 0). After JKO
    step, the macro should be closer to (10, 0) than before.

    With n=1 macro the OT is trivial (identity plan), so the JKO update
    just does y = x - τ∇U and blends. With α=1.0 we get x_new = y.
    """
    macro_pos = torch.tensor([[0.0, 0.0]])
    grad_U = torch.tensor([[-10.0, 0.0]])    # gradient = -10 in x means target +10
    x_new, diag = jko_proximal_step(
        macro_pos, grad_U,
        tau=1.0, alpha=1.0, sinkhorn_eps=1.0, sinkhorn_iters=30,
        n_hard=0, soft_only=False,
    )
    print(f"  x_new = {x_new.numpy().tolist()} (expect close to [[10, 0]])")
    # With α=1 and n=1, π ≈ 1 identity → x_new ≈ y = (10, 0).
    assert abs(float(x_new[0, 0]) - 10.0) < 0.5
    assert abs(float(x_new[0, 1])) < 0.5


if __name__ == "__main__":
    test_sinkhorn_marginals()
    test_jko_moves_toward_gradient_target()
    print("JKO TEST PASSED")
