"""Smoke test for B7: free-energy / Gaussian-smoothed proxy.

Math claims:
  1. With sigma=0 or K=1, fe_call returns the raw smooth_proxy_call (passthrough).
  2. F̂(x) = (1/K) Σ U(x + ε_k) is a noisy estimator of E[U(x+ε)] = U_smoothed(x).
  3. For a sharp basin U(x) = ½ x^T x, F̂(0) > U(0) = 0 (smoothing adds
     positive cost). Specifically, F̂(0) ≈ σ² · dim / 2 in expectation.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import torch
from _free_energy import make_free_energy_proxy


def test_passthrough_at_sigma_zero():
    """sigma=0 → exact smooth_proxy_call returned."""
    def U(x): return 0.5 * (x[0, 0] ** 2 + x[0, 1] ** 2)
    fe = make_free_energy_proxy(U, sigma=0.0, K=4, seed=0)
    out = float(fe(torch.tensor([[1.0, 2.0]])))
    expected = 0.5 * (1.0 + 4.0)
    print(f"  σ=0 passthrough: {out} (expect {expected})")
    assert abs(out - expected) < 1e-6


def test_passthrough_at_K1():
    """K=1 → also passthrough (no MC averaging)."""
    def U(x): return 0.5 * (x[0, 0] ** 2 + x[0, 1] ** 2)
    fe = make_free_energy_proxy(U, sigma=5.0, K=1, seed=0)
    out = float(fe(torch.tensor([[1.0, 2.0]])))
    expected = 0.5 * (1.0 + 4.0)
    print(f"  K=1 passthrough: {out} (expect {expected})")
    assert abs(out - expected) < 1e-6


def test_inflation_at_origin_quadratic():
    """U(x) = ½ ||x||². At x=0, F̂(0) = (1/K) Σ ½ ||ε_k||² → σ²·d/2 as K→∞.

    For d=2, σ=1.0, K=64: F̂(0) ≈ 1 (within MC noise ~0.2).
    """
    def U(x):
        return 0.5 * (x[0, 0] ** 2 + x[0, 1] ** 2)
    # Many samples for tighter estimate.
    fe = make_free_energy_proxy(U, sigma=1.0, K=64, seed=0)
    out = float(fe(torch.tensor([[0.0, 0.0]])))
    expected = 0.5 * 2 * 1.0 ** 2     # = 1.0
    print(f"  F̂(0) at σ=1, K=64: {out:.4f} (expect ≈ {expected})")
    assert abs(out - expected) < 0.5, \
        f"free-energy inflation off: {out} vs {expected}"


if __name__ == "__main__":
    test_passthrough_at_sigma_zero()
    test_passthrough_at_K1()
    test_inflation_at_origin_quadratic()
    print("FREE_ENERGY TEST PASSED")
