"""Smoke test for B3: sparse L1/Lp/Linf cong aggregators.

Math claims:
  1. l1_excess(v, target) = Σ softplus(v − target) — zero for v ≪ target,
     ≈ (v − target) for v ≫ target.
  2. lp_excess at p=1 ≈ l1_excess (modulo normalization 1/p).
  3. linf_excess (LSE-max) ≥ max(v − target) and converges as τ → ∞.
  4. Gradient of l1_excess is concentrated on cells where v ≥ target.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
import torch
from _smooth_proxy import l1_excess, lp_excess, linf_excess


def test_l1_excess_form():
    """At target=1.0 with values [0.1, 0.9, 1.0, 1.5, 3.0]:
    softplus(x − 1.0) with μ=100 is essentially ReLU(x-1):
      0.1 → ≈ 0
      0.9 → ≈ 0
      1.0 → ≈ 0
      1.5 → 0.5
      3.0 → 2.0
    Sum ≈ 2.5
    """
    v = torch.tensor([0.1, 0.9, 1.0, 1.5, 3.0])
    out = float(l1_excess(v, 1.0, mu=100.0))
    print(f"  l1_excess = {out:.4f} (expect ≈ 2.5)")
    assert abs(out - 2.5) < 0.05


def test_linf_excess_max():
    """linf_excess ≥ max(v − target). With τ=30: should be close to true max."""
    v = torch.tensor([0.5, 1.2, 5.0, 1.0])
    target = 0.0
    out = float(linf_excess(v, target, tau=30.0))
    print(f"  linf_excess (τ=30) = {out:.4f} (true max = 5.0)")
    assert out >= 5.0 - 0.01    # smooth max is at least true max minus LSE bias


def test_l1_grad_concentrated():
    """Gradient of l1_excess w.r.t. each v_i = softplus'(v_i - t) = σ(μ(v_i - t)).

    With μ=100, σ(x) ≈ {0 if x<0, 1 if x>0}. So grad is concentrated on
    cells exceeding target.
    """
    v = torch.tensor([0.1, 0.9, 1.0, 1.5, 3.0], requires_grad=True)
    out = l1_excess(v, 1.0, mu=100.0)
    out.backward()
    g = v.grad.numpy()
    print(f"  grad: {g.tolist()}")
    # Expect g ≈ [0, 0, 0.5, 1, 1].
    assert g[0] < 0.01
    assert g[1] < 0.01
    assert g[3] > 0.99
    assert g[4] > 0.99


def test_lp_at_p1_matches_l1():
    """lp_excess at p=1: (Σ softplus^1)^1 = Σ softplus = l1_excess. Verify."""
    v = torch.tensor([0.1, 1.5, 3.0])
    target = 1.0
    a = float(l1_excess(v, target, mu=100.0))
    b = float(lp_excess(v, target, p=1.0, mu=100.0))
    print(f"  l1={a:.4f}  lp(p=1)={b:.4f}")
    assert abs(a - b) < 1e-3


if __name__ == "__main__":
    test_l1_excess_form()
    test_linf_excess_max()
    test_l1_grad_concentrated()
    test_lp_at_p1_matches_l1()
    print("SPARSE_CONG TEST PASSED")
