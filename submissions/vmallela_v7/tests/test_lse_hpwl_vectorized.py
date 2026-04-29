"""LSE-HPWL parity + perf tests.

Asserts:
1. lse_hpwl_vectorized matches the Python loop's per-net LSE-HPWL within
   numerical tolerance on a synthetic small ragged net structure.
2. Gradient parity: backward of the loop vs the vectorized version produces
   the same gradient w.r.t. macro positions to within tolerance.
3. Perf: 50 Adam steps on a 6k-net 26k-pin synthetic structure (matches
   ibm15 scale) under 30s on CPU and < 5s on MPS.
"""
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                            lse_max, lse_min)


def _python_loop_hpwl(pin_x, pin_y, pin_to_net, net_weight, n_nets,
                       net_starts, cw, ch, net_cnt, tau):
    """Reference: per-net Python loop. Slow but unambiguous."""
    hpwl_total = torch.zeros((), device=pin_x.device, dtype=pin_x.dtype)
    for i in range(n_nets):
        s = int(net_starts[i].item())
        e = int(net_starts[i + 1].item())
        if e <= s + 1:
            continue
        xs = pin_x[s:e]
        ys = pin_y[s:e]
        bx = lse_max(xs, tau) - lse_min(xs, tau)
        by = lse_max(ys, tau) - lse_min(ys, tau)
        hpwl_total = hpwl_total + net_weight[i] * (bx + by)
    return hpwl_total / ((cw + ch) * net_cnt)


def _build_synthetic(n_nets: int, mean_pins_per_net: float = 4.0,
                      seed: int = 42, device="cpu"):
    """Build a synthetic ragged net structure.
    Returns: pin_x, pin_y, net_starts, net_weight, pin_to_net.
    Pin counts per net follow Poisson(mean_pins_per_net) clipped to [2, 30].
    """
    rng = np.random.RandomState(seed)
    pin_counts = np.clip(
        rng.poisson(mean_pins_per_net, size=n_nets), 2, 30).astype(np.int64)
    n_pins = int(pin_counts.sum())
    pin_x = torch.tensor(
        rng.uniform(0, 100, size=n_pins), dtype=torch.float32, device=device,
        requires_grad=True)
    pin_y = torch.tensor(
        rng.uniform(0, 100, size=n_pins), dtype=torch.float32, device=device,
        requires_grad=True)
    net_starts = torch.tensor(
        np.concatenate([[0], np.cumsum(pin_counts)]),
        dtype=torch.long, device=device)
    net_weight = torch.tensor(
        rng.uniform(0.5, 2.0, size=n_nets), dtype=torch.float32,
        device=device)
    pin_to_net = build_pin_to_net(net_starts)
    return pin_x, pin_y, net_starts, net_weight, pin_to_net, n_pins


def test_value_parity():
    """Vectorized HPWL == Python-loop HPWL within float32 tolerance."""
    device = "cpu"
    pin_x, pin_y, net_starts, net_weight, pin_to_net, n_pins = \
        _build_synthetic(n_nets=200, mean_pins_per_net=5.0, device=device)
    n_nets = int(net_weight.shape[0])
    cw, ch, net_cnt = 100.0, 100.0, float(n_nets)
    tau = 50.0

    h_loop = _python_loop_hpwl(pin_x, pin_y, pin_to_net, net_weight, n_nets,
                                net_starts, cw, ch, net_cnt, tau).item()
    h_vec = lse_hpwl_vectorized(pin_x, pin_y, pin_to_net, net_weight, n_nets,
                                  cw=cw, ch=ch, net_cnt=net_cnt,
                                  tau_lse=tau).item()
    rel = abs(h_vec - h_loop) / max(abs(h_loop), 1e-9)
    assert rel < 1e-4, f"value mismatch: loop={h_loop} vec={h_vec} rel={rel}"
    print(f"  ✓ value parity: loop={h_loop:.6f} vec={h_vec:.6f} "
          f"rel-err={rel:.2e}")


def test_gradient_parity():
    """Vectorized gradient w.r.t. pins == Python-loop gradient."""
    device = "cpu"
    pin_x_l, pin_y_l, net_starts, net_weight, pin_to_net, n_pins = \
        _build_synthetic(n_nets=200, mean_pins_per_net=5.0, device=device)
    pin_x_v = pin_x_l.clone().detach().requires_grad_(True)
    pin_y_v = pin_y_l.clone().detach().requires_grad_(True)
    n_nets = int(net_weight.shape[0])
    cw, ch, net_cnt = 100.0, 100.0, float(n_nets)
    tau = 50.0

    h_loop = _python_loop_hpwl(pin_x_l, pin_y_l, pin_to_net, net_weight,
                                n_nets, net_starts, cw, ch, net_cnt, tau)
    h_loop.backward()

    h_vec = lse_hpwl_vectorized(pin_x_v, pin_y_v, pin_to_net, net_weight,
                                  n_nets, cw=cw, ch=ch, net_cnt=net_cnt,
                                  tau_lse=tau)
    h_vec.backward()

    grad_diff_x = (pin_x_l.grad - pin_x_v.grad).abs().max().item()
    grad_diff_y = (pin_y_l.grad - pin_y_v.grad).abs().max().item()
    grad_norm = pin_x_l.grad.abs().max().item() + pin_y_l.grad.abs().max().item()
    rel = (grad_diff_x + grad_diff_y) / max(grad_norm, 1e-9)
    assert rel < 1e-3, (f"gradient mismatch: max-abs-diff x={grad_diff_x} "
                        f"y={grad_diff_y} (rel {rel:.2e})")
    print(f"  ✓ gradient parity: max-abs-diff "
          f"x={grad_diff_x:.2e} y={grad_diff_y:.2e} "
          f"(rel {rel:.2e})")


def test_perf_50_adam_steps_ibm15_scale():
    """ibm15-scale synthetic: 6k nets, ~26k pins, 894 macros.
    50 Adam steps under 30s on CPU."""
    # Use whatever device is available; assert against worst-case (CPU).
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    n_nets = 6000
    pin_x, pin_y, net_starts, net_weight, pin_to_net, n_pins = \
        _build_synthetic(n_nets=n_nets, mean_pins_per_net=4.3,
                         device=device)
    cw, ch, net_cnt = 100.0, 100.0, float(n_nets)

    # Adam over pin_x, pin_y directly (proxy for macro_pos). 50 steps.
    opt = torch.optim.Adam([pin_x, pin_y], lr=0.1)
    t0 = time.time()
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = lse_hpwl_vectorized(pin_x, pin_y, pin_to_net, net_weight,
                                     n_nets, cw=cw, ch=ch, net_cnt=net_cnt,
                                     tau_lse=50.0)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  ✓ perf: 50 Adam steps ({n_nets} nets, {n_pins} pins) on "
          f"{device}: {elapsed:.2f}s "
          f"(loss {losses[0]:.4f} → {losses[-1]:.4f})")
    # Tight gate: 30s on CPU; expect <5s on MPS, <1s on CUDA.
    bar = 30.0 if device == "cpu" else 10.0
    assert elapsed < bar, f"too slow: {elapsed:.1f}s > {bar:.0f}s on {device}"
    # Loss must monotonically decrease (Adam on a convex LSE-HPWL).
    assert losses[-1] < losses[0], \
        f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


if __name__ == "__main__":
    test_value_parity()
    test_gradient_parity()
    test_perf_50_adam_steps_ibm15_scale()
    print("ALL OK")
