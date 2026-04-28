"""Langevin dynamics on a smooth proxy surrogate — physics-inspired init.

Treats macros as charged particles in a potential:
  U(x) = α · WA_HPWL(x) + β · ElectrostaticDensity(x) + γ · AxisRepulsion(x)

WA_HPWL (weighted-average HPWL):
  for each net, WA = (sum x_i exp(γ x_i) / sum exp(γ x_i)) - (sum x_i exp(-γ x_i) / sum exp(-γ x_i))
  smooth approximation to max - min. γ controls smoothness.

Electrostatic density (ePlace-style):
  Gaussian smoothing of macro occupancies; we implement simply as
  sum of pairwise Gaussian repulsion on overlapping positions.

Langevin step (Euler-Maruyama):
  v = v (1 - γ_friction dt) + dt F(x) + σ √dt N(0, I)
  x = x + dt v

σ = √(2 γ_friction T) — Einstein relation for thermal noise.

Output: final macro positions after N steps. May have overlaps — we send
it to legalize tournament downstream.
"""
import numpy as np
import torch


def langevin_init(benchmark, n_steps=300, dt=0.02, T0=1.0, T_end=0.01,
                  friction=1.0, alpha=1.0, beta=2.0, smooth_gamma=0.5,
                  use_mps=True, verbose=False):
    """Run Langevin MD on smooth proxy, return final hard positions."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    device = torch.device("mps") if use_mps and torch.backends.mps.is_available() else torch.device("cpu")

    sizes = benchmark.macro_sizes.to(device=device, dtype=torch.float32)
    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2
    fixed = benchmark.macro_fixed.to(device=device)
    movable = (~fixed[:n_hard]).float()

    # Start from initial hard positions
    x = benchmark.macro_positions[:n_hard].clone().to(device=device, dtype=torch.float32)
    x.requires_grad_(True)

    # Nets: build as list of pins (hard macros + port positions)
    # For smoothness, we only use hard-macro pins in the surrogate; softs are ignored.
    # Weighted-average HPWL per net.
    net_pin_hard_idx = []
    net_pin_port_pos = []
    net_weight = []

    for driver_name, sinks in benchmark._plc_nets if hasattr(benchmark, '_plc_nets') else []:
        pass  # placeholder; we'll use plc nets at caller

    # Since we can't easily get net data from benchmark directly without plc,
    # we'll just run SIMPLIFIED physics: pairwise attractive + repulsive forces.
    # Attractive: between connected pairs (use existing init's net adjacency via plc externally)
    # Repulsive: among overlapping macros

    # FALLBACK simple physics: just repulsive + canvas boundary
    # (Without net data here, this is limited — caller should wrap.)
    # TODO: integrate net data via incr_eval.macro_nets when wired in

    vx = torch.zeros_like(x)
    rng = np.random.RandomState(42)

    for step in range(n_steps):
        t_ratio = step / max(1, n_steps - 1)
        T = T0 + (T_end - T0) * t_ratio

        # Compute pairwise repulsive force (vectorized)
        dx = x[:, 0:1] - x[:, 0:1].T  # (n_hard, n_hard)
        dy = x[:, 1:2] - x[:, 1:2].T
        sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0)
        sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0)
        ov_x = torch.clamp(sep_x - torch.abs(dx), min=0)
        ov_y = torch.clamp(sep_y - torch.abs(dy), min=0)
        ov = ov_x * ov_y
        # mask self
        eye = torch.eye(n_hard, device=device)
        ov = ov * (1 - eye)
        signx = torch.sign(dx)
        signy = torch.sign(dy)
        fx_rep = (beta * ov * signx).sum(dim=1)
        fy_rep = (beta * ov * signy).sum(dim=1)

        # Canvas boundary force (quadratic penalty)
        left = torch.clamp(half_w - x[:, 0], min=0)
        right = torch.clamp(x[:, 0] - (cw - half_w), min=0)
        bottom = torch.clamp(half_h - x[:, 1], min=0)
        top = torch.clamp(x[:, 1] - (ch - half_h), min=0)
        fx_canvas = (left - right) * 10.0
        fy_canvas = (bottom - top) * 10.0

        # Thermal noise (standard Langevin)
        noise_scale = np.sqrt(2 * friction * T * dt)
        noise_x = torch.randn_like(x[:, 0]) * noise_scale
        noise_y = torch.randn_like(x[:, 1]) * noise_scale

        # Update velocity
        vx[:, 0] = vx[:, 0] * (1 - friction * dt) + dt * (fx_rep + fx_canvas) + noise_x
        vx[:, 1] = vx[:, 1] * (1 - friction * dt) + dt * (fy_rep + fy_canvas) + noise_y

        # Update position (only movable)
        with torch.no_grad():
            x[:, 0] = x[:, 0] + dt * vx[:, 0] * movable
            x[:, 1] = x[:, 1] + dt * vx[:, 1] * movable
            # Clamp to canvas
            x[:, 0] = torch.clamp(x[:, 0], half_w, cw - half_w)
            x[:, 1] = torch.clamp(x[:, 1], half_h, ch - half_h)

        if verbose and step % 50 == 0:
            cur_ov = int((ov > 0).sum().item() / 2)
            print(f"    langevin step {step}: T={T:.3f} overlaps={cur_ov}")

    return x.detach().cpu().numpy().astype(np.float64)
