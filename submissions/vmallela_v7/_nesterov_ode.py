"""Nesterov continuous-time ODE optimizer — zeus B4.

Why this exists
===============
Nesterov accelerated gradient (NAG) is a well-known discretization of
    ẍ + (3/t) · ẋ + ∇U(x) = 0
(Su, Boyd & Candès, 2014 — "A Differential Equation for Modeling Nesterov's
Accelerated Gradient Method"). The (3/t) damping is the key: it decays
from large initial damping (slow, careful steps) to zero asymptotic
damping (fast, accelerated steps), giving Nesterov's O(1/t²) convergence
on convex U.

The discrete form everyone uses (Nesterov 1983, also called NAG / FISTA):
    v_{k+1} = α_k · v_k - lr · ∇U(x_k + α_k · v_k)
    x_{k+1} = x_k + v_{k+1}
with α_k = (k-1)/(k+2). This is a 1st-order discretization of the ODE,
and it's O(h²) accurate vs the continuous flow.

What's new here
---------------
Adam uses RMSprop + momentum at fixed coefficients (β1 ≈ 0.9, β2 ≈ 0.999).
The momentum term has CONSTANT decay rate β1 — equivalent to a fixed
damping coefficient in the underlying ODE. That works well for stochastic
gradients but is suboptimal for our deterministic smooth surrogate:
  • The fixed damping is too high early (no exploration) and too low late
    (oscillation around local min).
  • Adam's per-parameter scaling helps with stiffness but masks the
    underlying acceleration structure.

The Nesterov ODE with time-decaying (3/t) damping is theoretically
optimal for smooth convex U; for non-convex U (our case), the time-
decaying damping still helps because it allows large early steps to
escape shallow basins, then anneals to convergence at the asymptotic rate.

The "RK4" wrinkle: we discretize the ODE with explicit 4th-order
Runge-Kutta on the 1st-order system (x, v). Each step uses 4 gradient
evaluations but the per-step error is O(h⁵), so we can use ~10× larger
step than NAG at the same accuracy → fewer total iterations.

Math (1st-order rephrasing)
===========================
ẋ = v
v̇ = -∇U(x) - (3/t) v
f(x, v, t) = (v, -∇U(x) - (3/t)v)
RK4 step:
    k1 = f(x_n,           v_n,           t_n)
    k2 = f(x_n + h/2·k1x, v_n + h/2·k1v, t_n + h/2)
    k3 = f(x_n + h/2·k2x, v_n + h/2·k2v, t_n + h/2)
    k4 = f(x_n + h·k3x,   v_n + h·k3v,   t_n + h)
    (x_{n+1}, v_{n+1}) = (x_n, v_n) + h/6 · (k1 + 2k2 + 2k3 + k4)

For early times (t<1), the (3/t) damping is huge — we damp this with
(3/(t + t_init)) where t_init ≥ 1 to avoid divergence near t=0.

Failure modes
=============
- Non-convex U: NAG isn't guaranteed to converge. We add a restart
  rule (O'Donoghue & Candès 2015): if v · ∇U > 0 (going uphill), reset
  v ← 0. This guarantees monotone descent.
- Stiff U: large ||∇U|| means the explicit RK4 step blows up. Cap step
  by max_step_norm in canvas units.

Usage
-----
A torch.optim.Optimizer subclass, drop-in replacement for Adam.
"""

from __future__ import annotations
from typing import Iterable
import torch


class NesterovODE_RK4(torch.optim.Optimizer):
    """RK4 integrator for ẍ + (3/(t+t_init))·ẋ + ∇U = 0.

    Parameters
    ----------
    params : iterable of torch tensors. Each gets a v-state.
    lr : the ODE time step h.  Default 1.0 means one "ODE time unit"
        per optimizer step. Combined with the position scale, lr should
        be on the order of canvas_diag / n_steps.
    t_init : initial damping-time offset.  Default 5.0. Larger means
        smoother damping ramp.
    restart : if True, apply O'Donoghue-Candès momentum restart
        (zero v_k when ⟨v_k, ∇U⟩ > 0). Default True.
    max_step_norm : per-step cap on ||Δx|| in raw units (microns).
        Default None = no cap. Use canvas_diag·0.05 for safety.
    """
    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float = 1.0,
                 t_init: float = 5.0,
                 restart: bool = True,
                 max_step_norm: float | None = None):
        defaults = dict(lr=lr, t_init=t_init, restart=restart,
                         max_step_norm=max_step_norm)
        super().__init__(params, defaults)
        self._step_counter = 0
        self._cached_gradf = None    # for k2/k3 sharing

    @torch.no_grad()
    def step(self, closure):
        """closure: callable returning scalar loss (with create_graph=False).

        Called 4 times per step (one per RK4 stage), so the caller's
        closure must rebuild the graph each call.
        """
        if closure is None:
            raise RuntimeError("NesterovODE_RK4 requires a closure")
        self._step_counter += 1
        n = self._step_counter

        # We do RK4 on the (x, v) system.  Each stage needs ∇U at a
        # provisional position.  We mutate parameters in-place, evaluate
        # the closure to get gradients, then restore.  This requires the
        # closure to be deterministic in (x, t_density, t_cong, ...).

        # Snapshot current params and velocities.
        snapshot_x: list[list[torch.Tensor]] = []
        v_states: list[list[torch.Tensor]] = []
        for group in self.param_groups:
            sx, sv = [], []
            for p in group["params"]:
                sx.append(p.data.clone())
                st = self.state[p]
                if "v" not in st:
                    st["v"] = torch.zeros_like(p.data)
                sv.append(st["v"])
            snapshot_x.append(sx)
            v_states.append(sv)

        def _eval_grad():
            # Caller's closure is expected to zero_grad, compute loss,
            # backward, and apply any per-parameter grad masking
            # (e.g., zeroing hard-macro rows). We just harvest p.grad.
            with torch.enable_grad():
                loss = closure()
            grads = []
            for group in self.param_groups:
                gg = []
                for p in group["params"]:
                    gg.append(p.grad.detach().clone()
                               if p.grad is not None
                               else torch.zeros_like(p.data))
                grads.append(gg)
            return grads, float(loss.detach())

        def _set_params(positions: list[list[torch.Tensor]]):
            for group, pos_g in zip(self.param_groups, positions):
                for p, x in zip(group["params"], pos_g):
                    p.data.copy_(x)

        h = float(self.param_groups[0]["lr"])
        t_init = float(self.param_groups[0]["t_init"])
        damping = lambda t: 3.0 / (t + t_init)

        # RK4 stage 1: f(x_n, v_n, t_n).
        # x_dot = v_n; v_dot = -∇U(x_n) - damping(n)·v_n
        _set_params(snapshot_x)
        grads, loss_init = _eval_grad()
        k1_x = v_states
        k1_v = [[-g - damping(n) * v for g, v in zip(g_g, v_g)]
                 for g_g, v_g in zip(grads, v_states)]

        # RK4 stage 2: f(x_n + h/2·k1x, v_n + h/2·k1v, t_n + h/2)
        x_mid1 = [[x + 0.5 * h * kx for x, kx in zip(sx, k1x_g)]
                    for sx, k1x_g in zip(snapshot_x, k1_x)]
        v_mid1 = [[v + 0.5 * h * kv for v, kv in zip(v_g, k1v_g)]
                    for v_g, k1v_g in zip(v_states, k1_v)]
        _set_params(x_mid1)
        grads2, _ = _eval_grad()
        k2_x = v_mid1
        k2_v = [[-g - damping(n + 0.5) * v for g, v in zip(g_g, v_g)]
                 for g_g, v_g in zip(grads2, v_mid1)]

        # RK4 stage 3: f(x_n + h/2·k2x, v_n + h/2·k2v, t_n + h/2)
        x_mid2 = [[x + 0.5 * h * kx for x, kx in zip(sx, k2x_g)]
                    for sx, k2x_g in zip(snapshot_x, k2_x)]
        v_mid2 = [[v + 0.5 * h * kv for v, kv in zip(v_g, k2v_g)]
                    for v_g, k2v_g in zip(v_states, k2_v)]
        _set_params(x_mid2)
        grads3, _ = _eval_grad()
        k3_x = v_mid2
        k3_v = [[-g - damping(n + 0.5) * v for g, v in zip(g_g, v_g)]
                 for g_g, v_g in zip(grads3, v_mid2)]

        # RK4 stage 4: f(x_n + h·k3x, v_n + h·k3v, t_n + h)
        x_end = [[x + h * kx for x, kx in zip(sx, k3x_g)]
                   for sx, k3x_g in zip(snapshot_x, k3_x)]
        v_end = [[v + h * kv for v, kv in zip(v_g, k3v_g)]
                   for v_g, k3v_g in zip(v_states, k3_v)]
        _set_params(x_end)
        grads4, _ = _eval_grad()
        k4_x = v_end
        k4_v = [[-g - damping(n + 1.0) * v for g, v in zip(g_g, v_g)]
                 for g_g, v_g in zip(grads4, v_end)]

        # Combine: (x_{n+1}, v_{n+1}) = (x_n, v_n) + h/6·(k1+2k2+2k3+k4)
        new_x = []
        new_v = []
        for g_idx, group in enumerate(self.param_groups):
            sx_g = snapshot_x[g_idx]
            v_g = v_states[g_idx]
            new_x_g = []
            new_v_g = []
            for p_idx, p in enumerate(group["params"]):
                dx = (h / 6.0) * (k1_x[g_idx][p_idx]
                                    + 2.0 * k2_x[g_idx][p_idx]
                                    + 2.0 * k3_x[g_idx][p_idx]
                                    + k4_x[g_idx][p_idx])
                dv = (h / 6.0) * (k1_v[g_idx][p_idx]
                                    + 2.0 * k2_v[g_idx][p_idx]
                                    + 2.0 * k3_v[g_idx][p_idx]
                                    + k4_v[g_idx][p_idx])
                # Optional step-norm cap on Δx.
                max_step = group.get("max_step_norm")
                if max_step is not None:
                    norm = float(dx.norm().item())
                    if norm > max_step:
                        dx = dx * (max_step / max(norm, 1e-30))
                new_x_p = sx_g[p_idx] + dx
                new_v_p = v_g[p_idx] + dv
                # O'Donoghue-Candès restart: if v · ∇U > 0, going uphill;
                # damp v to zero.
                if group["restart"]:
                    g_now = grads[g_idx][p_idx]    # ∇U at start of step
                    if (new_v_p * g_now).sum() > 0:
                        new_v_p = torch.zeros_like(new_v_p)
                new_x_g.append(new_x_p)
                new_v_g.append(new_v_p)
            new_x.append(new_x_g)
            new_v.append(new_v_g)

        # Commit.
        _set_params(new_x)
        for g_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group["params"]):
                self.state[p]["v"] = new_v[g_idx][p_idx]
        return loss_init
