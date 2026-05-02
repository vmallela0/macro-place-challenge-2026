"""Riemannian descent on the no-overlap manifold.

Manifold M = {x : no macro overlap, all on canvas}. Riemannian gradient
descent at x_k:

    g          = ∇f(x_k)                 (ambient gradient)
    g_T        = Proj_{T_x M}(g)         (project onto tangent space)
    x_pre      = x_k - η g_T             (tangent step)
    x_{k+1}    = R_{x_k}(x_pre - x_k)    (retract via short-range push-apart)

We approximate the tangent projection by zeroing the components of g
along the active constraints (overlap axes pointing inward). When no
constraints are active at x_k, g_T = g (descent is unconstrained
locally). When near a contact, project out the inward-normal
component.

The retraction is the windowed legalize operator from
_short_pushapart.short_pushapart, which only fixes local overlaps so
the gradient signal is preserved across consecutive steps.

Auto-tuning of `retraction_radius`:
- start at 2 * mean_macro_size
- halve if any iterate still has overlaps after retraction
- double if iterates barely move (||x_{k+1} - x_k|| < tiny ratio of
  the proposed step)

Reference: Boumal, "Introduction to Optimization on Smooth Manifolds"
(2023), §3 (retractions) and §4 (Riemannian gradient descent).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable

from _short_pushapart import short_pushapart, overlap_pairs


# ── Tangent projection ─────────────────────────────────────────────────


def tangent_projection(
    grad: NDArray,
    pos: NDArray,
    w: NDArray,
    h: NDArray,
    *,
    contact_eps: float = 1e-3,
) -> NDArray:
    """Project ambient gradient onto the tangent space of the no-overlap
    manifold at pos.

    For each pair of macros that are "in contact" (gap ≤ contact_eps
    along one axis, overlapping along the other), zero out the
    components of grad that would push them deeper into the contact.

    Implementation: for each contact pair (i, j) along axis a with
    sign(pos[i, a] - pos[j, a]) = +1, the inward-normal direction is
    (-e_a for i, +e_a for j). Remove the component of grad along that
    direction.
    """
    g = grad.astype(np.float64).copy()
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            gap_x = abs(dx) - (w[i] + w[j]) / 2.0
            gap_y = abs(dy) - (h[i] + h[j]) / 2.0
            # In contact when one axis has near-zero gap and the OTHER
            # axis has overlap (negative gap). Otherwise the macros are
            # not touching.
            if (abs(gap_x) <= contact_eps and gap_y < 0.0) or \
               (abs(gap_y) <= contact_eps and gap_x < 0.0):
                # Determine contact axis (the small-gap one)
                contact_axis = 0 if abs(gap_x) <= contact_eps else 1
                sign_i = 1.0 if pos[i, contact_axis] >= pos[j, contact_axis] else -1.0
                # Inward normal for i: -sign_i e_a. For j: +sign_i e_a.
                # Remove inward component from each:
                g_i = g[i, contact_axis]
                g_j = g[j, contact_axis]
                # The constraint is "i and j stay separated along axis a".
                # Inward-pointing means sign(g_i) opposes sign_i (i.e.
                # gradient pushes i toward j) AND sign(g_j) is sign_i
                # (gradient pushes j toward i).
                # Project out the symmetric inward component:
                inward = 0.5 * (-sign_i * g_i + sign_i * g_j)
                if inward > 0:  # inward push present
                    g[i, contact_axis] += sign_i * inward
                    g[j, contact_axis] -= sign_i * inward
    return g


# ── Retraction ─────────────────────────────────────────────────────────


def retract(
    pos_old: NDArray,
    tangent_step: NDArray,
    w: NDArray,
    h: NDArray,
    *,
    n_hard: int,
    radius: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[NDArray, dict]:
    """Retract from pos_old along tangent_step onto the manifold.

    R_x(0) = x exactly (caller relies on this for first-order test).
    For nonzero tangent_step, take the ambient step then run windowed
    push-apart to resolve any overlaps within `radius` of the movers.
    """
    if np.linalg.norm(tangent_step) < 1e-15:
        return pos_old.copy(), {"moved_macros": np.array([], dtype=np.int64),
                                 "n_iters": 0, "n_overlaps": 0}
    pos_new = pos_old + tangent_step
    pos_legal, info = short_pushapart(
        pos_old, pos_new, w, h,
        n_hard=n_hard, radius=radius,
        canvas_w=canvas_w, canvas_h=canvas_h)
    return pos_legal, info


# ── Riemannian step ────────────────────────────────────────────────────


def riemannian_step(
    pos: NDArray,
    grad_fn: Callable[[NDArray], NDArray],
    w: NDArray, h: NDArray,
    *,
    n_hard: int,
    eta: float,
    radius: float,
    canvas_w: float, canvas_h: float,
) -> tuple[NDArray, dict]:
    """One Riemannian gradient-descent step.

    Returns (pos_new, info). info has step magnitudes, contact stats,
    and retraction outcome.
    """
    g = grad_fn(pos)
    g_T = tangent_projection(g, pos, w, h)
    g_T_norm = float(np.linalg.norm(g_T))
    tangent_step = -eta * g_T
    pos_new, retr_info = retract(
        pos, tangent_step, w, h,
        n_hard=n_hard, radius=radius,
        canvas_w=canvas_w, canvas_h=canvas_h)
    info = {
        "g_norm": float(np.linalg.norm(g)),
        "g_T_norm": g_T_norm,
        "tangent_step_norm": float(np.linalg.norm(tangent_step)),
        "actual_step_norm": float(np.linalg.norm(pos_new - pos)),
        "retract": retr_info,
    }
    return pos_new, info


def riemannian_descent(
    pos: NDArray,
    grad_fn: Callable[[NDArray], NDArray],
    energy_fn: Callable[[NDArray], float],
    w: NDArray, h: NDArray,
    *,
    n_hard: int,
    eta: float,
    radius_init: float,
    canvas_w: float, canvas_h: float,
    n_steps: int = 200,
    autotune_radius: bool = True,
) -> tuple[NDArray, float, dict]:
    """Run n_steps Riemannian GD with strict-improvement gating.

    `radius_init` defaults to 2 * mean_macro_size if caller passes
    something sensible. Auto-tunes mid-run per the spec:
        - halve if any retraction leaves overlaps
        - double if actual_step / tangent_step < 0.5 too many times in a row
          (gradient signal compressed by retraction)

    Returns (best_pos, best_energy, info).
    """
    pos_cur = pos.astype(np.float64).copy()
    E_cur = float(energy_fn(pos_cur))
    best_pos = pos_cur.copy()
    best_E = E_cur

    radius = radius_init
    halve_streak = 0
    double_streak = 0
    history: list[dict] = []

    for k in range(n_steps):
        pos_try, step_info = riemannian_step(
            pos_cur, grad_fn, w, h,
            n_hard=n_hard, eta=eta, radius=radius,
            canvas_w=canvas_w, canvas_h=canvas_h)

        # Strict-improvement gate via energy.
        E_try = float(energy_fn(pos_try))
        retr = step_info["retract"]
        accepted = False
        if retr["n_overlaps"] == 0 and E_try < E_cur - 1e-9:
            pos_cur = pos_try
            E_cur = E_try
            accepted = True
            if E_try < best_E:
                best_E = E_try
                best_pos = pos_try.copy()

        # Auto-tune radius per spec.
        if autotune_radius:
            if retr["n_overlaps"] > 0:
                radius *= 0.5
                halve_streak += 1
                double_streak = 0
            else:
                ts = step_info["tangent_step_norm"]
                ac = step_info["actual_step_norm"]
                if ts > 1e-12 and (ac / ts) < 0.5:
                    double_streak += 1
                    halve_streak = 0
                    if double_streak >= 3:
                        radius *= 2.0
                        double_streak = 0
                else:
                    double_streak = 0
                    halve_streak = 0
            radius = float(np.clip(radius, 1e-3, 1e6))

        history.append({
            "k": k, "E": E_cur, "accepted": accepted,
            "radius": radius, **{k_: v for k_, v in step_info.items()
                                  if k_ != "retract"},
            "n_overlaps_after_retract": retr["n_overlaps"],
        })

        # Early stop on tiny gradient.
        if step_info["g_T_norm"] < 1e-9:
            break

    return best_pos, float(best_E), {
        "n_steps_done": len(history),
        "final_radius": radius,
        "history_tail": history[-5:],
        "accept_rate": (sum(h["accepted"] for h in history) /
                        max(1, len(history))),
    }
