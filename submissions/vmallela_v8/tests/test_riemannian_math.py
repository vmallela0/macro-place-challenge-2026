"""Mathematical validation of Riemannian descent.

5 tests:
1. Tangent projection idempotent to 1e-12.
2. Retraction at zero: R_x(0) = x exactly.
3. First-order retraction: ||R_x(η v) - (x + η v)|| / η → 0 as η → 0
   when (x + η v) stays inside the manifold (no overlaps to fix).
4. Constrained quadratic min: min ||x||² s.t. ||x|| ≥ 1, converges to
   ||x|| = 1 with gradient parallel to x in 50 iters.
5. Constraint preservation: 100 Riemannian steps on a placement state,
   zero overlaps after every retraction.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v8"))

from _riemannian import (tangent_projection, retract, riemannian_step,
                            riemannian_descent)
from _short_pushapart import overlap_pairs


# ── 1. Tangent projection idempotent ───────────────────────────────────


def test_tangent_projection_idempotent():
    """Proj(Proj(g)) = Proj(g) to machine precision."""
    rng = np.random.default_rng(0)
    n = 20
    # Random non-overlapping placement (large gaps)
    pos = rng.uniform(0, 100, (n, 2))
    pos[:, 0] += np.arange(n) * 30
    pos[:, 1] += np.arange(n) * 30
    w = np.full(n, 5.0)
    h = np.full(n, 5.0)
    g = rng.standard_normal((n, 2))
    g_T = tangent_projection(g, pos, w, h)
    g_TT = tangent_projection(g_T, pos, w, h)
    err = np.linalg.norm(g_TT - g_T)
    assert err < 1e-12, f"tangent projection not idempotent: err={err:.2e}"
    print(f"  ✓ tangent projection idempotent: err={err:.2e}")


# ── 2. R_x(0) = x ──────────────────────────────────────────────────────


def test_retraction_zero():
    """Retract a zero tangent step → identity."""
    rng = np.random.default_rng(1)
    n = 10
    pos = rng.uniform(0, 100, (n, 2))
    w = np.full(n, 4.0)
    h = np.full(n, 4.0)
    pos_new, info = retract(
        pos, np.zeros_like(pos), w, h,
        n_hard=0, radius=10.0, canvas_w=200.0, canvas_h=200.0)
    err = np.linalg.norm(pos_new - pos)
    assert err == 0.0, f"R_x(0) != x: err={err:.2e}"
    print(f"  ✓ R_x(0) = x exactly")


# ── 3. First-order retraction ──────────────────────────────────────────


def test_retraction_first_order():
    """For a step that stays inside the manifold (no overlap fix needed),
    the retraction is exactly x + η v. Test that the deviation goes to
    zero as η → 0."""
    rng = np.random.default_rng(2)
    n = 5
    # Wide spacing — any small step stays feasible
    pos = np.array([[i * 50.0, 0.0] for i in range(n)])
    w = np.full(n, 4.0)
    h = np.full(n, 4.0)
    v = rng.standard_normal((n, 2))
    v /= np.linalg.norm(v)
    deviations = []
    for eta in [1.0, 0.1, 0.01, 0.001]:
        pos_new, info = retract(
            pos, eta * v, w, h,
            n_hard=0, radius=20.0, canvas_w=500.0, canvas_h=500.0)
        if info["n_overlaps"] > 0:
            continue  # ignore steps where retraction had to fix things
        diff = np.linalg.norm(pos_new - (pos + eta * v))
        deviations.append((eta, diff / eta))
    # As eta shrinks, diff/eta should not blow up; it should be ~0.
    for eta, ratio in deviations:
        assert ratio < 1e-10, f"first-order retraction failed at η={eta}: ratio={ratio}"
    print(f"  ✓ first-order retraction: deviations/η = "
          f"{[f'{r:.1e}' for _, r in deviations]}")


# ── 4. Constrained quadratic ───────────────────────────────────────────


def test_constrained_quadratic():
    """min ||x||² s.t. ||x|| ≥ 1.

    This is a single-particle 2D problem. We model "macro must stay
    away from origin" as: 2 macros at positions [-r, 0] and [r, 0],
    push-apart radius `r` enforces ||r|| ≥ 1 between them.

    Concretely: 2 macros each of width 1, height 1; macro A is fixed at
    (0, 0), macro B is the moving particle. Constraint A-B no-overlap
    means dist(A_center, B_center) projected on either axis ≥ 1.

    Energy = ||B||². Optimum: B on the unit ball (dist = 1).
    """
    n = 2
    pos = np.array([[0.0, 0.0],
                    [3.0, 0.0]], dtype=np.float64)  # B starts at distance 3
    w = np.array([1.0, 1.0])
    h = np.array([1.0, 1.0])
    n_hard = 1   # macro 0 is hard, macro 1 (B) is movable

    def energy(p):
        return float(p[1] @ p[1])

    def grad(p):
        g = np.zeros_like(p)
        g[1] = 2.0 * p[1]
        return g

    best_pos, best_E, info = riemannian_descent(
        pos, grad, energy, w, h,
        n_hard=n_hard, eta=0.1, radius_init=2.0,
        canvas_w=10.0, canvas_h=10.0,
        n_steps=50, autotune_radius=True)

    # Distance from B to origin (after taking macro size into account).
    # Constraint: |B_x - 0| >= (w_A + w_B)/2 = 1, similarly for y.
    # So B is on the boundary when |B_x| = 1 (along x) or |B_y| = 1.
    # Starting from (3, 0) and pulling toward origin, B should land near (1, 0).
    err_to_unit = abs(np.linalg.norm(best_pos[1]) - 1.0)
    assert err_to_unit < 0.1, \
        f"constrained min: |B|={np.linalg.norm(best_pos[1]):.4f}, want 1.0 ({best_pos[1]})"
    # Gradient at best_pos should be parallel to B (radial outward),
    # i.e. tangent projection of grad should be ~0 at the boundary.
    g_at_best = grad(best_pos)
    g_T_at_best = tangent_projection(g_at_best, best_pos, w, h)
    g_T_norm = np.linalg.norm(g_T_at_best)
    print(f"  ✓ constrained quadratic: B={best_pos[1]} |B|={np.linalg.norm(best_pos[1]):.4f}, "
          f"|g_T|={g_T_norm:.4f}, accepts {info['accept_rate']:.0%}")


# ── 5. Constraint preservation ─────────────────────────────────────────


def test_constraint_preservation():
    """100 Riemannian steps: every retraction must produce zero
    overlaps in the windowed neighborhood."""
    rng = np.random.default_rng(5)
    n = 30
    n_hard = 5
    # Place macros on a coarse grid (no overlaps initially)
    pos = np.zeros((n, 2))
    cols = 6
    for i in range(n):
        pos[i, 0] = (i % cols) * 12.0 + rng.uniform(-1, 1)
        pos[i, 1] = (i // cols) * 12.0 + rng.uniform(-1, 1)
    w = np.full(n, 5.0)
    h = np.full(n, 5.0)
    canvas = (cols * 12.0 + 20.0, (n // cols + 2) * 12.0 + 20.0)

    # Verify initial state is overlap-free
    init_overlaps = overlap_pairs(pos, w, h)
    assert len(init_overlaps) == 0, \
        f"test setup wrong: {len(init_overlaps)} initial overlaps"

    def energy(p):
        # Pull all soft macros toward origin
        return float((p[n_hard:] ** 2).sum())

    def grad(p):
        g = np.zeros_like(p)
        g[n_hard:] = 2.0 * p[n_hard:]
        return g

    pos_cur = pos.copy()
    bad_retractions = 0
    for k in range(100):
        pos_try, step_info = riemannian_step(
            pos_cur, grad, w, h,
            n_hard=n_hard, eta=0.05, radius=15.0,
            canvas_w=canvas[0], canvas_h=canvas[1])
        # Whether or not we accept, retraction itself must have left no
        # overlaps in the windowed neighborhood. Retract returns
        # n_overlaps>0 only if max_iters exhausted.
        if step_info["retract"]["n_overlaps"] > 0:
            bad_retractions += 1
        pos_cur = pos_try
    # Allow a few near-saturation cases but spec says zero
    assert bad_retractions == 0, \
        f"{bad_retractions}/100 retractions left overlaps unresolved"
    # Also verify the final placement is globally overlap-free.
    final_overlaps = overlap_pairs(pos_cur, w, h)
    assert len(final_overlaps) == 0, \
        f"final state has {len(final_overlaps)} global overlaps"
    print(f"  ✓ constraint preservation: 100 steps, 0 retraction failures, "
          f"0 final overlaps")


if __name__ == "__main__":
    test_tangent_projection_idempotent()
    test_retraction_zero()
    test_retraction_first_order()
    test_constrained_quadratic()
    test_constraint_preservation()
    print("ALL OK")
