"""Granular tap-and-settle physics init (user's idea #8).

Pure numpy physics simulation:
  - Repulsive force on overlapping macros (radial, inverse-distance)
  - Attractive force on net neighbors (toward centroid)
  - Periodic "taps" — inject velocity noise to break local equilibria

Not using incremental evaluator — just a fast 2D particle sim.
After convergence, feed result to legalization tournament.

Serves as a diverse init generator.
"""
import numpy as np


def tap_and_settle(init_positions, macro_sizes, canvas_w, canvas_h, macro_nets,
                   net_macros, movable_mask, n_hard,
                   n_steps=2000, tap_every=200, tap_magnitude=2.0,
                   attract_k=0.02, repel_k=5.0, damping=0.9,
                   seed=42, verbose=False):
    """Run granular physics on hard macros.

    Returns positions after n_steps of simulation. Overlaps may remain;
    legalize downstream.
    """
    rng = np.random.RandomState(seed)
    n = init_positions.shape[0]
    pos = init_positions.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    half_w = macro_sizes[:, 0] / 2
    half_h = macro_sizes[:, 1] / 2

    # Pre-build hard net neighbors (ignore soft — this is hard-only init)
    hard_net_neighbors = [[] for _ in range(n_hard)]
    for nid in range(len(net_macros)):
        hard_in_net = [m for m in net_macros[nid] if 0 <= m < n_hard]
        for a in hard_in_net:
            for b in hard_in_net:
                if a != b and b not in hard_net_neighbors[a]:
                    hard_net_neighbors[a].append(b)

    for step in range(n_steps):
        forces = np.zeros_like(pos)

        # Net-attractive force: pull toward net-neighbor centroid
        for i in range(n_hard):
            if not movable_mask[i] or not hard_net_neighbors[i]:
                continue
            nbr = hard_net_neighbors[i]
            cx = np.mean(pos[nbr, 0])
            cy = np.mean(pos[nbr, 1])
            forces[i, 0] += attract_k * (cx - pos[i, 0])
            forces[i, 1] += attract_k * (cy - pos[i, 1])

        # Repulsive force: push apart overlapping macros
        # Vectorised pairwise (N^2, fine for N<500)
        dx = pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T  # (n_hard, n_hard)
        dy = pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T
        sep_x = half_w[:n_hard, None] + half_w[None, :n_hard]
        sep_y = half_h[:n_hard, None] + half_h[None, :n_hard]
        # Overlap amount (positive = overlap)
        ov_x = np.maximum(0, sep_x - np.abs(dx))
        ov_y = np.maximum(0, sep_y - np.abs(dy))
        ov = ov_x * ov_y
        np.fill_diagonal(ov, 0)
        # Repulsive direction
        signx = np.sign(dx)
        signy = np.sign(dy)
        # Handle zero-difference by randomizing
        signx[np.abs(dx) < 1e-6] = (rng.rand(*signx[np.abs(dx) < 1e-6].shape) - 0.5) * 2
        signy[np.abs(dy) < 1e-6] = (rng.rand(*signy[np.abs(dy) < 1e-6].shape) - 0.5) * 2
        # Force is overlap magnitude × direction; sum over j for each i
        fx = np.sum(repel_k * ov * signx, axis=1)
        fy = np.sum(repel_k * ov * signy, axis=1)
        forces[:n_hard, 0] += fx
        forces[:n_hard, 1] += fy

        # Periodic tap
        if step > 0 and step % tap_every == 0:
            jitter = rng.randn(n_hard, 2) * tap_magnitude
            vel[:n_hard] += jitter

        # Integrate (Verlet-like)
        vel = damping * vel + forces * 0.5
        pos += vel * 0.1

        # Canvas clamp
        pos[:n_hard, 0] = np.clip(pos[:n_hard, 0], half_w[:n_hard], canvas_w - half_w[:n_hard])
        pos[:n_hard, 1] = np.clip(pos[:n_hard, 1], half_h[:n_hard], canvas_h - half_h[:n_hard])

        if verbose and step % 500 == 0:
            # Count overlaps
            np.fill_diagonal(ov, 0)
            n_ov = int(np.sum(ov > 0) / 2)
            print(f"    tap_settle step {step}: overlaps={n_ov}")

    return pos
