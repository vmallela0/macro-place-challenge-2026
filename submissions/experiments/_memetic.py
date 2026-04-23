"""Memetic crossover between solutions from different restart basins.

Currently the parallel-restart phase runs 15 workers from perturbed starts,
and we keep only the single best. This discards diversity.

Memetic crossover: given several good solutions, combine them by taking
macros from whichever parent has the better local cost signature (via net
adjacency), legalize the hybrid, refine with a short CD.

The hypothesis: if parent A is good in the upper-left and parent B is good
in the lower-right, the crossover takes the good regions from each and
produces a child better than either.
"""
import time
import numpy as np
import random


def _hybrid_by_region(parent_a, parent_b, benchmark, split_frac=0.5):
    """Take left-portion from A and right-portion from B (or top/bottom)."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    child = parent_a.copy()
    # Decide split axis randomly
    if random.random() < 0.5:
        # Vertical split (by x)
        cutoff = cw * split_frac
        for i in range(n_hard):
            if parent_b[i, 0] > cutoff:
                child[i] = parent_b[i]
    else:
        cutoff = ch * split_frac
        for i in range(n_hard):
            if parent_b[i, 1] > cutoff:
                child[i] = parent_b[i]
    return child


def _hybrid_by_cluster(parent_a, parent_b, benchmark, incr_eval, frac=0.5):
    """Take macros from A or B by connectivity cluster.

    Builds clusters of ~5 connected macros each, and for each cluster,
    takes all positions from the same parent (picked at random).
    Preserves intra-cluster relative positions, which matters for WL.
    """
    n_hard = benchmark.num_hard_macros

    adj = [set() for _ in range(n_hard)]
    for nid in range(incr_eval.n_nets):
        hard_in_net = [m for m in incr_eval.net_macros[nid] if 0 <= m < n_hard]
        for a in hard_in_net:
            for b in hard_in_net:
                if a != b:
                    adj[a].add(b)
    degree = [len(adj[i]) for i in range(n_hard)]

    assigned = [False] * n_hard
    clusters = []
    for start in sorted(range(n_hard), key=lambda i: -degree[i]):
        if assigned[start]:
            continue
        cluster = [start]
        assigned[start] = True
        queue = [start]
        while queue and len(cluster) < 5:
            node = queue.pop(0)
            for nbr in sorted(adj[node], key=lambda x: -degree[x]):
                if not assigned[nbr] and len(cluster) < 5:
                    cluster.append(nbr)
                    assigned[nbr] = True
                    queue.append(nbr)
        clusters.append(cluster)

    child = parent_a.copy()
    for cluster in clusters:
        if random.random() < frac:
            for i in cluster:
                child[i] = parent_b[i]
    return child


def crossover_candidates(parents, benchmark, incr_eval, n_candidates=6):
    """Generate several hybrid candidates from a list of parents.

    parents: list of [n_hard, 2] np.float64 position arrays
    Returns list of candidate positions (may contain overlaps!)
    """
    if len(parents) < 2:
        return []

    candidates = []
    rng = random.Random(777)
    for _ in range(n_candidates):
        # Pick two distinct parents (bias toward better: parents[0] is best)
        a_idx = rng.choices(range(len(parents)), weights=[1.0 / (1 + i) for i in range(len(parents))])[0]
        b_idx = rng.choices(range(len(parents)), weights=[1.0 / (1 + i) for i in range(len(parents))])[0]
        while b_idx == a_idx:
            b_idx = rng.randrange(len(parents))
        a = parents[a_idx]
        b = parents[b_idx]

        # Half region, half cluster
        if rng.random() < 0.5:
            child = _hybrid_by_region(a, b, benchmark, split_frac=rng.uniform(0.35, 0.65))
        else:
            child = _hybrid_by_cluster(a, b, benchmark, incr_eval, frac=rng.uniform(0.3, 0.7))
        candidates.append(child)

    return candidates
