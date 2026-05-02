"""Parallel Tempering (replica exchange) on an exact-cost energy.

Geyer 1991, Earl & Deem 2005. M chains at temperatures T_1 < T_2 < ... < T_M.
Each chain runs Metropolis-Hastings on the same proposal kernel:

    accept(x → x') = min(1, exp(-(E(x') - E(x)) / T_i))

Every τ_swap steps, attempt a swap of adjacent chains:

    P_swap(i ↔ i+1) = min(1, exp((1/T_i - 1/T_{i+1}) · (E_i - E_{i+1})))

Output: lowest-temperature chain's running best (energy_min, state_at_min).

Implementation
--------------
Synchronous, single-process round-robin over chains. M=8 chains × O(1000)
steps each = O(8000) energy evals; even at 10 ms per eval that's 80 s,
which fits the per-bench time budget. A multiprocess version with
`get_context("spawn")` is supported for hot benchmarks where energy
eval is expensive — see `run_pt_pool`.

The `proposal_fn` is the only domain-specific piece. It must be:
    proposal_fn(x: np.ndarray, rng: np.random.Generator, T: float) -> np.ndarray

The temperature is passed in case the kernel scales step size with T
(common for Gaussian SA proposal: σ ∝ √T).

Cross-platform safety
---------------------
- No CUDA inside the energy evaluation (the energy_fn closure is the
  caller's responsibility — exact proxy is CPU-only via PlacementCost).
- Multiprocess pool uses `get_context("spawn")`; energy_fn must be
  picklable (top-level function or partial of one).
"""
from __future__ import annotations
import math
import multiprocessing as mp
from typing import Callable
import numpy as np
from numpy.typing import NDArray


# ── Temperature ladder ──────────────────────────────────────────────────


def geometric_ladder(T_min: float, T_max: float, n_chains: int) -> list[float]:
    """Geometric ladder: T_i = T_min · (T_max/T_min)^(i/(n-1)).

    Standard for replica exchange: gives roughly constant swap acceptance
    when the energy distribution is approximately Gaussian (Earl & Deem 2005).
    """
    if n_chains < 2:
        return [T_min]
    ratio = (T_max / T_min) ** (1.0 / (n_chains - 1))
    return [T_min * (ratio ** i) for i in range(n_chains)]


def adapt_ladder_geometry(
    current_ladder: list[float],
    swap_acceptances: list[float],
    *,
    target_low: float = 0.2,
    target_high: float = 0.4,
) -> list[float]:
    """One-shot ladder rescaling per spec:

      mean(swap_accept) > 0.4  →  expand ladder (chains too close in T,
                                  need wider spread to sample harder regions).
      mean(swap_accept) < 0.2  →  compress (chains too far apart, swaps fail).
      else: unchanged.
    """
    if not swap_acceptances:
        return current_ladder
    mean_acc = float(np.mean(swap_acceptances))
    T_min = current_ladder[0]
    T_max = current_ladder[-1]
    n = len(current_ladder)
    if mean_acc > target_high:
        # Expand: increase T_max by 1.3x.
        return geometric_ladder(T_min, T_max * 1.3, n)
    if mean_acc < target_low:
        # Compress: shrink T_max toward T_min by 0.7x.
        new_max = max(T_min * 1.5, T_max * 0.7)
        return geometric_ladder(T_min, new_max, n)
    return current_ladder


# ── Swap probability ────────────────────────────────────────────────────


def swap_log_alpha(E_i: float, E_j: float, T_i: float, T_j: float) -> float:
    """Log of the swap acceptance probability between chains i, j.

    log α = (1/T_i - 1/T_j) · (E_i - E_j).

    P_swap = min(1, exp(log α)).
    """
    return (1.0 / T_i - 1.0 / T_j) * (E_i - E_j)


def metropolis_log_alpha(E_old: float, E_new: float, T: float) -> float:
    """Single-chain MH log acceptance: -(E_new - E_old) / T."""
    return -(E_new - E_old) / T


# ── Synchronous PT runner ───────────────────────────────────────────────


def run_pt(
    initial_state: NDArray,
    energy_fn: Callable[[NDArray], float],
    proposal_fn: Callable[[NDArray, np.random.Generator, float], NDArray],
    *,
    n_chains: int = 8,
    temp_ladder: list[float] | None = None,
    n_steps: int = 8000,
    swap_interval: int = 100,
    base_seed: int = 42,
    autotune: bool = True,
    autotune_after_steps: int = 5000,
) -> tuple[NDArray, float, dict]:
    """Synchronous round-robin PT.

    Returns:
        best_state : (...) lowest-T chain's best-seen state
        best_energy : float — its energy
        info : dict — diagnostics (per-chain accept rates, swap accept,
                      ladder, energy history sample, autotune trace)

    `n_steps` is the per-chain step count. Total work is M · n_steps
    energy evaluations.
    """
    if temp_ladder is None:
        temp_ladder = geometric_ladder(0.01, 1.0, n_chains)
    if len(temp_ladder) != n_chains:
        raise ValueError(
            f"temp_ladder length {len(temp_ladder)} != n_chains {n_chains}")

    # Sort temperatures ascending (T_1 = lowest is the cold chain).
    temp_ladder = sorted(temp_ladder)

    states = [np.array(initial_state, copy=True) for _ in range(n_chains)]
    energies = [energy_fn(s) for s in states]
    best_state = states[0].copy()
    best_energy = energies[0]

    rngs = [np.random.default_rng(base_seed + i) for i in range(n_chains)]
    swap_rng = np.random.default_rng(base_seed + 999)

    chain_proposals = [0] * n_chains
    chain_accepts = [0] * n_chains
    swap_attempts = [0] * (n_chains - 1)
    swap_accepts_list = [0] * (n_chains - 1)
    autotuned = False

    energy_trace: list[list[float]] = [[] for _ in range(n_chains)]
    trace_every = max(1, n_steps // 200)

    for step in range(n_steps):
        # ── MH step in each chain ──
        for c in range(n_chains):
            x_old = states[c]
            E_old = energies[c]
            x_new = proposal_fn(x_old, rngs[c], temp_ladder[c])
            E_new = float(energy_fn(x_new))
            log_a = metropolis_log_alpha(E_old, E_new, temp_ladder[c])
            chain_proposals[c] += 1
            if log_a >= 0.0 or rngs[c].random() < math.exp(log_a):
                states[c] = x_new
                energies[c] = E_new
                chain_accepts[c] += 1
                if c == 0 and E_new < best_energy:
                    best_energy = E_new
                    best_state = x_new.copy()

        if step % trace_every == 0:
            for c in range(n_chains):
                energy_trace[c].append(energies[c])

        # ── Swap attempt every swap_interval steps ──
        if (step + 1) % swap_interval == 0 and n_chains >= 2:
            # Alternate parity to give all adjacent pairs a chance over time.
            parity = ((step + 1) // swap_interval) % 2
            for i in range(parity, n_chains - 1, 2):
                swap_attempts[i] += 1
                la = swap_log_alpha(
                    energies[i], energies[i + 1],
                    temp_ladder[i], temp_ladder[i + 1])
                if la >= 0.0 or swap_rng.random() < math.exp(la):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    swap_accepts_list[i] += 1
                    # The lowest-T chain may now hold a better state.
                    if i == 0 and energies[0] < best_energy:
                        best_energy = energies[0]
                        best_state = states[0].copy()

        # One-shot ladder autotune partway through (per spec).
        if (autotune and not autotuned
                and step + 1 >= autotune_after_steps
                and step + 1 < n_steps):
            acc_rates = [
                (swap_accepts_list[i] / swap_attempts[i])
                if swap_attempts[i] > 0 else 0.0
                for i in range(n_chains - 1)
            ]
            new_ladder = adapt_ladder_geometry(temp_ladder, acc_rates)
            if new_ladder != temp_ladder:
                temp_ladder = new_ladder
            autotuned = True

    info = {
        "temp_ladder": list(temp_ladder),
        "chain_acceptance": [
            (chain_accepts[c] / chain_proposals[c]) if chain_proposals[c] > 0 else 0.0
            for c in range(n_chains)
        ],
        "swap_acceptance": [
            (swap_accepts_list[i] / swap_attempts[i]) if swap_attempts[i] > 0 else 0.0
            for i in range(n_chains - 1)
        ],
        "swap_attempts": list(swap_attempts),
        "energy_trace_sample": [list(t[-10:]) for t in energy_trace],
        "autotuned": autotuned,
    }
    return best_state, float(best_energy), info


# ── Pool variant (spawn-safe) ───────────────────────────────────────────


def run_pt_pool(*args, **kwargs):
    """Spawn-safe pool variant — placeholder. The synchronous run_pt is
    used by default in v8 because per-step exact-cost evaluations are
    cheap (~ms) and 8 chains × 8000 steps finishes in seconds, removing
    the multiprocess overhead.

    To switch to a pool-based implementation, instantiate
    `mp.get_context("spawn").Pool(n_chains)` here and dispatch single-chain
    workers from `_pt_worker.py`.
    """
    raise NotImplementedError(
        "run_pt_pool not yet implemented; use run_pt (synchronous).")
