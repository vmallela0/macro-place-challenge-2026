"""Single-chain PT worker for spawn-safe multiprocessing.

Module-level imports only — no CUDA initialisation here, no torch
multiprocessing context-state, since `spawn` re-imports the module in
a fresh interpreter on each pool process.

Currently unused by the default synchronous PT runner in
_replica_exchange.run_pt. Kept as a stub for future scaling needs.
"""
from __future__ import annotations
import math
from typing import Callable
import numpy as np
from numpy.typing import NDArray


def chain_step_batch(
    state: NDArray,
    energy: float,
    n_steps: int,
    temperature: float,
    seed: int,
    proposal_fn: Callable[[NDArray, np.random.Generator, float], NDArray],
    energy_fn: Callable[[NDArray], float],
) -> tuple[NDArray, float, int, NDArray]:
    """Run `n_steps` Metropolis-Hastings steps in a single chain at
    fixed temperature.

    Returns:
        state, energy : final
        n_accepted    : count of accepted moves
        best_state    : lowest-energy state seen during the batch
    """
    rng = np.random.default_rng(seed)
    accepted = 0
    best_state = state.copy()
    best_energy = energy
    for _ in range(n_steps):
        x_new = proposal_fn(state, rng, temperature)
        E_new = float(energy_fn(x_new))
        log_a = -(E_new - energy) / temperature
        if log_a >= 0.0 or rng.random() < math.exp(log_a):
            state = x_new
            energy = E_new
            accepted += 1
            if E_new < best_energy:
                best_energy = E_new
                best_state = x_new.copy()
    return state, energy, accepted, best_state
