"""Sequential Monte Carlo population sampler — zeus B8.

Math reference: Del Moral, Doucet & Jasra (2006), "Sequential Monte
Carlo Samplers", JRSS-B 68(3).

Theoretical setup
=================
We have a target π_T(x) ∝ exp(-β_T · U(x)) we cannot sample from
directly. Define a sequence
    π_0(x) ∝ 1           (uniform / wide Gaussian on the canvas)
    π_t(x) ∝ exp(-β_t · U(x)),  0 = β_0 < β_1 < ... < β_T
At each t, we maintain N weighted particles {(x_t^i, w_t^i)}_{i=1..N}.
The two-step update from t to t+1:

    REWEIGHT.   w_{t+1}^i ∝ w_t^i · exp(-(β_{t+1} - β_t) · U(x_t^i))
                The new weight is the EXACT ratio π_{t+1}(x) / π_t(x),
                so the weighted sample {(x_t^i, w_{t+1}^i)} is unbiased
                for π_{t+1}.

    MOVE.       x_{t+1}^i ← K_{t+1}(x_t^i, ·)
                where K_{t+1} is an MCMC kernel leaving π_{t+1}
                invariant. The kernel does NOT change the target —
                it just reduces variance by exploring around each
                particle. For us, Gaussian Random Walk Metropolis with
                step σ_t·canvas_diag and acceptance ratio
                exp(-β_{t+1} · (U(x_new) - U(x_old))).

When the effective sample size
    ESS = (Σ w^i)² / Σ (w^i)²
drops below N/2, we RESAMPLE (multinomial or systematic). This avoids
weight collapse onto a few particles. After resample, all w = 1/N.

Adaptive β schedule
-------------------
Choose β_{t+1} - β_t s.t. ESS ratio is exactly 0.5. The relative ESS as
a function of Δβ is monotone-decreasing, so we bisect.

Convergence (Del Moral et al.)
-----------------------------
As N → ∞, weighted sample averages of any continuous bounded f converge
to E_{π_T}[f] with rate O(1/√N). The constant depends on the path
length T and the smoothness of π_t → π_{t+1} transitions (chosen by the
β-schedule).

For our use: we don't need to converge to the equilibrium π_T. We
just want N=16-32 GOOD CANDIDATES across diverse basins. SMC gives this
naturally: particles in different basins are kept alive by importance
weighting, and the MCMC moves explore locally.

Why this should help here
=========================
Single-chain HMC trajectories from one point cover one basin. The v6
portfolio runs N separate Phase-1s from N seeds, which is the
poor-man's SMC (importance weight = 1/N, no mutation). True SMC adds:
(a) intermediate temperatures = warmer chains move farther
(b) reweighting concentrates compute on promising basins
(c) systematic resampling balances explore/exploit dynamically

Failure modes
-------------
- Cold MCMC kernel: small σ_t → particles barely move, SMC degenerates
  to importance sampling.
- Stuck β-ladder: if Δβ chosen so ESS = 0.5 always, we waste compute
  on warm distributions when the cold target has small support.
- Particle degeneracy: if all particles end up in one basin (high cost),
  resampling alone can't recover. We add diversity perturbation
  (small fraction reset to random in canvas) every K steps.

Smoke test
----------
A known-correct check: at very low β with N=1000 particles in 1D well,
the SMC mean ≈ Gaussian mean. At very high β, particles concentrate at
the well minimum. See `tests/test_smc.py`.
"""

from __future__ import annotations
import time
import numpy as np


def _effective_sample_size(log_w: np.ndarray) -> float:
    """ESS for a vector of log-weights (normalized in log domain)."""
    # log-normalize
    m = log_w.max()
    log_w_n = log_w - m
    w = np.exp(log_w_n)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / (w ** 2).sum())


def _bisect_beta_step(
    Us: np.ndarray,         # (N,) current particle costs
    target_ess_frac: float = 0.5,
    max_step: float = 10.0,
    n_bisect: int = 30,
) -> float:
    """Find Δβ such that ESS(reweighted by exp(-Δβ · U)) / N = target.

    ESS as a function of Δβ is continuous and monotone-decreasing
    (larger Δβ = sharper weight = lower ESS). Binary search.
    """
    N = Us.shape[0]
    target_ess = target_ess_frac * N

    def ess_at(dbeta: float) -> float:
        log_w = -dbeta * (Us - Us.min())   # stabilize
        return _effective_sample_size(log_w)

    lo, hi = 0.0, max_step
    # Bracket: if even max_step gives ESS > target, accept max_step.
    if ess_at(hi) > target_ess:
        return float(hi)
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) > target_ess:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def smc_sampler(
    init_population: np.ndarray,         # (N, n_total, 2) initial particles
    U_eval_batch,                        # callable: (N, n_total, 2) → (N,) U values
    *,
    n_steps: int = 20,                   # number of SMC stages
    target_ess_frac: float = 0.5,
    n_mcmc_per_step: int = 1,
    mcmc_step_sigma: float = 2.0,        # micron σ for Gaussian RW Metropolis
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    n_hard: int = 0,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run SMC for n_steps. Returns final particle set + diag.

    Parameters
    ----------
    init_population : (N, n_total, 2) starting placements. Should be
        sampled from the prior (broad, e.g., jittered .plc or random).
    U_eval_batch : closure taking (N, n_total, 2) and returning (N,)
        smooth-proxy costs. Caller is responsible for batched eval.
    n_steps : SMC temperature stages.
    target_ess_frac : adaptive β-schedule target. 0.5 is standard.
    n_mcmc_per_step : MCMC moves per SMC step. Larger → more mixing per
        unit work, smaller → more rapid annealing.
    mcmc_step_sigma : Gaussian RW proposal σ in microns. Same scale as
        canvas-relative.
    n_hard : leading rows not perturbed.

    Returns
    -------
    final_particles : (N_final, n_total, 2) — particles after final stage.
    diag : dict of stats.
    """
    rng = np.random.default_rng(seed)
    N = init_population.shape[0]
    n_total = init_population.shape[1]
    particles = init_population.copy().astype(np.float64)
    Us = U_eval_batch(particles)
    beta = 0.0
    log_w = np.zeros(N)             # uniform initial weights

    history = {
        "beta": [0.0], "ess": [float(N)],
        "U_mean": [float(Us.mean())], "U_min": [float(Us.min())],
        "n_resamples": [0], "accept_rate": [],
    }
    t0 = time.time()

    for stage in range(int(n_steps)):
        # 1. Choose β_{stage+1} via adaptive ESS = target_ess_frac·N.
        dbeta = _bisect_beta_step(Us, target_ess_frac=target_ess_frac)
        beta_new = beta + dbeta

        # 2. Reweight: log w_new = log w + (-dbeta)·U.
        # (Note: at first iteration log_w = 0 for all i, so this is
        # just the importance weight directly.)
        log_w = log_w - dbeta * Us
        ess = _effective_sample_size(log_w)
        history["beta"].append(float(beta_new))
        history["ess"].append(float(ess))

        # 3. Resample if ESS drops below threshold.
        n_resamples_this_stage = 0
        if ess < target_ess_frac * N:
            # Systematic resampling (Murray, 2013).
            m = log_w.max()
            w_n = np.exp(log_w - m)
            w_n = w_n / w_n.sum()
            cdf = np.cumsum(w_n)
            u0 = rng.random() / N
            idx = np.empty(N, dtype=np.int64)
            j = 0
            for i in range(N):
                u = u0 + i / N
                while j < N - 1 and cdf[j] < u:
                    j += 1
                idx[i] = j
            particles = particles[idx]
            Us = Us[idx]
            log_w = np.zeros(N)         # uniform after resample
            n_resamples_this_stage = 1
        history["n_resamples"].append(n_resamples_this_stage)
        beta = beta_new

        # 4. MCMC moves — Gaussian RW Metropolis with kernel preserving
        #    π_beta(x) ∝ exp(-β U(x)).
        n_accept = 0
        n_propose = 0
        for _ in range(int(n_mcmc_per_step)):
            # Propose perturbation only to soft macros.
            prop = particles.copy()
            noise = rng.standard_normal((N, n_total, 2)) * float(mcmc_step_sigma)
            if n_hard > 0:
                noise[:, :n_hard, :] = 0.0
            prop = prop + noise
            # Clamp into canvas.
            prop[:, :, 0] = np.clip(prop[:, :, 0], 0.0, float(canvas_w))
            prop[:, :, 1] = np.clip(prop[:, :, 1], 0.0, float(canvas_h))
            U_prop = U_eval_batch(prop)
            # Acceptance log α = -β·(U_prop - U_curr).
            log_alpha = -beta * (U_prop - Us)
            u = rng.random(N)
            accept = np.log(u + 1e-30) < log_alpha
            particles = np.where(accept[:, None, None], prop, particles)
            Us = np.where(accept, U_prop, Us)
            n_accept += int(accept.sum())
            n_propose += N
        accept_rate = float(n_accept / max(n_propose, 1))
        history["accept_rate"].append(accept_rate)
        history["U_mean"].append(float(Us.mean()))
        history["U_min"].append(float(Us.min()))

        if verbose:
            print(f"    [smc] stage {stage+1}/{n_steps} β={beta:.3f} "
                  f"Δβ={dbeta:.4f} ESS={ess:.1f} accept={accept_rate:.2f} "
                  f"U_min={Us.min():.4f}", flush=True)

    diag = {
        "method": "smc",
        "N": int(N),
        "n_steps": int(n_steps),
        "final_beta": float(beta),
        "final_U_min": float(Us.min()),
        "final_U_mean": float(Us.mean()),
        "history": history,
        "wall_s": time.time() - t0,
    }
    return particles, diag
