# Zeus achilles bet portfolio — 12 mechanisms, 17 arms

Branch: `achilles` (off `albania2`).
Each bet has its math documented in the corresponding module's docstring
and verified by a smoke test in `submissions/vmallela_v7/tests/`.

## The 12 mechanisms

| # | Bet name | Inspiration | Module | Test |
|---|----------|-------------|--------|------|
| B1 | Yoshida 4th-order symplectic integrator | Hamiltonian mechanics (Yoshida 1990) | `_subspace_hmc.py` | `test_yoshida.py` |
| B2 | Replica overlap diverse-subset selection | Spin glass (Parisi RSB) | `_subspace_hmc.py` | `test_replica.py` |
| B3 | L1/Lp/Linf sparse cong aggregator | Information geometry | `_smooth_proxy.py` | `test_sparse_cong.py` |
| B4 | Nesterov-ODE RK4 integrator | Su-Boyd-Candès ODE | `_nesterov_ode.py` | `test_nesterov_ode.py` |
| B5 | Diffusion Monte Carlo walkers | Quantum many-body | `_dmc_walker.py` | `test_dmc.py` |
| B6 | JKO/Wasserstein-2 proximal step | Optimal transport | `_jko_step.py` | `test_jko.py` |
| B7 | Gaussian-smoothed free energy | Statistical mechanics | `_free_energy.py` | `test_free_energy.py` |
| B8 | Sequential Monte Carlo (tempered) | Del Moral-Doucet-Jasra | `_smc.py` | `test_smc.py` |
| B9 | RG net-length curriculum (homotopy) | Wilson RG | `_rg_curriculum.py` | `test_rg_curriculum.py` |
| B10 | Catastrophe-fold cubic unfolding | Thom-Arnold singularity theory | `_catastrophe.py` | `test_catastrophe.py` |
| B11 | NEB minimum-energy path | Chemistry (Henkelman-Jónsson) | `_neb.py` | `test_neb.py` |
| B12 | Continuous orientation + discretization | Differential geometry / gauge | `_continuous_orientation.py` | `test_continuous_orientation.py` |

All 13 tests pass (12 bets + 1 rudy_smooth from prior session).

## The 17 portfolio arms

The achilles harness (`scripts/zeus_achilles_ab.sh`) runs 17 env-var
configurations. 12 single-bet arms test mechanisms individually; 5
combination arms stack them.

| Arm | Description | Env flags |
|-----|-------------|-----------|
| `baseline` | Verified 0.9975 config | (default) |
| `yoshida` | B1 only (HMC w/ yoshida4) | `HMC_INTEGRATOR=yoshida4` |
| `replica` | B2 only (over-sample HMC + farthest-point) | `HMC_REPLICA_KEEP=8` |
| `l1_cong` | B3 (L1 cong aggregator) + RUDY | `CONG_AGG=l1` |
| `linf_cong` | B3 (Linf cong aggregator) + RUDY | `CONG_AGG=linf` |
| `nesterov` | B4 (Nesterov-ODE Phase 0) | `PHASE0_OPTIMIZER=nesterov_ode` |
| `dmc` | B5 (DMC walkers) | `HESSIAN_DMC_WALKERS=16` |
| `jko` | B6 (JKO post-Adam refine) | `PHASE0_JKO_STEPS=8` |
| `free_energy` | B7 (Gaussian-smoothed wrapper) | `HESSIAN_FREE_ENERGY=1` |
| `smc` | B8 (SMC sampler) | `HESSIAN_SMC_N=16` |
| `rg` | B9 (RG curriculum) | `PHASE0_RG_CURRICULUM=1` |
| `catastrophe` | B10 (catastrophe-fold candidates) | `HESSIAN_CATASTROPHE_K=4` |
| `neb` | B11 (NEB saddle finder) + HMC pool | `HESSIAN_NEB=1` |
| `yoshida_replica` | B1+B2 combo | yoshida4 + REPLICA_KEEP |
| `dmc_smc` | B5+B8 (two population methods) | DMC + SMC together |
| `rg_nesterov` | B4+B9 (Phase-0 stack) | RG + Nesterov |
| `hmc_full` | B1+B2+B5+B8+B10+B11 (all Hessian-phase) | everything except RUDY |

## Math snippets — what each smoke test proves

- **B1 Yoshida**: leapfrog and yoshida4 produce DIFFERENT trajectories
  on a harmonic oscillator (confirms integrator dispatch).
- **B2 Replica**: greedy farthest-point on `{0, 1, 2, 5, 10}` selects
  `{0, 5, 10}` (min-pairwise = 5, maximal). ✓
- **B3 Sparse cong**:
  - `l1_excess([0.1, 0.9, 1.0, 1.5, 3.0], target=1.0, μ=100)` = 2.5 (= 0.5 + 2.0).
  - Gradient is `[0, 0, 0.5, 1.0, 1.0]` — concentrated above threshold.
  - `lp_excess(..., p=1)` == `l1_excess(...)` (1e-3 numerical agreement).
- **B4 Nesterov-ODE**: converges to ||x||=2e-5 on a convex quadratic
  diag(1, 10) from x_0=(1, 1) in 200 steps. Restart prevents divergence
  on a saddle with eigval (1, -0.5).
- **B5 DMC**: walker population stays bounded ≤ 4·N0 (no exponential
  blowup); concentrates ||x|| → 0 on a quadratic well.
- **B6 JKO**: log-stabilized Sinkhorn preserves marginals to 1e-7;
  uniform sources → row sums of 1/n. JKO step moves macro toward
  gradient target.
- **B7 Free-energy**: passthrough at σ=0 and K=1. At σ=1, K=64 on
  quadratic, F̂(0) ≈ 1.0 matches Laplace formula F = U(x*) + σ²·d/2.
- **B8 SMC**: ESS-bisection finds Δβ s.t. ESS = N/2 exactly. High-β
  particles concentrate at mode (std → 0.04 from 5.0).
- **B9 RG curriculum**: γ_n form matches exp(-L²/(2σ²)) to machine
  precision. Schedule σ(t) is monotone. Per-net bbox computation
  matches manual calc.
- **B10 Catastrophe**: 4-point cubic estimator recovers c to 1e-15.
  Fold formula `t* = -2λ/c`, predicted `U(t*) = 2λ³/(3c²)` match
  numeric solver to 1e-10.
- **B11 NEB**: Henkelman-Jónsson improved tangent finds saddle of
  `(x²-1)²` at x=0 (true 0) with U=1.000 (true 1.0). No spurious
  barrier on a single-well quadratic.
- **B12 Continuous orientation**: -cos(4θ) has minima exactly at
  {0, π/2, π, 3π/2} (value -1) and maxima at {π/4, 3π/4} (value +1).
  R(θ=π/2) pin rotation: (xoff=5, yoff=3) → (-3, 5) — verified.

## Running the portfolio

The autopilot orchestrates everything as a detached PPID=1 process:

```bash
bash scripts/zeus_run_detached.sh achilles_autopilot \
    bash scripts/zeus_achilles_autopilot.sh
```

It performs three stages without intervention:

1. **Stage A — screen**: 17 arms × 3 benches (ibm06/12/15) = 51 full-v7
   placer runs, wave width 16. ~4-6h wall.
2. **Stage B — rank + pick top-4**: mean Δ over the 3 screening benches.
3. **Stage C — full-17 with top-4**: 4 × 17 = 68 full-v7 runs, wave 16.
   ~5-8h wall.

Total wall: ~12-16 hours. Status visible at any time:

```bash
cat ~/zeus_runs/ACHILLES_STATUS.md
bash scripts/zeus_status.sh                 # all detached runs
bash scripts/zeus_status.sh tail achilles_autopilot
```

## Survival guarantee

| event | screen | autopilot | full-17 | this claude |
|-------|:------:|:---------:|:-------:|:-----------:|
| SSH connection drops | ✓ | ✓ | ✓ | ✗ |
| this claude process exits | ✓ | ✓ | ✓ | n/a |
| laptop closes | ✓ | ✓ | ✓ | ✗ |
| reboot of `optimeshr640` | ✗ | ✗ | ✗ | ✗ |

After a connection drop, recover with:
```bash
ssh vedu@optimeshr640
cd ~/vmallela/personal/macro-place-challenge-2026
cat ~/zeus_runs/ACHILLES_STATUS.md
bash scripts/zeus_claude.sh   # tmux-wrapped claude (durable from now on)
```
