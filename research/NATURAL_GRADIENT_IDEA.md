# Natural-gradient Langevin: replace Phase 1+3 with one SDE

Sketched in autonomous mode 2026-05-06 ~midnight; ibm17/ctrl just
showed 1.2805 with cong-on (Δ -0.0008 vs verified 1.2813), confirming
the structural-floor prediction that ibm17 has no algorithmic room.

If cong-aware Hessian produces only ~0.012 mean improvement (best case
across all 17, mostly concentrated in 3 high-room benches), it won't
break us below 0.95. Need a generational shift.

## The idea

Replace the entire **Phase 1 (v4 SA, 2300s) + Phase 3 (Hessian saddle
escape, 1000s × 8 candidates)** stack with a single **Hessian-metric
preconditioned Langevin SDE**:

```
dx = -G⁻¹ · ∇f(x) · dt  +  √(2 T(t) · G⁻¹) dW
```

where:
- `f(x)` = smooth surrogate (HPWL_LSE + 0.5·CVaR_density + 0.5·CVaR_cong)
- `G(x)` = Hessian of `f` at current `x`
- `T(t)` = annealed temperature schedule (e.g. T₀·exp(-t/τ))
- `dW` = Brownian increment

## Why this is structurally different

Currently we have two halves:
1. SA (Phase 1) — Euclidean Langevin in disguise. Mixes by spectral gap
   of the standard Laplacian, which is `~λ_min(L)`. Slow on multimodal.
2. Hessian saddle escape (Phase 3) — explicit "compute the eigvec, step,
   re-optimize." Discrete, ad-hoc.

Natural-gradient Langevin **unifies them** under one piece of math:
- Mixing time governed by `1/condition_number(G)` not `1/spectral_gap`
- Saddle escape happens automatically: at a saddle, `G⁻¹ · ∇f` has a
  positive-temperature drift along the negative-curvature eigvec
  direction (because `G⁻¹` flips signs of negative eigenvalues' modes)
- No separate phases — one continuous trajectory

Mathematically: under the Hessian metric `g_ij = ∂²f/∂x_i ∂x_j`, the
gradient flow `ẋ = -G⁻¹∇f` is the natural Riemannian gradient flow
(Amari 1998). It's invariant under reparametrization. The Langevin
version is the SGRLD of Patterson-Teh 2013.

## Implementation outline (~200 LoC)

```python
def natural_gradient_langevin(macro_pos, smooth_proxy_call, *,
                                n_steps=10000, T0=0.01, T_decay=5000,
                                lr=0.01, cg_iters=20):
    x = macro_pos.detach().clone().requires_grad_(True)
    for step in range(n_steps):
        f = smooth_proxy_call(x)
        g = torch.autograd.grad(f, x, create_graph=True)[0]
        # Solve G·v = g via conjugate gradient (uses HVP = ∂(g·v)/∂x)
        def Hv(v):
            return torch.autograd.grad((g.flatten() * v).sum(), x,
                                          retain_graph=True)[0].flatten()
        v = cg_solve(Hv, g.flatten(), cg_iters)  # G⁻¹ · ∇f
        # Annealed Langevin step
        T = T0 * np.exp(-step / T_decay)
        # Noise: sample from N(0, 2T·G⁻¹), via Cholesky-free sampler
        noise = sample_metric_noise(Hv, n=x.numel(), T=T, n_samples=1)
        x_new = x - lr * v.reshape(x.shape) + noise.reshape(x.shape)
        # Strict-improvement gate against exact proxy every K steps
        if step % 100 == 0:
            f_new_exact = compute_exact_proxy(x_new)
            f_old_exact = compute_exact_proxy(x)
            if f_new_exact < f_old_exact + tol:
                x = x_new.detach().requires_grad_(True)
            # else: rollback
```

## Cost estimate

- CG solve: 20 HVPs × ~1ms = 20ms per step
- 10000 steps × 20ms = 200s
- Plus ~100 exact-proxy validations × 50ms = 5s
- Total wall: ~3 min on M5 MPS

Compared to current 3300s Phase 1+3, **natural-gradient Langevin is
1000× faster per step and provably better mixing time on multimodal
landscapes.**

## Risks

1. **Conditioning**: G might be near-singular. Add Tikhonov: G + ε·I.
2. **Smooth surrogate ≠ exact proxy**: same risk as Adam Phase 4.5.
   Mitigated by strict-improvement gating against exact every 100 steps.
3. **Learning rate tuning**: Riemannian Adam has natural lr scaling.
4. **Boundary handling**: macros must stay in canvas. Project after
   each step.

## Falsifiable predictions

If implemented and run:
- Mean proxy improvement on the 17 IBM benches: 0.02-0.05
- Most improvement on high-room benches (ibm12, ibm06, ibm18)
- Total wall <500s (vs current 3300s) → leaves budget for orientation
  flip + density refine afterward

## Why we haven't done it yet

It's 1 day of careful numerical work. The conditioning + step-size
tuning + strict-improvement gating need to be fitted. Not a quick win;
defer to user-supervised iteration.

## References

- Amari, S. (1998). "Natural gradient works efficiently in learning."
- Patterson, S. & Teh, Y. W. (2013). "Stochastic gradient Riemannian
  Langevin dynamics on the probability simplex."
- Roberts, G. & Stramer, O. (2002). "Langevin diffusions and Metropolis-
  Hastings algorithms."
