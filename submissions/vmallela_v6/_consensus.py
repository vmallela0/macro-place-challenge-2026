"""Consensus warm-start (T3.4): ensemble distillation across portfolio workers.

Idea
----
Given N portfolio placements with their proxy costs, compute a per-macro
"consensus position" as the trimmed-mean of the top-K cheapest placements.
This averages out per-seed noise — extreme placements (a macro stuck in a
corner because one worker's RNG happened to land there) get smoothed
away, while consistently-good positions (where most top workers agree)
are preserved. The consensus is then push-apart + legalize'd to fix any
overlaps introduced by averaging, run through a final exact-cost CD
refinement, and compared against the portfolio min.

Why this matters for the OpenROAD downstream (Tier-2 of the competition)
------------------------------------------------------------------------
The Tier-2 evaluation runs the placement through OpenROAD for actual
synthesis/place/route on NG45 and reports WNS / TNS / Area. The proxy
cost we optimize is a smooth approximation of these metrics — not a
perfect predictor. A placement that scores marginally better on the
proxy by exploiting a per-seed pathology (e.g., one macro pushed to an
extreme corner) is likely to underperform on OpenROAD vs a "median pose"
that 15 of 16 portfolio workers agree on. Trimmed-mean across 16 seeds
is a Bayesian way to find that median pose.

Algorithm
---------
1. Sort portfolio placements by proxy cost ascending. Keep the top
   `k_best`.
2. Per macro i: compute trimmed-mean of x and y across those k_best
   placements (drop top/bottom `trim_frac` to clip outliers).
3. The result may have overlaps (averaging two valid placements does
   NOT preserve non-overlap). Run push-apart + legalize to fix.
4. Run one final CD pass on the legalized consensus with the GPU-CD
   if available, else CPU CD.
5. Validate: if cost(consensus_refined) < cost(portfolio_min), return
   the consensus; else return the portfolio min.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def trimmed_mean_per_macro(placements, k_best=16, trim_frac=0.2):
    """Per-macro trimmed mean of (x, y) over the top-k_best placements.

    Parameters
    ----------
    placements : list of (n_macros, 2) numpy float64 arrays, sorted ascending
        by their proxy cost (caller's responsibility to sort).
    k_best : how many of the top placements to include in the consensus.
    trim_frac : fraction of high/low values to trim per macro per axis
        before averaging. trim_frac=0.2 with k_best=16 trims top-3 and
        bottom-3 along each axis, averages the middle 10.

    Returns
    -------
    consensus : (n_macros, 2) float64 array
    """
    if len(placements) == 0:
        raise ValueError("placements is empty")
    k = min(k_best, len(placements))
    top = np.stack(placements[:k], axis=0)  # (k, n_macros, 2)
    n_trim = max(0, int(np.floor(k * trim_frac)))
    if 2 * n_trim >= k:
        n_trim = max(0, (k - 1) // 2)

    # Sort along the workers axis per (macro, axis).
    sorted_top = np.sort(top, axis=0)  # ascending
    if n_trim > 0:
        kept = sorted_top[n_trim:k - n_trim]
    else:
        kept = sorted_top
    consensus = kept.mean(axis=0)
    return consensus


def _refine_and_return(start_pos, start_cost, sorted_placements, sorted_costs,
                       benchmark, plc, v1, refine_max_time,
                       use_gpu_refine, verbose, label, n_hard):
    """Run CD refinement on `start_pos` and return whichever of (refined,
    portfolio_min) is cheaper. `start_pos` is hard-only (n_hard, 2)."""
    import time
    incr = v1.IncrementalEvaluator(v1._load_plc(benchmark.name), benchmark)
    # Stitch portfolio_min's softs onto start_pos's hards so the refine
    # CD operates on the correct full state (the soft positions are what
    # the portfolio min worker found; the new hards came from graft or
    # trimmed-mean).
    portfolio_min_full = sorted_placements[0]
    if portfolio_min_full.shape[0] > n_hard:
        full = portfolio_min_full.copy()
        full[:n_hard] = start_pos[:n_hard]
        _sync_full_placement(incr, full)
    else:
        incr.sync_positions(start_pos)
    legal_cost = float(incr.get_proxy_cost())
    if verbose:
        print(f"  [consensus.{label}] pre-refine cost: {legal_cost:.6f}",
              flush=True)

    t_refine = time.time()
    refined_pos = start_pos.copy()
    refined_cost = legal_cost
    if use_gpu_refine:
        try:
            from _torch_eval import TorchBatchEvaluator
            from _gpu_cd import gpu_mass_cd
            gpu = TorchBatchEvaluator(incr, benchmark)
            refined_pos, refined_cost = gpu_mass_cd(
                start_pos.copy(), benchmark, plc,
                incr_eval=incr, gpu_eval=gpu,
                max_time=refine_max_time, K=32,
                sa_T0=0.00005, sa_cooling=0.9995, seed=999)
        except Exception as e:
            if verbose:
                print(f"  [consensus.{label}] GPU refine err: {e}, "
                      f"fallback to CPU", flush=True)
            refined_pos, refined_cost = v1._coord_descent(
                start_pos.copy(), benchmark, plc,
                max_time=refine_max_time, incr_eval=incr,
                sa_T0=0.00005, sa_cooling=0.9995, sa_rng_seed=999)
    else:
        refined_pos, refined_cost = v1._coord_descent(
            start_pos.copy(), benchmark, plc,
            max_time=refine_max_time, incr_eval=incr,
            sa_T0=0.00005, sa_cooling=0.9995, sa_rng_seed=999)
    refined_cost = float(refined_cost)
    if verbose:
        print(f"  [consensus.{label}] refined cost: {refined_cost:.6f}  "
              f"({time.time()-t_refine:.1f}s)", flush=True)

    portfolio_min_cost = sorted_costs[0]
    portfolio_min_pos = sorted_placements[0]
    if refined_cost < portfolio_min_cost - 1e-7:
        if verbose:
            print(f"  [consensus.{label}] WIN: {refined_cost:.6f} < "
                  f"{portfolio_min_cost:.6f} (Δ {portfolio_min_cost-refined_cost:+.4f})",
                  flush=True)
        if portfolio_min_pos.shape[0] > n_hard:
            full = portfolio_min_pos.copy()
            full[:n_hard] = refined_pos[:n_hard]
            return full, refined_cost, label
        return refined_pos, refined_cost, label
    if verbose:
        print(f"  [consensus.{label}] FALLBACK to portfolio min: "
              f"{portfolio_min_cost:.6f} <= {refined_cost:.6f}", flush=True)
    return portfolio_min_pos, portfolio_min_cost, "portfolio_min"


def _sync_full_placement(incr_eval, full_placement):
    """Sync incr_eval to a full (hard + soft) placement. The standard
    `sync_positions(hard_pos)` only updates the n_hard hard positions and
    leaves the soft positions at whatever they were initialized to (the
    initial benchmark values). For consensus to compare against portfolio
    workers' actual states, we MUST also sync the soft positions —
    otherwise graft optimizes against a state that has the worker's hard
    positions but the initial soft positions, and the resulting moves
    don't generalize back to the worker's actual state.
    """
    n_total = full_placement.shape[0]
    n_hard = incr_eval.n_hard
    full = np.asarray(full_placement, dtype=np.float32)
    incr_eval.macro_pos[:n_hard] = full[:n_hard]
    if n_total > n_hard:
        incr_eval.macro_pos[n_hard:n_total] = full[n_hard:n_total]
    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()


def per_macro_graft(portfolio_min, sorted_placements, sizes,
                    incr_eval, k_best=16,
                    *, n_candidates_per_macro=4):
    """Robust alternative to trimmed-mean averaging.

    Start from `portfolio_min` (full placement, hard + soft) and, per
    hard macro, test substituting each of several "consensus-derived"
    candidate positions:
      - per-axis median of top-k_best
      - per-axis trimmed mean of top-k_best
      - the macro's position from the second-best worker (in case top-1
        had a per-seed pathology)
    Accept any substitution that strictly improves the proxy cost (CPU
    incremental evaluator). Result is by construction `<= portfolio_min`.

    Returns: (grafted_pos_hard, grafted_cost, n_grafted)
    """
    n_hard = len(sizes)
    # Sync FULL placement (hard + soft) so the cost measurements match
    # what compute_proxy_cost would report on portfolio_min.
    _sync_full_placement(incr_eval, portfolio_min)
    pos = np.asarray(incr_eval.macro_pos[:n_hard], dtype=np.float64).copy()
    current_cost = float(incr_eval.get_proxy_cost())

    k = min(k_best, len(sorted_placements))
    top = np.stack([p[:n_hard] for p in sorted_placements[:k]],
                   axis=0)  # (k, n_hard, 2)
    median_pos = np.median(top, axis=0)
    n_trim = max(0, int(np.floor(k * 0.2)))
    if 2 * n_trim < k:
        sorted_top = np.sort(top, axis=0)
        if n_trim > 0:
            kept = sorted_top[n_trim:k - n_trim]
        else:
            kept = sorted_top
        trim_pos = kept.mean(axis=0)
    else:
        trim_pos = median_pos

    # Per-macro candidates
    n_grafted = 0
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    cw = float(incr_eval.cw)
    ch = float(incr_eval.ch)
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    gap = 0.05

    for m in range(n_hard):
        candidates = []
        candidates.append((float(median_pos[m, 0]), float(median_pos[m, 1])))
        candidates.append((float(trim_pos[m, 0]), float(trim_pos[m, 1])))
        if k >= 2:
            candidates.append((float(top[1, m, 0]), float(top[1, m, 1])))
        if k >= 3:
            candidates.append((float(top[2, m, 0]), float(top[2, m, 1])))

        best_dx = best_dy = None
        best_c = current_cost
        for nx, ny in candidates:
            nx = float(np.clip(nx, half_w[m], cw - half_w[m]))
            ny = float(np.clip(ny, half_h[m], ch - half_h[m]))
            if abs(nx - pos[m, 0]) < 1e-3 and abs(ny - pos[m, 1]) < 1e-3:
                continue
            ddx = np.abs(nx - pos[:, 0])
            ddy = np.abs(ny - pos[:, 1])
            ov = (ddx < sep_x[m] + gap) & (ddy < sep_y[m] + gap)
            ov[m] = False
            if ov.any():
                continue
            cost = float(incr_eval.move_macro(m, nx, ny))
            if cost < best_c - 1e-7:
                best_c = cost
                best_dx, best_dy = nx, ny
            incr_eval.undo_move()

        if best_dx is not None:
            incr_eval.move_macro(m, best_dx, best_dy)
            pos[m, 0] = best_dx
            pos[m, 1] = best_dy
            current_cost = best_c
            n_grafted += 1

    return pos, current_cost, n_grafted


def consensus_warm_start(placements, costs, benchmark, plc,
                         *, k_best=16, trim_frac=0.2,
                         refine_max_time=180.0,
                         use_gpu_refine=True,
                         mode="auto",
                         verbose=False):
    """Compute a consensus warm-start, refine, and return the better of
    consensus-refined vs portfolio-min.

    Parameters
    ----------
    placements : list of (n_macros, 2) numpy float64 arrays — all portfolio
        worker results (any order; sorted internally).
    costs : list of float — proxy_cost for each placement (overlap-rejected
        placements should be excluded BEFORE calling this function, or
        marked with cost = +inf).
    benchmark : Benchmark object.
    plc : PlacementCost (passed to the refinement CD).
    k_best : how many top placements to consensus.
    trim_frac : trimmed-mean trim fraction (top+bottom).
    refine_max_time : seconds budget for the final CD refinement pass.
    use_gpu_refine : if True, use gpu_mass_cd for refinement (with CPU CD
        fallback). If False, only CPU CD.

    Returns
    -------
    (best_pos, best_cost, source_label)
    """
    import importlib.util
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE.parent / "vmallela"))
    spec = importlib.util.spec_from_file_location(
        "_v1_for_consensus", str(HERE.parent / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)

    # Step 1+2: trimmed-mean consensus. Sort and pick top-k.
    valid_pairs = [(c, p) for c, p in zip(costs, placements)
                   if c is not None and np.isfinite(c)]
    if len(valid_pairs) == 0:
        raise RuntimeError("no valid placements for consensus")
    valid_pairs.sort(key=lambda x: x[0])
    sorted_placements = [p for _, p in valid_pairs]
    sorted_costs = [c for c, _ in valid_pairs]

    if verbose:
        print(f"  [consensus] {len(sorted_placements)} placements, "
              f"costs (top 5): {[f'{c:.4f}' for c in sorted_costs[:5]]}",
              flush=True)

    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)

    # Mode selection. "auto" = try graft first (never worse than min);
    # if graft accepts no moves, fall through to trimmed-mean refinement.
    # "graft" = graft only. "trimmed_mean" = trimmed-mean only.
    if mode in ("auto", "graft"):
        incr_graft = v1.IncrementalEvaluator(v1._load_plc(benchmark.name),
                                             benchmark)
        portfolio_min_pos = sorted_placements[0]
        graft_pos, graft_cost, n_grafted = per_macro_graft(
            portfolio_min_pos, sorted_placements, sizes, incr_graft,
            k_best=k_best)
        if verbose:
            print(f"  [consensus.graft] {n_grafted} per-macro grafts "
                  f"accepted; cost {sorted_costs[0]:.6f} -> {graft_cost:.6f}",
                  flush=True)
        if mode == "graft":
            # Refine the graft via CD then return.
            return _refine_and_return(graft_pos, graft_cost, sorted_placements,
                                      sorted_costs, benchmark, plc, v1,
                                      refine_max_time, use_gpu_refine, verbose,
                                      label="graft", n_hard=n_hard)
        if n_grafted == 0:
            # No improvements found via graft; fall through to trimmed-mean
            # refinement (which has its own fallback to portfolio_min).
            pass
        else:
            # Graft made progress. Use it as the warm-start for the refine.
            return _refine_and_return(graft_pos, graft_cost, sorted_placements,
                                      sorted_costs, benchmark, plc, v1,
                                      refine_max_time, use_gpu_refine, verbose,
                                      label="graft", n_hard=n_hard)

    consensus = trimmed_mean_per_macro(sorted_placements,
                                       k_best=k_best, trim_frac=trim_frac)
    consensus_hard = consensus[:n_hard].astype(np.float64)

    # Trimmed-mean path: legalize the averaged consensus, refine, return.
    pushed = v1._push_apart(consensus_hard, benchmark,
                            max_iters=300, damping=0.4)
    legal = v1._legalize(pushed, benchmark, order_type=0, step_mult=0.05)
    refined_init = v1._refine_toward_initial(legal, consensus_hard, benchmark)
    return _refine_and_return(refined_init, None, sorted_placements,
                              sorted_costs, benchmark, plc, v1,
                              refine_max_time, use_gpu_refine, verbose,
                              label="trimmed_mean", n_hard=n_hard)
