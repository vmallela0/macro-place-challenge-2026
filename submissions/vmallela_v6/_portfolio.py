"""Multi-process portfolio runner.

Runs N independent placer workers in parallel via `multiprocessing`, each
with a different seed. Returns the best (min proxy_cost) result across all
workers, validated.

The core idea: v4 leaves 17 cores idle on a 16-core grader (PARALLEL_WORKERS=0
in `submissions/vmallela_v2/run.sh`). Per the v4 HANDOFF.md, multi-worker
portfolio is the highest-EV unspent lever (-0.005 to -0.015 estimated). This
file wires it.

We also mix in one GPU-augmented worker that uses TorchBatchEvaluator for an
extra hard-CD phase. The torch context is per-process; this isolates the GPU
client so that GPU exceptions on one worker don't take down the others. Backend
is auto-selected (cuda > mps > cpu), so the same code runs on the grader's
RTX 6000 Ada (CUDA) and the M5 Pro dev box (MPS).
"""
from __future__ import annotations
import os
import sys
import time
import math
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch


def _worker_v4_with_seed(args):
    """A worker that runs the full v4 pipeline (push-apart → legalize →
    hard CD → soft cycles → escape basin) for one benchmark with a given
    seed. Returns the (best_pos_bytes, n_hard, best_cost) tuple.

    Runs in a subprocess.  Reloads everything from scratch — no shared
    state.  Caps its own time at `time_budget` seconds.
    """
    (bench_path, time_budget, seed, use_gpu, log_prefix) = args
    # Restrict child-process BLAS threads to 1 to avoid contention.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PLACER_TOTAL_BUDGET"] = str(time_budget)

    # Late imports (so subprocess gets clean state).
    from pathlib import Path as _Path
    ROOT = _Path(bench_path).resolve().parents[3]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

    import importlib.util
    # Use the v2 placer (= v4 logic) as the workhorse; only the hard-CD
    # phase gets a GPU swap when use_gpu=True.
    spec = importlib.util.spec_from_file_location(
        "_v2_placer", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OptimalPlacer = mod.OptimalPlacer

    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost

    # If GPU worker, swap _coord_descent at module level for the hard CD path.
    if use_gpu:
        try:
            from _gpu_cd import gpu_mass_cd
            from _torch_eval import TorchBatchEvaluator
            # Access v1 evaluator+_coord_descent through the v2 placer's module
            # references and patch the hard-CD function. We add a wrapper that
            # tries gpu_mass_cd first, then falls back to the original CPU CD
            # for the same time budget if the GPU path raises.
            v1 = mod._mod  # exec_module re-imported v1 placer at module load
            orig_cd = v1._coord_descent

            def _gpu_cd_wrapper(pos_np, benchmark, plc_eval, max_time=3000,
                                incr_eval=None, sa_T0=None, sa_cooling=0.9995,
                                sa_rng_seed=None):
                # Use GPU for the bulk; fall back to CPU CD for fine-tuning
                # in the last 25% of the budget (small deltas are where the
                # CPU lattice still wins).
                try:
                    if incr_eval is None:
                        return orig_cd(pos_np, benchmark, plc_eval,
                                       max_time=max_time,
                                       sa_T0=sa_T0, sa_cooling=sa_cooling,
                                       sa_rng_seed=sa_rng_seed)
                    gpu = TorchBatchEvaluator(incr_eval, benchmark)
                    gpu_budget = max_time * 0.7
                    cpu_budget = max_time - gpu_budget - 1.0
                    pos1, _c = gpu_mass_cd(
                        pos_np.copy(), benchmark, plc_eval,
                        incr_eval=incr_eval, gpu_eval=gpu,
                        max_time=gpu_budget, K=32,
                        sa_T0=sa_T0, sa_cooling=sa_cooling,
                        seed=(sa_rng_seed or 0))
                    if cpu_budget <= 1.0:
                        return pos1, _c
                    return orig_cd(pos1, benchmark, plc_eval,
                                   max_time=cpu_budget,
                                   incr_eval=incr_eval,
                                   sa_T0=sa_T0, sa_cooling=sa_cooling,
                                   sa_rng_seed=sa_rng_seed)
                except Exception as e:
                    print(f"{log_prefix}  GPU path err: {e}, fallback to CPU",
                          flush=True)
                    return orig_cd(pos_np, benchmark, plc_eval,
                                   max_time=max_time, incr_eval=incr_eval,
                                   sa_T0=sa_T0, sa_cooling=sa_cooling,
                                   sa_rng_seed=sa_rng_seed)

            v1._coord_descent = _gpu_cd_wrapper
            mod._coord_descent = _gpu_cd_wrapper

            # T1.2 (Hungarian LNS repair) was explored and killed by smoke
            # test — see _hungarian_lns.py docstring and EXPERIMENTS.md.
            # On ibm10 at 300s, v4 greedy LNS reached 1.272 while Hungarian
            # reached 1.298 (96% infeasibility on dense layouts). v4 LNS is
            # retained on every worker (CPU and GPU). The Hungarian module
            # ships for reference / future revival on sparse benchmarks.
        except Exception as e:
            print(f"{log_prefix}  GPU init err: {e}, no-GPU fallback", flush=True)
            use_gpu = False

    bench = Benchmark.load(bench_path)
    placer = OptimalPlacer(seed=seed)
    t0 = time.time()
    placement = placer.place(bench)
    elapsed = time.time() - t0

    # Validate proxy cost via official PlacementCost (matches grader).
    plc = mod._load_plc(bench.name)
    r = compute_proxy_cost(placement, bench, plc)
    cost = float(r["proxy_cost"])
    overlaps = int(r["overlap_count"])

    # Convert to bytes for IPC return.
    np_pos = placement.cpu().numpy()
    return (np_pos.tobytes(), np_pos.shape, np_pos.dtype.str,
            cost, overlaps, elapsed, seed, use_gpu)


def run_portfolio(bench_path: str, *, total_budget: int = 3300,
                  n_workers: int = 8, gpu_workers: int = 1,
                  base_seed: int = 42, log_prefix: str = "",
                  apply_consensus: bool = True,
                  consensus_refine_budget: int = 180,
                  consensus_k_best: int = 16,
                  consensus_trim_frac: float = 0.2):
    """Run a multi-process portfolio of placer workers.

    Parameters
    ----------
    bench_path : str
        Path to the .pt benchmark file.
    total_budget : int
        Wall-clock seconds available for the WHOLE portfolio. Each worker
        gets the full `total_budget` (workers run in parallel).
    n_workers : int
        Total number of worker processes.
    gpu_workers : int
        Of the n_workers, how many use the GPU-augmented hard-CD phase.
        Remaining workers use the pure-CPU v4 pipeline.
    base_seed : int
        First worker uses base_seed, second uses base_seed+1, etc.
    apply_consensus : if True, after the portfolio finishes, compute a
        trimmed-mean consensus warm-start across the top `consensus_k_best`
        valid placements, push-apart + legalize + refine, and return the
        better of (consensus-refined, portfolio-min). T3.4. Robust against
        per-seed pathologies that score well on the proxy but pathologically
        on OpenROAD (Tier-2 of the competition).
    consensus_refine_budget : seconds for the post-consensus CD refinement.
        Should be 5-10% of total_budget.
    consensus_k_best : how many top portfolio placements to consensus.
        Effective k is min(k_best, n_valid_workers).
    consensus_trim_frac : trimmed-mean trim fraction (top+bottom).
    """
    n_cpu_workers = max(0, n_workers - gpu_workers)
    args = []
    for w in range(n_workers):
        seed = base_seed + w
        use_gpu = w < gpu_workers
        prefix = f"{log_prefix}[w{w}{'G' if use_gpu else 'C'}-s{seed}]"
        args.append((bench_path, total_budget, seed, use_gpu, prefix))

    print(f"{log_prefix}portfolio: {n_workers} workers "
          f"({gpu_workers} GPU + {n_cpu_workers} CPU), seeds "
          f"{base_seed}..{base_seed + n_workers - 1}, budget "
          f"{total_budget}s", flush=True)

    t0 = time.time()
    results = []
    if n_workers == 1:
        # Inline run for debugging
        results = [_worker_v4_with_seed(args[0])]
    else:
        # Use 'spawn' to ensure clean process state for MLX/torch.
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_v4_with_seed, args)
    elapsed = time.time() - t0

    # Pick best valid (overlap-free) by cost.
    best = None
    for r in results:
        pos_bytes, shape, dtype_str, cost, overlaps, t, seed, used_gpu = r
        tag = "GPU" if used_gpu else "CPU"
        ok = "VALID" if overlaps == 0 else f"INVALID({overlaps})"
        print(f"{log_prefix}  worker seed={seed} {tag}: cost={cost:.6f} "
              f"{ok} time={t:.0f}s", flush=True)
        if overlaps != 0:
            continue
        if best is None or cost < best[3]:
            best = r

    if best is None:
        print(f"{log_prefix}portfolio: NO VALID workers (all overlaps)",
              flush=True)
        # Return the lowest-cost overall (even if invalid) as a fallback.
        results.sort(key=lambda r: r[3])
        best = results[0]
    pos_bytes, shape, dtype_str, cost, overlaps, t, seed, used_gpu = best
    pos_np = np.frombuffer(pos_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    print(f"{log_prefix}portfolio: BEST(min) seed={seed} cost={cost:.6f} "
          f"overlaps={overlaps} elapsed={elapsed:.0f}s", flush=True)

    # ---- T3.4 consensus warm-start (optional) ----
    if apply_consensus:
        valid_results = []
        for r in results:
            (b, sh, dt, c, ov, _t, _s, _g) = r
            if ov != 0:
                continue
            arr = np.frombuffer(b, dtype=np.dtype(dt)).reshape(sh).copy()
            valid_results.append((arr, c))
        if len(valid_results) >= 2:
            try:
                from macro_place.benchmark import Benchmark as _B
                from _consensus import consensus_warm_start
                bench = _B.load(bench_path)
                # Need a placer module for v1 helpers.
                import importlib.util
                ROOT = Path(bench_path).resolve().parents[3]
                spec = importlib.util.spec_from_file_location(
                    "_v2_for_consensus",
                    str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
                mod_c = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod_c)
                plc_for_consensus = mod_c._load_plc(bench.name)
                placements_only = [p for p, _ in valid_results]
                costs_only = [c for _, c in valid_results]
                cons_pos, cons_cost, source = consensus_warm_start(
                    placements_only, costs_only, bench, plc_for_consensus,
                    k_best=consensus_k_best,
                    trim_frac=consensus_trim_frac,
                    refine_max_time=consensus_refine_budget,
                    use_gpu_refine=True,
                    verbose=True)
                # Validate cons_pos is overlap-free via the same official
                # PlacementCost path used elsewhere.
                from macro_place.objective import compute_proxy_cost
                cons_full = torch.tensor(cons_pos)
                if cons_full.shape[0] < bench.macro_positions.shape[0]:
                    full = bench.macro_positions.clone()
                    full[:cons_full.shape[0]] = cons_full
                    cons_full = full
                r2 = compute_proxy_cost(cons_full, bench, plc_for_consensus)
                cons_cost_check = float(r2["proxy_cost"])
                cons_overlaps = int(r2["overlap_count"])
                print(f"{log_prefix}consensus[{source}]: cost={cons_cost_check:.6f} "
                      f"overlaps={cons_overlaps}", flush=True)
                if cons_overlaps == 0 and cons_cost_check < cost - 1e-7:
                    print(f"{log_prefix}portfolio: CONSENSUS WIN "
                          f"({cons_cost_check:.6f} vs min={cost:.6f})",
                          flush=True)
                    return cons_full, cons_cost_check, cons_overlaps, -1
            except Exception as e:
                print(f"{log_prefix}consensus err: {e} — falling back "
                      f"to portfolio min", flush=True)
    return torch.tensor(pos_np), cost, overlaps, seed
