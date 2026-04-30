"""vmallela_v7 — Combinatorial Hail Mary

Combines three independently-validated lifts on top of the v6 pipeline:

1. **Laplacian soft-resolve** (`_soft_laplacian.py`): closed-form
   HPWL-quadratic global minimum for soft macros given fixed hards,
   followed by per-soft line search with full-proxy acceptance. Smoke
   tested at -0.072 cost lift on ibm01 from legalize+refine state.

2. **Basin-hopping** (`_basin_hop.py`): outer-loop perturbation +
   re-optimization to escape soft-state saddles. Targets the plateau
   pattern observed in the v6 diagnostic GIFs on ibm15-18.

3. **Adam on smooth surrogate** (`_smooth_proxy.py`, scaffolded): LSE-
   HPWL + CVaR top-K for density / congestion. The CVaR reformulation
   (Rockafellar-Uryasev 2000) makes top-K averages globally smooth.
   Currently HPWL-only in the surrogate; full density / congestion
   gradient with cell-window truncation is deferred. Disabled by
   default in v7.

The Laplacian piece is the load-bearing one. Basin-hopping wraps the
v6 portfolio for the saddle-escape problem.

Pipeline (per benchmark):
   Phase 0:  initial benchmark placement (the .plc init)
   Phase 1:  push-apart + legalize + refine_toward_initial
   Phase 2:  Laplacian soft-resolve (line-search)
                   -> closed-form HPWL warm-start
   Phase 3:  v6 pipeline (full)
                   -> portfolio (8 workers) + GPU CD on 1
                   + per-net + LNS + soft cycles + escape basin
                   + consensus warm-start
   Phase 4:  if budget remains: basin-hopping outer loop (each hop
                   re-runs Phase 1-3 from a perturbed start; keep best)
                   -> escape from soft saddles

Hardware-portable determinism layer is inherited from v6 (see
submissions/vmallela_v6/placer.py); same defenses apply here.
"""
from __future__ import annotations
# Self-applying locked env (mirrors v6/placer.py — see comments there).
import os as _os
for _k, _v in [
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "42"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
]:
    _os.environ.setdefault(_k, _v)

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

try:
    import threadpoolctl as _threadpoolctl
    _threadpoolctl.threadpool_limits(1)
except ImportError:
    pass
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# Make sibling and v6 imports work.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vmallela"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v2"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v6"))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from _soft_laplacian import apply_laplacian_refine
from _basin_hop import basin_hop


def _benchmark_pt_path(bench_name: str) -> str:
    root = _HERE.parents[1]
    p = root / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"
    if p.exists():
        return str(p)
    if Path(bench_name).exists():
        return str(Path(bench_name).resolve())
    return str(p)


def _sp_hop_worker(args):
    """Module-level worker for the SP-multi-worker basin-hop pool.
    Takes the same SP-perturbed init position; runs the reduced single-
    worker pipeline with a worker-specific seed for LNS/SA diversity.
    Must be pickleable so mp.spawn can fork it.

    Returns (final_pos_bytes, shape, dtype_str, cost, overlaps, seed).
    """
    (bench_path, init_pos_bytes, init_pos_shape, init_pos_dtype,
     budget, seed) = args

    # Hardware-portable determinism (mirror v6 worker setup).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from pathlib import Path as _Path
    ROOT = _Path(bench_path).resolve().parents[3]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

    import importlib.util as _ilu
    v1_spec = _ilu.spec_from_file_location(
        "_v1_sp_hop_w", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = _ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)

    from macro_place.benchmark import Benchmark as _Benchmark
    from macro_place.objective import compute_proxy_cost as _cpc
    from _soft_laplacian import apply_laplacian_refine as _lap

    bench = _Benchmark.load(bench_path)
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    n_hard = bench.num_hard_macros
    n_total = bench.macro_positions.shape[0]

    init_pos = np.frombuffer(
        init_pos_bytes, dtype=np.dtype(init_pos_dtype)
    ).reshape(init_pos_shape).copy()

    # Reduced pipeline: push-apart → legalize → refine_toward_initial.
    init_hard = init_pos[:n_hard].astype(np.float64)
    pushed = v1._push_apart(init_hard, bench, max_iters=300, damping=0.4)
    legal = v1._legalize(pushed, bench, order_type=0, step_mult=0.05)
    refined_hard = v1._refine_toward_initial(legal, init_hard, bench)

    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.sync_positions(refined_hard)
    incr.macro_pos[n_hard:] = init_pos[n_hard:n_total].astype(
        incr.macro_pos.dtype)
    np.clip(incr.macro_pos[n_hard:, 0], 0.0, cw,
            out=incr.macro_pos[n_hard:, 0])
    np.clip(incr.macro_pos[n_hard:, 1], 0.0, ch,
            out=incr.macro_pos[n_hard:, 1])
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    _lap(incr, bench, verbose=False)

    cd_budget = budget * 0.5
    sa_T0 = float(os.environ.get("PLACER_SA_T0", "0.00005"))
    sa_cooling = float(os.environ.get("PLACER_SA_COOLING", "0.9995"))
    try:
        best_hard, best_cost = v1._coord_descent(
            refined_hard, bench, plc, max_time=cd_budget,
            incr_eval=incr, sa_T0=sa_T0, sa_cooling=sa_cooling,
            sa_rng_seed=seed)
    except Exception:
        best_hard = refined_hard
        best_cost = float(incr.get_proxy_cost())

    try:
        from _per_net import per_net_optimize as _pn
        new_pos, c = _pn(best_hard, bench, incr, max_time=budget * 0.1)
        if c < best_cost:
            best_cost, best_hard = c, new_pos
    except Exception:
        pass

    try:
        from _moves import lns_destroy_repair_phase as _lns
        p, c = _lns(best_hard, bench, incr, max_time=budget * 0.15,
                    n_destroy=5, n_candidates=50)
        if c < best_cost:
            best_cost, best_hard = c, p
    except Exception:
        pass

    try:
        _lap(incr, bench, verbose=False)
    except Exception:
        pass

    full_out = np.zeros((n_total, 2), dtype=np.float64)
    full_out[:n_hard] = np.asarray(incr.macro_pos[:n_hard])
    full_out[n_hard:] = np.asarray(incr.macro_pos[n_hard:n_total])

    full_tensor = torch.tensor(full_out, dtype=torch.float32)
    r = _cpc(full_tensor, bench, plc)
    final_cost = float(r["proxy_cost"])
    final_overlaps = int(r["overlap_count"])

    return (full_out.tobytes(), full_out.shape, str(full_out.dtype),
            final_cost, final_overlaps, seed)


class OptimalPlacer:
    """v7 = v6 + Laplacian soft-resolve + (optional) basin-hopping."""

    _COMPETITION_CAP_SECONDS = 3300

    def __init__(self, seed: int = 42):
        self.seed = seed
        requested = int(os.environ.get("PLACER_TOTAL_BUDGET",
                                       self._COMPETITION_CAP_SECONDS))
        self.TOTAL_TIME_LIMIT = min(requested, self._COMPETITION_CAP_SECONDS)

        # v6 portfolio config (passed through)
        self.N_WORKERS = int(os.environ.get("PLACER_V6_WORKERS", 8))
        self.GPU_WORKERS = int(os.environ.get("PLACER_V6_GPU_WORKERS", 1))
        if self.GPU_WORKERS > self.N_WORKERS:
            self.GPU_WORKERS = self.N_WORKERS

        # v7-specific knobs
        self.LAPLACIAN_REFINE = os.environ.get(
            "PLACER_V7_LAPLACIAN", "1") == "1"
        self.LAPLACIAN_PASSES = int(os.environ.get(
            "PLACER_V7_LAPLACIAN_PASSES", "2"))
        self.BASIN_HOP_N = int(os.environ.get("PLACER_V7_BASIN_HOPS", "0"))
        # 0.10·canvas_diag is gentle enough to stay close to the
        # current basin while still moving softs by ~1-2 cell widths.
        # 0.30 (the original Wales-Doye default) is too aggressive for
        # a placement objective with steep basin walls.
        self.BASIN_HOP_SIGMA0 = float(os.environ.get(
            "PLACER_V7_BASIN_SIGMA0", "0.10"))
        self.LAPLACIAN_FRACTION = float(os.environ.get(
            "PLACER_V7_LAPLACIAN_BUDGET_FRAC", "0.04"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        bench_path = _benchmark_pt_path(benchmark.name)
        log_prefix = "  "
        t0 = time.time()

        # Reserve time at the tail for Laplacian + basin-hopping. The
        # portfolio's per-worker budget is shrunk by this amount so that
        # the final time_remaining check leaves room for at least
        # ceil(reserve / hop_budget) basin-hops on hard benches.
        # Default 0 preserves prior behavior (basin-hop only fires when
        # portfolio happens to underrun TOTAL_TIME_LIMIT, which is rare).
        # Sweep harnesses set this to ~450 to guarantee 1-2 hops fire.
        basin_reserve = int(os.environ.get("PLACER_V7_BASIN_HOP_RESERVE", "0"))
        per_worker_budget = max(300, self.TOTAL_TIME_LIMIT - basin_reserve)
        if basin_reserve > 0:
            print(f"  [v7] reserving {basin_reserve}s for laplacian+basin-hop "
                  f"(portfolio budget: {per_worker_budget}s)", flush=True)

        # Phase 1+2+3: standard v6 portfolio pipeline.
        from _portfolio import run_portfolio
        result_pos, best_cost, overlaps, best_seed = run_portfolio(
            bench_path,
            total_budget=per_worker_budget,
            n_workers=self.N_WORKERS,
            gpu_workers=self.GPU_WORKERS,
            base_seed=self.seed,
            log_prefix=log_prefix,
            apply_consensus=os.environ.get("PLACER_V6_CONSENSUS", "1") == "1",
            consensus_refine_budget=int(os.environ.get(
                "PLACER_V6_CONSENSUS_REFINE", "180")),
            consensus_k_best=int(os.environ.get(
                "PLACER_V6_CONSENSUS_K", "16")),
        )

        portfolio_cost = float(best_cost)
        portfolio_pos = result_pos.clone()
        elapsed_after_portfolio = time.time() - t0
        print(f"  [v7] after portfolio+consensus: cost={portfolio_cost:.6f} "
              f"overlaps={overlaps} ({elapsed_after_portfolio:.1f}s)",
              flush=True)

        # Phase 4: Laplacian soft-resolve. Loaded into a fresh
        # IncrementalEvaluator + applied as a sequence of per-soft line
        # searches. Strict-improvement gating means this can never make
        # things worse.
        if self.LAPLACIAN_REFINE and overlaps == 0:
            laplacian_budget_seconds = self.LAPLACIAN_FRACTION * \
                                       self.TOTAL_TIME_LIMIT
            try:
                import importlib.util as _ilu
                v1_spec = _ilu.spec_from_file_location(
                    "_v1_v7", str(_HERE.parent / "vmallela" / "placer.py"))
                v1 = _ilu.module_from_spec(v1_spec)
                v1_spec.loader.exec_module(v1)

                bench_for_eval = Benchmark.load(bench_path)
                plc = v1._load_plc(bench_for_eval.name)
                incr = v1.IncrementalEvaluator(plc, bench_for_eval)
                # Sync to portfolio's full placement (HARD positions
                # via sync_positions; SOFT positions via direct write
                # since sync_positions only handles hards).
                full_np = portfolio_pos.cpu().numpy()
                n_hard = bench_for_eval.num_hard_macros
                incr.sync_positions(full_np[:n_hard])
                # Soft positions overwrite (mirror _consensus.py's
                # _sync_full_placement helper).
                incr.macro_pos[n_hard:] = full_np[n_hard:].astype(
                    incr.macro_pos.dtype)
                incr._recompute_pin_positions()
                incr._full_recompute_wl()
                incr._full_recompute_density()
                incr._full_recompute_congestion()
                pre_lap = float(incr.get_proxy_cost())
                print(f"  [v7] laplacian: pre-cost={pre_lap:.6f}, "
                      f"running {self.LAPLACIAN_PASSES} pass(es)...",
                      flush=True)

                t_lap_start = time.time()
                for ip in range(self.LAPLACIAN_PASSES):
                    if time.time() - t_lap_start > laplacian_budget_seconds:
                        break
                    n_moved, post_cost = apply_laplacian_refine(
                        incr, bench_for_eval, verbose=True)
                    if n_moved == 0:
                        break

                # Build full placement from incr's final state.
                lap_full = np.zeros_like(full_np)
                lap_full[:n_hard] = np.asarray(incr.macro_pos[:n_hard])
                lap_full[n_hard:] = np.asarray(
                    incr.macro_pos[n_hard:full_np.shape[0]])
                # Validate via official PlacementCost.
                lap_tensor = torch.tensor(lap_full, dtype=torch.float32)
                r = compute_proxy_cost(lap_tensor, bench_for_eval, plc)
                lap_cost = float(r["proxy_cost"])
                lap_overlaps = int(r["overlap_count"])
                print(f"  [v7] laplacian: post-cost={lap_cost:.6f} "
                      f"overlaps={lap_overlaps} ({time.time()-t_lap_start:.1f}s)",
                      flush=True)
                if lap_overlaps == 0 and lap_cost < portfolio_cost - 1e-7:
                    print(f"  [v7] LAPLACIAN WIN: {lap_cost:.6f} < "
                          f"{portfolio_cost:.6f} (Δ {portfolio_cost-lap_cost:+.4f})",
                          flush=True)
                    portfolio_cost = lap_cost
                    portfolio_pos = lap_tensor
                    overlaps = lap_overlaps
                else:
                    print(f"  [v7] laplacian: keeping portfolio result "
                          f"({portfolio_cost:.6f})", flush=True)
            except Exception as e:
                print(f"  [v7] laplacian err: {e}", flush=True)

        # ── Phase 4.5: Adam polish on the smooth surrogate ─────────────
        # Vectorized LSE-HPWL + CVaR top-K density/congestion via
        # cell-window scatter, with GradNorm component balancing.
        # Strict-improvement gate via the official PlacementCost.
        # Off by default; PLACER_V7_ADAM=1 enables it.
        if (os.environ.get("PLACER_V7_ADAM", "0") == "1"
                and overlaps == 0):
            # Default 100: smooth surrogate plateau hits ~step 25-50, and
            # the smooth↔exact gap widens after that (cf. the ibm01
            # divergence post-mortem). Best-of-Adam tracking + validate-
            # every-25 captures the best exact-cost point we've seen.
            adam_steps = int(os.environ.get("PLACER_V7_ADAM_STEPS", "100"))
            adam_lr_frac = float(os.environ.get(
                "PLACER_V7_ADAM_LR_FRAC", "0.02"))
            adam_soft_only = (os.environ.get(
                "PLACER_V7_ADAM_SOFT_ONLY", "1") == "1")
            adam_inertia = float(os.environ.get(
                "PLACER_V7_ADAM_INERTIA", "1.0"))
            # α-aligned with exact proxy (top-K_d at 10 % of n_cells in the
            # exact proxy; top-K_c at 5 % of 2·n_cells in the exact proxy).
            # Using the same α as the scorer keeps the surrogate's
            # gradient pointed at the same cells the scorer cares about.
            # The earlier 2 % "sniper" configuration drove the top-2 % down
            # but lifted the 3-5 % tier (objective mismatch).
            k_dens_frac = float(os.environ.get(
                "PLACER_V7_K_DENS_FRAC", "0.10"))
            k_cong_frac = float(os.environ.get(
                "PLACER_V7_K_CONG_FRAC", "0.05"))
            # Window refresh + best-of-Adam validation cadence.
            adam_snapshot_every = int(os.environ.get(
                "PLACER_V7_ADAM_SNAPSHOT_EVERY", "10"))
            adam_validate_every = int(os.environ.get(
                "PLACER_V7_ADAM_VALIDATE_EVERY", "25"))
            # Honest-surrogate fail-safe modes:
            #   Mode A (HPWL+density only):  PLACER_V7_ADAM_ENABLE_CONG=0
            #     The congestion surrogate is "proxy of a proxy" — V_routing
            #     is frozen at the pre-Adam state and divides by grid_v_routes
            #     instead of vrouting_alloc. Adam happily drives surrogate
            #     congestion down while exact congestion gets worse. Disabling
            #     it removes the lying gradient.
            #   Mode B (HPWL only):  ADAM_ENABLE_CONG=0 + ADAM_ENABLE_DENS=0
            #     Smooth refinement of the Laplacian solve only. The
            #     "nuclear" honest option.
            adam_enable_dens = (os.environ.get(
                "PLACER_V7_ADAM_ENABLE_DENS", "1") == "1")
            adam_enable_cong = (os.environ.get(
                "PLACER_V7_ADAM_ENABLE_CONG", "1") == "1")
            try:
                from _smooth_proxy import adam_warm_start
                # Build a fresh IncrementalEvaluator synced to current pos.
                import importlib.util as _ilu
                v1_spec = _ilu.spec_from_file_location(
                    "_v1_v7_adam",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1 = _ilu.module_from_spec(v1_spec)
                v1_spec.loader.exec_module(v1)
                bench_for_eval = Benchmark.load(bench_path)
                plc = v1._load_plc(bench_for_eval.name)
                incr = v1.IncrementalEvaluator(plc, bench_for_eval)
                full_np = portfolio_pos.cpu().numpy()
                n_hard = bench_for_eval.num_hard_macros
                incr.sync_positions(full_np[:n_hard])
                incr.macro_pos[n_hard:] = full_np[n_hard:].astype(
                    incr.macro_pos.dtype)
                incr._recompute_pin_positions()
                incr._full_recompute_wl()
                incr._full_recompute_density()
                incr._full_recompute_congestion()

                t_adam_start = time.time()
                pos_adam, history = adam_warm_start(
                    incr, bench_for_eval,
                    n_steps=adam_steps,
                    lr_frac_canvas=adam_lr_frac,
                    proximal_weight_frac=adam_inertia,
                    soft_only=adam_soft_only,
                    enable_density=adam_enable_dens,
                    enable_congestion=adam_enable_cong,
                    window_margin_cells=4,
                    snapshot_every=adam_snapshot_every,
                    validate_every=adam_validate_every,
                    k_dens_frac=k_dens_frac,
                    k_cong_frac=k_cong_frac,
                    verbose=True,
                )
                adam_tensor = torch.tensor(pos_adam, dtype=torch.float32)
                # Validate via official PlacementCost.
                r = compute_proxy_cost(adam_tensor, bench_for_eval, plc)
                adam_cost = float(r["proxy_cost"])
                adam_overlaps = int(r["overlap_count"])
                print(f"  [v7] adam: {adam_steps} steps in "
                      f"{time.time()-t_adam_start:.1f}s; surrogate-loss "
                      f"{history['loss'][0]:.4f} → {history['loss'][-1]:.4f}; "
                      f"exact-proxy {portfolio_cost:.6f} → {adam_cost:.6f} "
                      f"overlaps={adam_overlaps}", flush=True)
                if (adam_overlaps == 0
                        and adam_cost < portfolio_cost - 1e-7):
                    print(f"  [v7] ADAM WIN: {adam_cost:.6f} < "
                          f"{portfolio_cost:.6f} "
                          f"(Δ {portfolio_cost-adam_cost:+.4f})",
                          flush=True)
                    portfolio_cost = adam_cost
                    portfolio_pos = adam_tensor
                    overlaps = adam_overlaps
                else:
                    print(f"  [v7] adam: rejected (no improvement or "
                          f"overlaps); keeping post-Laplacian "
                          f"({portfolio_cost:.6f})", flush=True)
            except Exception as e:
                print(f"  [v7] adam err: {type(e).__name__}: {e}; "
                      f"keeping post-Laplacian", flush=True)

        # Phase 5: basin-hopping on hard benches only (cost >= 1.0).
        # Each hop perturbs the current best by σ * canvas, runs a
        # SINGLE-WORKER reduced-budget pipeline, keeps if better.
        # We trigger only when (a) the main portfolio result is >= 1.0
        # (we have headroom to lose), (b) basin_hop_n > 0, and (c) we
        # have budget remaining.
        time_remaining = self.TOTAL_TIME_LIMIT - (time.time() - t0)
        BASIN_HOP_BUDGET_PER_HOP = int(os.environ.get(
            "PLACER_V7_BASIN_HOP_BUDGET", "300"))   # 5 min/hop default
        BASIN_HOP_AUTO_THRESHOLD = float(os.environ.get(
            "PLACER_V7_BASIN_HOP_AUTO", "1.00"))

        # Auto-enable hops when result is over the threshold and we have
        # ≥ 1 hop's worth of time remaining.
        auto_hop = (
            portfolio_cost >= BASIN_HOP_AUTO_THRESHOLD
            and time_remaining > BASIN_HOP_BUDGET_PER_HOP * 1.2
            and overlaps == 0
        )
        n_hops_to_run = self.BASIN_HOP_N
        if n_hops_to_run == 0 and auto_hop:
            # Auto: 1 hop if we have just over 1 budget; up to 3 if we
            # have plenty of time.
            n_hops_to_run = min(3, int(time_remaining // BASIN_HOP_BUDGET_PER_HOP))

        if n_hops_to_run > 0 and overlaps == 0:
            print(f"  [v7] basin-hopping: N={n_hops_to_run} hops × "
                  f"{BASIN_HOP_BUDGET_PER_HOP}s budget each, σ_0="
                  f"{self.BASIN_HOP_SIGMA0:.2f}·canvas_diag", flush=True)
            try:
                portfolio_pos, portfolio_cost = self._basin_hop_loop(
                    portfolio_pos, portfolio_cost, bench_path,
                    n_hops_to_run, BASIN_HOP_BUDGET_PER_HOP)
            except Exception as e:
                print(f"  [v7] basin-hop err: {e}; keeping portfolio result",
                      flush=True)
        elif self.BASIN_HOP_N > 0:
            print(f"  [v7] skipping basin-hop (budget remaining "
                  f"{time_remaining:.0f}s < {BASIN_HOP_BUDGET_PER_HOP}s "
                  f"per hop, or invalid)", flush=True)

        # Final report
        if overlaps != 0:
            print(f"  [v7] WARNING: best result has {overlaps} overlaps "
                  f"(cost={portfolio_cost:.6f})", flush=True)
        else:
            print(f"  [v7] DONE: cost={portfolio_cost:.6f} overlaps=0 "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # Persist the FINAL post-Laplacian/post-basin-hop placement for
        # offline plotting. Mirrors v6's PLACER_V6_SAVE_PLACEMENT contract
        # so scripts/v7_overnight_sweep.sh can render PNGs and GIFs from
        # the same .npy artifacts.
        save_template = os.environ.get("PLACER_V6_SAVE_PLACEMENT")
        if save_template:
            try:
                save_path = save_template.format(name=benchmark.name)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, portfolio_pos.detach().cpu().numpy())
                print(f"  [v7] saved placement to {save_path}", flush=True)
            except Exception as e:
                print(f"  [v7] WARNING: failed to save placement: {e}",
                      flush=True)
        return portfolio_pos

    def _basin_hop_loop(self, current_pos, current_cost, bench_path,
                        n_hops, hop_budget):
        """Run n_hops basin-hopping iterations from `current_pos`. Each
        hop:
            1. Perturb softs by σ_k · canvas_diag (Gaussian); softer
               perturbation on hards (× 0.25) to keep legalize feasible.
            2. Run a SINGLE-CPU-worker reduced-budget pipeline from the
               perturbed start: push-apart + legalize + refine + hard CD
               + Laplacian soft-resolve + per-net + hard LNS + soft cycle.
            3. Validate via official PlacementCost.
            4. Strict accept (only commit if cost strictly improves).
            5. Cool σ by 0.6× per hop.

        Single-worker keeps each hop cheap (~hop_budget seconds wall-
        clock). With 3 hops at 300s, total basin-hop overhead is ~15
        min on top of the main portfolio's ~30 min.
        """
        import importlib.util as _ilu
        # Reload the v1 placer module — fresh state, no cached gates.
        v1_spec = _ilu.spec_from_file_location(
            "_v1_basinhop", str(_HERE.parent / "vmallela" / "placer.py"))
        v1 = _ilu.module_from_spec(v1_spec)
        v1_spec.loader.exec_module(v1)

        bench = Benchmark.load(bench_path)
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        canvas_diag = math.hypot(cw, ch)
        n_hard = bench.num_hard_macros
        n_total = bench.macro_positions.shape[0]
        rng = np.random.default_rng(self.seed + 7777)

        best_pos = current_pos.clone()
        best_cost = float(current_cost)
        accepted = 0

        # Perturbation mode: gaussian (default) or sp (sequence-pair swap).
        # SP swaps change the topological order of hard macros (e.g., flips
        # "A is left of B" into "A is below B") which Gaussian noise can't
        # do reliably within a budget that doesn't destroy feasibility.
        # The 9-config Gaussian σ-grid showed 0/9 hops accepted on ibm15
        # because the post-perturbation legalize couldn't recover from any
        # meaningful Gaussian σ. SP perturbation is structural: decode
        # produces a tightly-packed feasible placement, which the local
        # minimizer can refine.
        perturb_mode = os.environ.get("PLACER_V7_BASIN_PERTURB", "gaussian")
        sp_n_swaps = int(os.environ.get("PLACER_V7_SP_N_SWAPS", "3"))

        for hop in range(1, n_hops + 1):
            if perturb_mode == "sp":
                # SP perturbation: encode current hards → swap → decode →
                # fit to canvas. Softs are taken from current state and
                # the reduced pipeline's push-apart/legalize/refine fixes
                # any overlap and re-routes softs around the new hard
                # arrangement.
                from _sequence_pair import (encode_sp, sp_swap as _sp_swap_fn,
                                              decode_sp, fit_to_canvas)
                import random as _random
                cur_np = best_pos.cpu().numpy().astype(np.float64).copy()
                # Benchmark stores positions as CENTERS and sizes as
                # (n, 2) tensor of (width, height). SP decode operates
                # on lower-left corners, so we convert.
                hard_sizes = bench.macro_sizes.cpu().numpy().astype(
                    np.float64)[:n_hard]
                hard_w_np = hard_sizes[:, 0]
                hard_h_np = hard_sizes[:, 1]
                hard_pos_ll = cur_np[:n_hard].copy()
                hard_pos_ll[:, 0] -= hard_w_np / 2.0
                hard_pos_ll[:, 1] -= hard_h_np / 2.0
                alpha0, beta0 = encode_sp(hard_pos_ll)
                py_rng = _random.Random(self.seed + 1000 * hop)
                alpha1, beta1 = _sp_swap_fn(
                    alpha0, beta0, n_swaps=sp_n_swaps, rng=py_rng)
                new_hx, new_hy = decode_sp(
                    alpha1, beta1, hard_w_np, hard_h_np)
                new_hard_xy = np.stack([new_hx, new_hy], axis=1)
                new_hx_fit, new_hy_fit, scale = fit_to_canvas(
                    new_hard_xy[:, 0], new_hard_xy[:, 1],
                    hard_w_np, hard_h_np, cw, ch)
                perturbed = cur_np.copy()
                # Convert lower-left back to center coords for the placer.
                perturbed[:n_hard, 0] = new_hx_fit + hard_w_np / 2.0
                perturbed[:n_hard, 1] = new_hy_fit + hard_h_np / 2.0
                # Soft positions: leave at current state. The downstream
                # pipeline's push-apart + legalize + refine will fix any
                # overlap with the relocated hards.
                print(f"  [v7.hop {hop}] SP swap k={sp_n_swaps} "
                      f"(scale={scale:.3f}), running single-worker pipeline "
                      f"at {hop_budget}s...", flush=True)
            else:
                sigma_soft = self.BASIN_HOP_SIGMA0 * (0.6 ** (hop - 1)) * canvas_diag
                sigma_hard = sigma_soft * 0.25
                # Gaussian perturbation: softs get σ_soft, hards get σ_hard.
                perturbed = best_pos.cpu().numpy().astype(np.float64).copy()
                noise = np.zeros_like(perturbed)
                noise[:n_hard] = rng.normal(0.0, sigma_hard, (n_hard, 2))
                noise[n_hard:n_total] = rng.normal(
                    0.0, sigma_soft, (n_total - n_hard, 2))
                perturbed = perturbed + noise
                perturbed[:, 0] = np.clip(perturbed[:, 0], 0.0, cw)
                perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, ch)
                print(f"  [v7.hop {hop}] σ_soft={sigma_soft:.2f} "
                      f"σ_hard={sigma_hard:.2f}, "
                      f"running single-worker pipeline at {hop_budget}s...",
                      flush=True)

            # Run the v4 single-worker pipeline at reduced budget from
            # the perturbed start. We achieve "start from perturbed" by
            # constructing a fresh IncrementalEvaluator at the
            # perturbed position (push_apart will be a no-op if
            # already overlap-free; legalize will fix anything broken).

            # Run a custom reduced pipeline inline (fewer phases, same
            # operators). For clarity, we just call push-apart + legal
            # + refine to fix the perturbation, then run the v6 GPU CD
            # at the hop_budget if available, else CPU CD, then
            # Laplacian, then per-net, then return.
            #
            # When PLACER_V7_SP_MULTI_WORKERS > 1 (Option C), spawn
            # N parallel workers from the same SP-perturbed init with
            # different LNS/SA seeds; take the min as the hop result.
            # The diversity across worker seeds is the lever — same
            # input, different exploration trajectories.
            n_sp_mw = int(os.environ.get("PLACER_V7_SP_MULTI_WORKERS", "1"))
            try:
                if perturb_mode == "sp" and n_sp_mw > 1:
                    hop_pos, hop_cost = self._sp_multi_worker_hop(
                        perturbed, bench_path, hop_budget, n_sp_mw,
                        base_seed=self.seed + 1000 + hop)
                else:
                    hop_pos, hop_cost = self._reduced_single_pipeline(
                        perturbed, bench, v1, hop_budget,
                        seed=self.seed + 1000 + hop)
                # Validate via official PlacementCost
                hop_tensor = torch.tensor(hop_pos, dtype=torch.float32)
                plc = v1._load_plc(bench.name)
                r = compute_proxy_cost(hop_tensor, bench, plc)
                hop_proxy = float(r["proxy_cost"])
                hop_overlaps = int(r["overlap_count"])
                print(f"  [v7.hop {hop}] result: cost={hop_proxy:.6f} "
                      f"overlaps={hop_overlaps}", flush=True)
                if hop_overlaps == 0 and hop_proxy < best_cost - 1e-7:
                    print(f"  [v7.hop {hop}] ACCEPTED: {hop_proxy:.6f} < "
                          f"{best_cost:.6f} (Δ {best_cost - hop_proxy:+.4f})",
                          flush=True)
                    best_pos = hop_tensor
                    best_cost = hop_proxy
                    accepted += 1
                else:
                    print(f"  [v7.hop {hop}] rejected (best stays "
                          f"{best_cost:.6f})", flush=True)
            except Exception as e:
                print(f"  [v7.hop {hop}] err: {e}", flush=True)

        print(f"  [v7] basin-hopping done. {accepted}/{n_hops} accepted.",
              flush=True)
        return best_pos, best_cost

    def _sp_multi_worker_hop(self, perturbed_full, bench_path, budget,
                              n_workers, base_seed):
        """Multi-worker SP basin-hop. N workers run the reduced pipeline
        in parallel from the SAME SP-perturbed init position; each uses
        a different LNS/SA seed for exploration diversity. Returns the
        (best_pos, best_cost) across all workers.

        This is Option C: trade per-bench wall time (~15 min/hop with 4
        workers × 900s) for the diversity that single-worker basin-hops
        couldn't get from time alone (the smoke verdict: 900s × 1
        plateaued at the same cost as 300s × 2; local minimizer was the
        ceiling, not time).
        """
        import multiprocessing as _mp

        init_pos_np = np.ascontiguousarray(perturbed_full, dtype=np.float64)
        args = []
        for w in range(n_workers):
            seed = base_seed + w
            args.append((bench_path, init_pos_np.tobytes(),
                          init_pos_np.shape, str(init_pos_np.dtype),
                          int(budget), seed))

        print(f"  [v7.hop] SP multi-worker pool: {n_workers} workers × "
              f"{int(budget)}s each, base_seed={base_seed}", flush=True)
        ctx = _mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_sp_hop_worker, args)

        best_cost = float("inf")
        best_pos = None
        best_seed = None
        for r in results:
            (pos_bytes, shape, dtype_str, cost, overlaps, seed) = r
            tag = "VALID" if overlaps == 0 else f"INVALID({overlaps})"
            print(f"    [v7.hop.w-s{seed}] cost={cost:.6f} {tag}", flush=True)
            if overlaps != 0:
                continue
            if cost < best_cost:
                best_cost = cost
                best_pos = np.frombuffer(
                    pos_bytes, dtype=np.dtype(dtype_str)
                ).reshape(shape).copy()
                best_seed = seed

        if best_pos is None:
            # All workers produced invalid placements — return the
            # lowest-cost result anyway (the gate will reject).
            results.sort(key=lambda r: r[3])
            r = results[0]
            best_pos = np.frombuffer(
                r[0], dtype=np.dtype(r[2])).reshape(r[1]).copy()
            best_cost = float(r[3])
            best_seed = r[5]

        print(f"  [v7.hop] SP multi-worker BEST: seed={best_seed} "
              f"cost={best_cost:.6f}", flush=True)
        return best_pos, best_cost

    def _reduced_single_pipeline(self, perturbed_full, bench, v1,
                                  budget, seed):
        """A reduced-cost single-worker pipeline used per basin-hop.

        Phases: push-apart → legalize → refine_toward_initial →
                Laplacian soft-resolve → CPU CD (or GPU CD) →
                per-net → hard LNS.

        Faster than the full v4 pipeline (skips the multi-cycle soft
        loop and the escape basin) since basin-hopping does the
        "explore" outer loop.
        """
        n_hard = bench.num_hard_macros
        n_total = bench.macro_positions.shape[0]
        init_hard = perturbed_full[:n_hard].astype(np.float64)
        # Push-apart fixes overlaps from the perturbation.
        pushed = v1._push_apart(init_hard, bench, max_iters=300, damping=0.4)
        legal = v1._legalize(pushed, bench, order_type=0, step_mult=0.05)
        refined_hard = v1._refine_toward_initial(legal, init_hard, bench)

        plc = v1._load_plc(bench.name)
        incr = v1.IncrementalEvaluator(plc, bench)
        incr.sync_positions(refined_hard)
        # Override soft positions with the perturbed softs (sync
        # only sets hards).
        incr.macro_pos[n_hard:] = perturbed_full[n_hard:n_total].astype(
            incr.macro_pos.dtype)
        # Clip softs to canvas
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        np.clip(incr.macro_pos[n_hard:, 0], 0.0, cw,
                out=incr.macro_pos[n_hard:, 0])
        np.clip(incr.macro_pos[n_hard:, 1], 0.0, ch,
                out=incr.macro_pos[n_hard:, 1])
        incr._recompute_pin_positions()
        incr._full_recompute_wl()
        incr._full_recompute_density()
        incr._full_recompute_congestion()

        # Phase: Laplacian soft-resolve (line-search; we can't make
        # things worse).
        n_moved, _ = apply_laplacian_refine(incr, bench, verbose=False)

        # Hard CD at ~50% of budget.
        cd_budget = budget * 0.5
        # Use SA from v4 defaults
        sa_T0 = float(os.environ.get("PLACER_SA_T0", "0.00005"))
        sa_cooling = float(os.environ.get("PLACER_SA_COOLING", "0.9995"))
        try:
            best_hard, best_cost = v1._coord_descent(
                refined_hard, bench, plc, max_time=cd_budget,
                incr_eval=incr, sa_T0=sa_T0, sa_cooling=sa_cooling,
                sa_rng_seed=seed)
        except Exception as e:
            print(f"      [reduced.cd] err: {e}", flush=True)
            best_hard = refined_hard
            best_cost = float(incr.get_proxy_cost())

        # Per-net step (hard + soft at expected gradient direction).
        try:
            from _per_net import per_net_optimize
            new_pos, c = per_net_optimize(
                best_hard, bench, incr, max_time=budget * 0.1)
            if c < best_cost:
                best_cost, best_hard = c, new_pos
        except Exception:
            pass

        # Hard LNS.
        try:
            from _moves import lns_destroy_repair_phase
            lns_t = budget * 0.15
            p, c = lns_destroy_repair_phase(
                best_hard, bench, incr, max_time=lns_t,
                n_destroy=5, n_candidates=50)
            if c < best_cost:
                best_cost, best_hard = c, p
        except Exception:
            pass

        # Final Laplacian soft refine
        try:
            apply_laplacian_refine(incr, bench, verbose=False)
        except Exception:
            pass

        # Build full output (hard + soft)
        full_out = np.zeros((n_total, 2), dtype=np.float64)
        full_out[:n_hard] = np.asarray(incr.macro_pos[:n_hard])
        full_out[n_hard:] = np.asarray(incr.macro_pos[n_hard:n_total])
        return full_out, best_cost
