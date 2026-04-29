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
        self.BASIN_HOP_SIGMA0 = float(os.environ.get(
            "PLACER_V7_BASIN_SIGMA0", "0.30"))
        self.LAPLACIAN_FRACTION = float(os.environ.get(
            "PLACER_V7_LAPLACIAN_BUDGET_FRAC", "0.04"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        bench_path = _benchmark_pt_path(benchmark.name)
        per_worker_budget = self.TOTAL_TIME_LIMIT
        log_prefix = "  "
        t0 = time.time()

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

        for hop in range(1, n_hops + 1):
            sigma_soft = self.BASIN_HOP_SIGMA0 * (0.6 ** (hop - 1)) * canvas_diag
            sigma_hard = sigma_soft * 0.25

            # Perturb: softs get σ_soft, hards get σ_hard.
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
            try:
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
