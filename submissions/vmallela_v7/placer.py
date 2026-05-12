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
# `setdefault` so explicit env from the launcher still wins; reproduces the
# exact config from scripts/v7_singlev4_full_sweep.sh (ibm15=1.0835,
# 17-bench mean=1.0003) when the grader invokes the placer with no env.
import os as _os
for _k, _v in [
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "42"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    ("PLACER_TOTAL_BUDGET", "2300"),
    ("PLACER_V6_WORKERS", "1"),
    ("PLACER_V6_GPU_WORKERS", "0"),
    ("PLACER_V6_CONSENSUS", "0"),
    ("PLACER_SA_T0", "0.00005"),
    ("PLACER_ESC_HARD_DESTROY", "80"),
    ("PLACER_V7_LAPLACIAN", "1"),
    ("PLACER_V7_LAPLACIAN_PASSES", "2"),
    ("PLACER_V7_LAPLACIAN_BUDGET_FRAC", "0.04"),
    ("PLACER_V7_BASIN_HOPS", "0"),
    ("PLACER_V7_BASIN_HOP_AUTO", "999.0"),
    ("PLACER_V7_BASIN_HOP_RESERVE", "0"),
    ("PLACER_V7_ADAM", "0"),
    ("PLACER_V7_EVICT", "0"),
    ("PLACER_V7_SINKHORN", "0"),
    ("PLACER_V7_HESSIAN", "1"),
    ("PLACER_V7_HESSIAN_STEPS", "0.02,-0.02,0.05,-0.05"),
    ("PLACER_V7_HESSIAN_BUDGET", "1000"),
    # Lanczos default bumped 50 → 100 because cong-included surrogate has
    # worse conditioning than HPWL+density alone. Combined with the new
    # auto-retry path in _hessian_escape (4× bump on convergence failure)
    # and Tikhonov regularization (1e-4 default), Lanczos now succeeds on
    # all 17 benches at default config.
    ("PLACER_V7_HESSIAN_LANCZOS", "100"),
    ("PLACER_V7_HESSIAN_TIKHONOV", "1e-4"),
    ("PLACER_V7_HESSIAN_MAX_ITERS", "1"),
    # istanbul: enable adaptive line-search + feasibility filter for
    # the saddle-escape phase. Replaces the hardcoded HESSIAN_STEPS
    # and drops candidates whose post-perturbation overlap count
    # exceeds MAX_OVERLAPS, saving wasted SA worker time.
    ("PLACER_V7_HESSIAN_ADAPTIVE", "1"),
    ("PLACER_V7_HESSIAN_ADAPTIVE_TOPK", "1"),
    ("PLACER_V7_HESSIAN_LS_INITIAL", "0.10"),
    ("PLACER_V7_HESSIAN_LS_STEPS", "10"),
    ("PLACER_V7_HESSIAN_LS_SHRINK", "0.6"),
    ("PLACER_V7_HESSIAN_MAX_OVERLAPS", "200"),
    # albania1: Tier 2 / Tier 1 levers. Default-on for orientation
    # sidecar (Tier 2 only, no Tier 1 effect), default-off for halo
    # (touches density surrogate; needs A/B before flipping on).
    ("PLACER_V7_ORIENTATION_FLIP", "1"),
    ("PLACER_V7_ORIENTATION_PASSES", "2"),
    ("PLACER_V7_HALO_FRAC", "0.0"),
    # Congestion-aware Hessian. Default ON (the breakthrough lever);
    # the prior surrogate omitted congestion entirely. Set to "0" to
    # reproduce the verified-1.0109 baseline behavior.
    ("PLACER_V7_HESSIAN_CONG", "1"),
    ("PLACER_V7_K_CONG_FRAC", "0.05"),
    # Per-component weights in the SURROGATE (defaults match proxy).
    # Boosting these above their proxy weights over-weights that term
    # in the eigenvector direction without changing the strict-improvement
    # gate (which uses the exact proxy with the original weights). Safe
    # upside: emphasizing the variance-dominant term during saddle search;
    # monotone gating prevents regressions. Setting any to 0 ablates that
    # term from the surrogate (e.g. cong-only saddle escape).
    ("PLACER_V7_HESSIAN_HPWL_WEIGHT", "1.0"),
    ("PLACER_V7_HESSIAN_DENS_WEIGHT", "0.5"),
    ("PLACER_V7_HESSIAN_CONG_WEIGHT", "0.5"),
    # zeus: Differentiable RUDY routing demand. When enabled (=1), the
    # frozen V_routing_smooth / H_routing_smooth tensors are replaced
    # with the Spindler-Johannes RUDY surrogate computed from the
    # current pin positions via LSE-bbox + softplus cell overlap. See
    # _rudy_smooth.py for the math and research/ITERATIONS.md (Iter 4d,
    # 7) for why this matters: the existing cong-aware Hessian was
    # bench-noise because routing was FROZEN — the eigvec direction
    # was good on the stale map but actively wrong on the live map.
    # Default 0 for backwards compatibility with verified 0.9975 sweep.
    ("PLACER_V7_HESSIAN_RUDY", "0"),
    ("PLACER_V7_HESSIAN_RUDY_MARGIN", "4"),
    ("PLACER_V7_HESSIAN_RUDY_MAX_WINDOW", "64"),
    # zeus: Subspace Hamiltonian Monte Carlo escape (see _subspace_hmc.py).
    # When K>0 and TRAJ>0, after the adaptive line-search candidates are
    # generated, run an additional Lanczos call to get K eigvecs of the
    # smooth-surrogate Hessian, then sample TRAJ HMC trajectories with
    # random momentum p ~ N(0, |Λ_K|). Each trajectory's endpoint becomes
    # one additional candidate; the existing strict-improvement gate
    # against EXACT proxy filters out bad samples. Default off (0,0).
    ("PLACER_V7_HESSIAN_HMC_K", "0"),
    ("PLACER_V7_HESSIAN_HMC_TRAJ", "0"),
    ("PLACER_V7_HESSIAN_HMC_L", "12"),
    ("PLACER_V7_HESSIAN_HMC_STEP", "0.5"),
    ("PLACER_V7_HESSIAN_HMC_CAP", "0.20"),
    # Per-bench auto-tuned cong weight based on netlist demand/supply
    # residual (research/lower_bounds/cong_difficulty.csv). High-room
    # benches benefit from aggressive cong weighting; low-room benches
    # are already at floor and would only see noise. Default off; when
    # enabled, env CONG_WEIGHT acts as a multiplier on the per-bench
    # auto-scale.
    ("PLACER_V7_HESSIAN_AUTO_CONG", "0"),
    # AUTO_LAMBDA_SCAN: pre-Hessian sweep over cong_weight candidates
    # to find the value maximizing |λ_min| (deepest saddle = most
    # effective escape). Physics: at the "eigenvector transition" point
    # where the dominant eigvec rotates from HPWL-dominated to cong-
    # dominated, the negative curvature is reinforced from both modes.
    # Cost ~25s overhead vs 1000s Hessian budget. Validates against
    # the cong-weight sensitivity test on ibm06 (peak |λ_min| at w=0.75).
    ("PLACER_V7_HESSIAN_AUTO_LAMBDA_SCAN", "0"),
    ("PLACER_V7_HESSIAN_LAMBDA_SCAN_WEIGHTS",
     "0.25,0.5,0.75,1.0,1.5,2.0"),
    # Electrostatic-field density (DREAMPlace/ePlace-style). Replaces
    # CVaR top-K density in the Hessian smooth surrogate with a
    # Poisson-energy formulation: ∇²φ = ρ - ρ̄, energy = ∫|φ|². The
    # Hessian eigvec under this surrogate captures GLOBAL density
    # structure (long-range repulsion) instead of CVaR's local myopia.
    # Strict-improvement gate against EXACT proxy (which still uses
    # top-K density) preserves Tier 1 safety. The combo gives DREAMPlace-
    # class density curvature direction inside our combinatorial pipeline.
    ("PLACER_V7_HESSIAN_ELECTROSTATIC", "0"),
    # Weight on electrostatic energy (compared to HPWL_LSE term).
    # Balances density-spreading force vs HPWL-collapse force.
    ("PLACER_V7_HESSIAN_ELECTRO_WEIGHT", "1.0"),
    # Use normalized (scale-balanced) electrostatic energy: divides by
    # var(ρ)·canvas_area so the term is O(1) instead of O(10^5). This
    # makes the Hessian eigvec a true mixture of HPWL+density modes
    # rather than electro-dominated.
    ("PLACER_V7_HESSIAN_ELECTRO_NORM", "0"),
    # Fiedler recursive-bisect warm-start. Computes a globally-structured
    # initial placement from the netlist Laplacian (Hall 1970 spectral
    # partitioning). Replaces .plc init for v4. Reduces v4-baseline
    # variance by giving v4 a deterministic, netlist-topology-aware
    # starting point. ~0.3s overhead per bench.
    ("PLACER_V7_RECURSIVE_BISECT", "0"),
]:
    _os.environ.setdefault(_k, _v)


# Per-bench cong residual table (from cong_difficulty.csv at v7 baseline).
# Used by the AUTO_CONG path: positive residual = "v7 above structural
# floor" = algorithmic room exists.  Higher residual → boost cong_weight.
_BENCH_CONG_RESIDUAL = {
    "ibm01": -0.165, "ibm02": +0.032, "ibm03": +0.062, "ibm04": -0.004,
    "ibm06": +0.262, "ibm07": +0.080, "ibm08": +0.045, "ibm09": -0.232,
    "ibm10": -0.072, "ibm11": -0.181, "ibm12": +0.269, "ibm13": -0.062,
    "ibm14": -0.086, "ibm15": +0.032, "ibm16": -0.007, "ibm17": -0.207,
    "ibm18": +0.234,
}


def _auto_cong_weight(bench_name, base_weight=0.5):
    """Per-bench cong weight scaling based on structural residual.

    residual > 0.20 → 3.0× base (high-room: ibm06, ibm12, ibm18)
    residual > 0.05 → 2.0× base (medium-room: ibm03, ibm07, ibm08)
    residual > 0.0  → 1.5× base (small-room: ibm02, ibm15)
    residual ≤ 0.0  → 0.5× base (below floor: 9 benches)

    Sub-1 weight on below-floor benches keeps cong from disturbing the
    well-converged HPWL+density saddle structure (where v7 is already
    beating the structural floor).
    """
    r = _BENCH_CONG_RESIDUAL.get(bench_name, 0.0)
    if r > 0.20: scale = 3.0
    elif r > 0.05: scale = 2.0
    elif r > 0.0: scale = 1.5
    else: scale = 0.5
    return base_weight * scale

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
        # PLACER_BASE_SEED env overrides constructor default so grader
        # invocation (which calls OptimalPlacer() with no args) can be
        # seed-swept via env without touching the evaluator harness.
        self.seed = int(os.environ.get("PLACER_BASE_SEED", seed))
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

        # albania1: optional Fiedler recursive-bisect warm-start.
        # Replaces .plc init with a globally-structured placement based
        # on netlist topology (Hall 1970 spectral partitioning).
        # Reduces v4-baseline variance — same netlist structure produces
        # the same init every run, so v4's refinement is more
        # deterministic. Not yet uniformly tested; default off.
        if os.environ.get("PLACER_V7_RECURSIVE_BISECT", "0") == "1":
            try:
                from _recursive_bisect import compute_recursive_bisect_init
                import importlib.util as _ilu
                v1_spec_b = _ilu.spec_from_file_location(
                    "_v1_bisect",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1_b = _ilu.module_from_spec(v1_spec_b)
                v1_spec_b.loader.exec_module(v1_b)
                bench_for_bisect = Benchmark.load(bench_path)
                plc_b = v1_b._load_plc(bench_for_bisect.name)
                incr_b = v1_b.IncrementalEvaluator(plc_b, bench_for_bisect)
                n_macros_b = int(np.asarray(incr_b.macro_pos).shape[0])
                bisect_pos = compute_recursive_bisect_init(
                    np.asarray(incr_b.pin_macro),
                    np.asarray(incr_b.net_starts),
                    np.asarray(incr_b.net_weight),
                    n_macros_b, bench_for_bisect.num_hard_macros,
                    float(bench_for_bisect.canvas_width),
                    float(bench_for_bisect.canvas_height),
                    np.asarray(incr_b.macro_w),
                    np.asarray(incr_b.macro_h),
                    verbose=True)
                # Save modified benchmark with bisect-init macro_positions.
                # MUST live inside the repo's benchmarks/processed/public/
                # because _portfolio.py computes ROOT = parents[3] of
                # bench_path. /tmp/... would give ROOT='/' breaking
                # relative imports of vmallela_v2.
                bench_for_bisect.macro_positions = torch.tensor(
                    bisect_pos.astype(np.float32))
                bench_dir = _HERE.parents[1] / "benchmarks" / "processed" / "public"
                tmp_path = bench_dir / f"{bench_for_bisect.name}_bisect.pt"
                bench_for_bisect.save(str(tmp_path))
                print(f"  [v7] RECURSIVE_BISECT: saved warm-start to "
                      f"{tmp_path}; v4 will use it as init", flush=True)
                bench_path = str(tmp_path)
            except Exception as e:
                print(f"  [v7] RECURSIVE_BISECT failed: {e} — "
                      f"falling back to .plc init", flush=True)

        # albania2 Bet A: Phase 0 homotopy spreader as warm-start.
        # CVaR-density + cosine λ-ramp from .plc init. Smoke-tested
        # to improve standalone proxy by ~6% on ibm06; if v6 + Hessian
        # carry that lift through, it's a real win.
        # Mutually exclusive with RECURSIVE_BISECT (whichever fires first
        # produces the warm-start; gated by env vars).
        if os.environ.get("PLACER_V7_PHASE0", "0") == "1":
            try:
                from _phase0_electrostatic import electrostatic_spread_homotopy
                import importlib.util as _ilu_p0
                v1_spec_p0 = _ilu_p0.spec_from_file_location(
                    "_v1_phase0",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1_p0 = _ilu_p0.module_from_spec(v1_spec_p0)
                v1_spec_p0.loader.exec_module(v1_p0)
                bench_for_p0 = Benchmark.load(bench_path)
                plc_p0 = v1_p0._load_plc(bench_for_p0.name)
                incr_p0 = v1_p0.IncrementalEvaluator(plc_p0, bench_for_p0)
                p0_n_iters = int(os.environ.get(
                    "PLACER_V7_PHASE0_ITERS", "500"))
                p0_n_stages = int(os.environ.get(
                    "PLACER_V7_PHASE0_STAGES", "20"))
                p0_lambda_0 = float(os.environ.get(
                    "PLACER_V7_PHASE0_LAMBDA_0", "0.05"))
                p0_lambda_f = float(os.environ.get(
                    "PLACER_V7_PHASE0_LAMBDA_F", "2.0"))
                p0_lr_frac = float(os.environ.get(
                    "PLACER_V7_PHASE0_LR_FRAC", "0.001"))
                t_p0 = time.time()
                phase0_pos = electrostatic_spread_homotopy(
                    bench_for_p0, incr_p0,
                    n_iters=p0_n_iters,
                    n_stages=p0_n_stages,
                    lambda_0=p0_lambda_0,
                    lambda_f=p0_lambda_f,
                    lr_frac_canvas=p0_lr_frac,
                    init_from_plc=True,
                    soft_only=False,
                    verbose=True)
                # Save the warm-start as a modified .pt next to the bench
                # (must live inside benchmarks/processed/public/ — see
                # ROOT computation note in RECURSIVE_BISECT block).
                bench_for_p0.macro_positions = torch.tensor(
                    phase0_pos.astype(np.float32))
                bench_dir = _HERE.parents[1] / "benchmarks" / "processed" / "public"
                p0_path = bench_dir / f"{bench_for_p0.name}_phase0.pt"
                bench_for_p0.save(str(p0_path))
                print(f"  [v7] PHASE0 saved warm-start to {p0_path} "
                      f"in {time.time()-t_p0:.1f}s; v4 will use it as init",
                      flush=True)
                bench_path = str(p0_path)
            except Exception as e:
                print(f"  [v7] PHASE0 failed: {type(e).__name__}: {e} — "
                      f"falling back to .plc init", flush=True)

        # albania1: optional fast-path — skip v4+Lap by loading a saved
        # post-Lap placement. Used for clean A/B testing of Hessian
        # variants without v4-baseline variance.
        load_post_lap = os.environ.get("PLACER_V7_LOAD_POST_LAP")
        if load_post_lap:
            load_path_pl = load_post_lap.format(name=benchmark.name)
            try:
                loaded = np.load(load_path_pl)
                portfolio_pos = torch.tensor(loaded, dtype=torch.float32)
                # Compute cost via official evaluator
                bench_load = Benchmark.load(bench_path)
                import importlib.util as _ilu
                v1_spec = _ilu.spec_from_file_location(
                    "_v1_load",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1 = _ilu.module_from_spec(v1_spec)
                v1_spec.loader.exec_module(v1)
                plc_load = v1._load_plc(bench_load.name)
                r = compute_proxy_cost(portfolio_pos, bench_load, plc_load)
                portfolio_cost = float(r["proxy_cost"])
                overlaps = int(r["overlap_count"])
                print(f"  [v7] LOAD_POST_LAP from {load_path_pl}: "
                      f"cost={portfolio_cost:.6f} overlaps={overlaps} "
                      f"(skipped v4+Lap)", flush=True)
            except Exception as e:
                print(f"  [v7] LOAD_POST_LAP failed: {e} — running v4+Lap",
                      flush=True)
                load_post_lap = None

        if not load_post_lap:
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

        # Phase 4: Laplacian soft-resolve. Skipped when LOAD_POST_LAP is
        # set (the loaded state is already post-Laplacian).
        if load_post_lap:
            self.LAPLACIAN_REFINE = False

        # Loaded into a fresh
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

        # albania1: save post-Lap state for clean A/B testing of
        # downstream variants. Set PLACER_V7_SAVE_POST_LAP=path to dump
        # the placement after v4+Laplacian. Then re-run with
        # PLACER_V7_LOAD_POST_LAP=path to skip v4+Lap and go straight
        # to Hessian. This isolates the Hessian effect from v4-baseline
        # variance (~0.005 per run).
        save_post_lap = os.environ.get("PLACER_V7_SAVE_POST_LAP")
        if save_post_lap:
            try:
                save_path_pl = save_post_lap.format(name=benchmark.name)
                Path(save_path_pl).parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path_pl, portfolio_pos.detach().cpu().numpy())
                # Also save the cost as metadata
                with open(save_path_pl + ".meta", "w") as fmeta:
                    fmeta.write(f"cost={portfolio_cost}\noverlaps={overlaps}\n")
                print(f"  [v7] saved post-Lap state to {save_path_pl} "
                      f"(cost={portfolio_cost:.6f})", flush=True)
            except Exception as e:
                print(f"  [v7] WARNING: post-Lap save failed: {e}",
                      flush=True)

        # ── Phase 4.6: Hessian negative-eigenvector escape ─────────────
        # At post-Laplacian state, compute Hessian of smooth surrogate.
        # If λ_min < 0, we're at a saddle: v_min is the curvature-down
        # escape direction (transition-state theory). Generate K
        # perturbed candidates by stepping along ±v_min at multiple
        # step sizes; run a SHORT pipeline from each in parallel; take
        # min, strict-improvement gate. Off by default;
        # PLACER_V7_HESSIAN=1 enables.
        if (os.environ.get("PLACER_V7_HESSIAN", "0") == "1"
                and overlaps == 0):
            hess_steps_str = os.environ.get(
                "PLACER_V7_HESSIAN_STEPS", "0.02,-0.02,0.05,-0.05")
            hess_steps = [float(s) for s in hess_steps_str.split(",") if s]
            hess_budget = int(os.environ.get(
                "PLACER_V7_HESSIAN_BUDGET", "300"))
            hess_n_lanczos = int(os.environ.get(
                "PLACER_V7_HESSIAN_LANCZOS", "50"))
            # Iterative Hessian: each iter crosses one saddle. Loop until
            # smooth surrogate's λ_min ≥ ε (true 2nd-order critical
            # point) OR max iters OR strict-improvement fails.
            hess_max_iters = int(os.environ.get(
                "PLACER_V7_HESSIAN_MAX_ITERS", "1"))
            hess_total_budget_s = int(os.environ.get(
                "PLACER_V7_HESSIAN_TOTAL_BUDGET", "0"))   # 0 = no cap
            t_hess_start = time.time()
            # albania2: spectral net criticality. Between iterations,
            # decompose the previous Hessian's negative eigenvectors
            # into per-net energy and amplify high-criticality nets in
            # the next iter's surrogate. Gated by env vars so we can
            # ablate cleanly. The base (incr.net_weight) — and therefore
            # the leaderboard cost — is unchanged; only the smooth
            # surrogate sees the override.
            spectral_on = (os.environ.get(
                "PLACER_V7_SPECTRAL_CRITICALITY", "0") == "1")
            spectral_gain = float(os.environ.get(
                "PLACER_V7_SPECTRAL_GAIN", "0.5"))
            net_weight_override = None
            for hess_iter in range(hess_max_iters):
                if (hess_total_budget_s > 0
                        and time.time() - t_hess_start
                        + hess_budget * 1.1 > hess_total_budget_s):
                    print(f"  [v7] hessian iter {hess_iter}: insufficient "
                          f"budget for another pass; stop",
                          flush=True)
                    break
                try:
                    self._last_hessian_eigeninfo = None
                    new_pos, new_cost = self._hessian_escape_phase(
                        portfolio_pos, portfolio_cost, bench_path,
                        hess_steps, hess_budget, hess_n_lanczos,
                        net_weight_override=net_weight_override)
                    if isinstance(new_pos, np.ndarray):
                        new_pos = torch.tensor(new_pos, dtype=torch.float32)
                    if new_cost >= portfolio_cost - 1e-7:
                        print(f"  [v7] hessian iter {hess_iter}: no further "
                              f"improvement; converged", flush=True)
                        break
                    print(f"  [v7] hessian iter {hess_iter}: cost "
                          f"{portfolio_cost:.6f} → {new_cost:.6f} "
                          f"(Δ {portfolio_cost-new_cost:+.4f})",
                          flush=True)
                    portfolio_pos = new_pos
                    portfolio_cost = float(new_cost)
                    overlaps = 0
                    # Spectral criticality reweighting for next iter.
                    if spectral_on:
                        eig = getattr(self,
                                      "_last_hessian_eigeninfo", None)
                        if (eig is not None
                                and eig.get("eigvals") is not None
                                and eig.get("eigvecs") is not None):
                            try:
                                from _spectral_criticality import (
                                    eigvec_net_criticality,
                                    apply_criticality_to_weights)
                                crit = eigvec_net_criticality(
                                    eig["eigvals"], eig["eigvecs"],
                                    eig["pin_macro"], eig["net_starts"],
                                    n_total=eig["n_total"])
                                base_w = eig["base_net_weight"]
                                new_w = apply_criticality_to_weights(
                                    base_w, crit, gain=spectral_gain)
                                k_topcrit = int((crit > 0.1).sum())
                                print(f"  [v7] spectral: criticality "
                                      f"computed; {k_topcrit}/{len(crit)} "
                                      f"nets > 0.1; base_w∈[{base_w.min():.2f},"
                                      f"{base_w.max():.2f}]→new_w∈"
                                      f"[{new_w.min():.2f},{new_w.max():.2f}], "
                                      f"gain={spectral_gain}", flush=True)
                                net_weight_override = new_w
                            except Exception as ce:
                                print(f"  [v7] spectral: criticality "
                                      f"err {type(ce).__name__}: {ce}",
                                      flush=True)
                                net_weight_override = None
                except Exception as e:
                    print(f"  [v7] hessian iter {hess_iter} err: "
                          f"{type(e).__name__}: {e}", flush=True)
                    break

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
                # zeus B6: JKO/Wasserstein-2 post-Adam refinement.
                # Applies N JKO proximal steps after Adam converges.
                # Each step: gradient → tentative target → Sinkhorn-transport.
                # Useful when Adam stalls near a basin with non-Euclidean
                # geometry (high-density region). Default off.
                jko_steps = int(os.environ.get(
                    "PLACER_V7_PHASE0_JKO_STEPS", "0"))
                if jko_steps > 0:
                    from _jko_step import jko_proximal_step
                    from _smooth_proxy import (
                        smooth_proxy_for_v7_v2, build_pin_to_net)
                    from _cell_window import build_window_indices
                    jko_tau = float(os.environ.get(
                        "PLACER_V7_PHASE0_JKO_TAU", "5.0"))
                    jko_alpha = float(os.environ.get(
                        "PLACER_V7_PHASE0_JKO_ALPHA", "0.5"))
                    jko_eps = float(os.environ.get(
                        "PLACER_V7_PHASE0_JKO_EPS", "10.0"))
                    jko_iters = int(os.environ.get(
                        "PLACER_V7_PHASE0_JKO_SINK_ITERS", "30"))
                    t_jko = time.time()
                    # Build tensors for the surrogate.
                    macro_pos_tj = adam_tensor.clone().detach().requires_grad_(True)
                    t_density = torch.zeros((), requires_grad=True)
                    t_cong = torch.zeros((), requires_grad=True)
                    pin_macro_tj = torch.tensor(np.asarray(incr.pin_macro), dtype=torch.long)
                    pin_xoff_tj = torch.tensor(np.asarray(incr.pin_xoff), dtype=torch.float32)
                    pin_yoff_tj = torch.tensor(np.asarray(incr.pin_yoff), dtype=torch.float32)
                    net_starts_tj = torch.tensor(np.asarray(incr.net_starts), dtype=torch.long)
                    net_weight_tj = torch.tensor(np.asarray(incr.net_weight), dtype=torch.float32)
                    macro_w_tj = torch.tensor(np.asarray(incr.macro_w), dtype=torch.float32)
                    macro_h_tj = torch.tensor(np.asarray(incr.macro_h), dtype=torch.float32)
                    V_smooth_tj = torch.tensor(np.asarray(incr.V_routing_smooth), dtype=torch.float32)
                    H_smooth_tj = torch.tensor(np.asarray(incr.H_routing_smooth), dtype=torch.float32)
                    pin_to_net_tj = build_pin_to_net(net_starts_tj)
                    n_nets_tj = int(net_weight_tj.shape[0])
                    K_d_tj = max(1, int(incr.n_cells * k_dens_frac))
                    K_c_tj = max(1, int(2 * incr.n_cells * k_cong_frac))
                    cell_idx_tj, _ = build_window_indices(
                        macro_pos_tj.detach(), macro_w_tj, macro_h_tj,
                        grid_col=incr.grid_col, grid_row=incr.grid_row,
                        grid_w=incr.grid_width, grid_h=incr.grid_height,
                        margin_cells=4)
                    for jko_iter in range(jko_steps):
                        # Forward + grad for ∇U at current state.
                        macro_pos_tj.requires_grad_(True)
                        if macro_pos_tj.grad is not None:
                            macro_pos_tj.grad.zero_()
                        loss_tj, _ = smooth_proxy_for_v7_v2(
                            macro_pos_tj, t_density, t_cong,
                            macro_w=macro_w_tj, macro_h=macro_h_tj,
                            pin_macro=pin_macro_tj, pin_xoff=pin_xoff_tj,
                            pin_yoff=pin_yoff_tj, pin_to_net=pin_to_net_tj,
                            net_weight=net_weight_tj, n_nets=n_nets_tj,
                            grid_col=incr.grid_col, grid_row=incr.grid_row,
                            grid_w=incr.grid_width, grid_h=incr.grid_height,
                            grid_v_routes=incr.grid_v_routes,
                            grid_h_routes=incr.grid_h_routes,
                            V_smooth_frozen=V_smooth_tj,
                            H_smooth_frozen=H_smooth_tj,
                            cw=float(incr.cw), ch=float(incr.ch),
                            net_cnt=float(incr.net_cnt),
                            K_density=K_d_tj, K_cong=K_c_tj,
                            cell_area=incr.grid_area,
                            vrouting_alloc=incr.vrouting_alloc,
                            hrouting_alloc=incr.hrouting_alloc,
                            tau_lse=50.0, mu_softplus=100.0,
                            proximal_pos=adam_tensor,
                            proximal_weight=0.0,    # off for JKO
                            hard_only_proximal=True,
                            n_hard=incr.n_hard,
                            cell_idx_density=cell_idx_tj,
                            cell_idx_cong=cell_idx_tj,
                        )
                        loss_tj.backward()
                        grad_U_tj = macro_pos_tj.grad.detach().clone()
                        # JKO step.
                        with torch.no_grad():
                            x_new, jko_diag = jko_proximal_step(
                                macro_pos_tj, grad_U_tj,
                                tau=jko_tau, alpha=jko_alpha,
                                sinkhorn_eps=jko_eps,
                                sinkhorn_iters=jko_iters,
                                n_hard=incr.n_hard, soft_only=True)
                            macro_pos_tj = x_new.requires_grad_(True)
                    adam_tensor = macro_pos_tj.detach().clone()
                    pos_adam = adam_tensor.cpu().numpy().astype(np.float64)
                    print(f"  [v7] phase0 JKO: {jko_steps} steps "
                          f"τ={jko_tau:.1f} α={jko_alpha:.2f} ε={jko_eps:.1f} "
                          f"({time.time()-t_jko:.1f}s)", flush=True)
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

        # ── Phase 4.7: Greedy top-K congestion eviction ────────────────
        # Identify softs whose footprint touches a top-K hot cell;
        # search radius R for a cooler cell; validate via exact
        # compute_proxy_cost; strict accept. Bypasses the smooth-vs-
        # exact divergence that broke Adam, and the local-minimizer
        # plateau that broke basin-hop. Off by default;
        # PLACER_V7_EVICT=1 enables.
        if (os.environ.get("PLACER_V7_EVICT", "0") == "1"
                and overlaps == 0):
            evict_top_k = float(os.environ.get(
                "PLACER_V7_EVICT_TOP_K", "0.05"))
            evict_radius = int(os.environ.get(
                "PLACER_V7_EVICT_RADIUS", "5"))
            evict_passes = int(os.environ.get(
                "PLACER_V7_EVICT_PASSES", "3"))
            evict_max_per_pass_env = os.environ.get(
                "PLACER_V7_EVICT_MAX_PER_PASS", "")
            evict_max_per_pass = (int(evict_max_per_pass_env)
                                   if evict_max_per_pass_env else None)
            try:
                from _eviction import evict_hot_softs
                import importlib.util as _ilu
                v1_spec = _ilu.spec_from_file_location(
                    "_v1_v7_evict",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1 = _ilu.module_from_spec(v1_spec)
                v1_spec.loader.exec_module(v1)
                bench_for_evict = Benchmark.load(bench_path)
                plc_e = v1._load_plc(bench_for_evict.name)
                incr_e = v1.IncrementalEvaluator(plc_e, bench_for_evict)
                full_np_e = portfolio_pos.cpu().numpy()
                n_hard_e = bench_for_evict.num_hard_macros
                incr_e.sync_positions(full_np_e[:n_hard_e])
                incr_e.macro_pos[n_hard_e:] = \
                    full_np_e[n_hard_e:].astype(incr_e.macro_pos.dtype)
                incr_e._recompute_pin_positions()
                incr_e._full_recompute_wl()
                incr_e._full_recompute_density()
                incr_e._full_recompute_congestion()

                t_evict_start = time.time()
                evict_pos, evict_cost, n_accepted = evict_hot_softs(
                    incr_e, bench_for_evict, plc_e,
                    top_k_frac=evict_top_k,
                    radius_cells=evict_radius,
                    n_passes=evict_passes,
                    max_softs_per_pass=evict_max_per_pass,
                    verbose=True,
                )
                print(f"  [v7] evict: {n_accepted} accepted moves in "
                      f"{time.time()-t_evict_start:.1f}s; "
                      f"exact-proxy {portfolio_cost:.6f} → "
                      f"{evict_cost:.6f}", flush=True)
                if evict_cost < portfolio_cost - 1e-7:
                    print(f"  [v7] EVICT WIN: {evict_cost:.6f} < "
                          f"{portfolio_cost:.6f} "
                          f"(Δ {portfolio_cost-evict_cost:+.4f})",
                          flush=True)
                    portfolio_cost = evict_cost
                    portfolio_pos = torch.tensor(
                        evict_pos, dtype=torch.float32)
                    overlaps = 0
                else:
                    print(f"  [v7] evict: no net improvement; "
                          f"keeping post-Laplacian "
                          f"({portfolio_cost:.6f})", flush=True)
            except Exception as e:
                print(f"  [v7] evict err: {type(e).__name__}: {e}; "
                      f"keeping post-Laplacian", flush=True)

        # ── Phase 4.8: Sinkhorn optimal-transport eviction ────────────
        # Globally optimal soft → cell assignment via Sinkhorn (entropy-
        # regularized OT). Cost = current cong[cell] + α·dist²(soft, cell).
        # Apply: each soft moves to argmax(T) cell. Validate via exact
        # compute_proxy_cost; if full-apply fails, fall back to top-K
        # most-confident partial application. Off by default;
        # PLACER_V7_SINKHORN=1 enables.
        if (os.environ.get("PLACER_V7_SINKHORN", "0") == "1"
                and overlaps == 0):
            sk_alpha = float(os.environ.get(
                "PLACER_V7_SINKHORN_ALPHA", "0.5"))
            sk_eps = float(os.environ.get(
                "PLACER_V7_SINKHORN_EPS", "0.05"))
            sk_iters = int(os.environ.get(
                "PLACER_V7_SINKHORN_ITERS", "50"))
            try:
                from _sinkhorn_ot import sinkhorn_evict
                import importlib.util as _ilu
                v1_spec = _ilu.spec_from_file_location(
                    "_v1_v7_sk",
                    str(_HERE.parent / "vmallela" / "placer.py"))
                v1 = _ilu.module_from_spec(v1_spec)
                v1_spec.loader.exec_module(v1)
                bench_for_sk = Benchmark.load(bench_path)
                plc_sk = v1._load_plc(bench_for_sk.name)
                incr_sk = v1.IncrementalEvaluator(plc_sk, bench_for_sk)
                full_np_sk = portfolio_pos.cpu().numpy()
                n_hard_sk = bench_for_sk.num_hard_macros
                incr_sk.sync_positions(full_np_sk[:n_hard_sk])
                incr_sk.macro_pos[n_hard_sk:] = \
                    full_np_sk[n_hard_sk:].astype(incr_sk.macro_pos.dtype)
                incr_sk._recompute_pin_positions()
                incr_sk._full_recompute_wl()
                incr_sk._full_recompute_density()
                incr_sk._full_recompute_congestion()

                t_sk_start = time.time()
                sk_pos, sk_cost, sk_diag = sinkhorn_evict(
                    incr_sk, bench_for_sk, plc_sk,
                    alpha_hpwl=sk_alpha, eps=sk_eps, iters=sk_iters,
                    verbose=True)
                print(f"  [v7] sinkhorn: {time.time()-t_sk_start:.1f}s "
                      f"total; exact-proxy {portfolio_cost:.6f} → "
                      f"{sk_cost:.6f}", flush=True)
                if sk_cost < portfolio_cost - 1e-7:
                    print(f"  [v7] SINKHORN WIN: {sk_cost:.6f} < "
                          f"{portfolio_cost:.6f} "
                          f"(Δ {portfolio_cost-sk_cost:+.4f})",
                          flush=True)
                    portfolio_cost = sk_cost
                    portfolio_pos = torch.tensor(
                        sk_pos, dtype=torch.float32)
                    overlaps = 0
                else:
                    print(f"  [v7] sinkhorn: no net improvement; "
                          f"keeping post-Laplacian", flush=True)
            except Exception as e:
                print(f"  [v7] sinkhorn err: {type(e).__name__}: {e}; "
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

        # albania1: Klein-4 orientation flip → orientations.pt sidecar.
        # Tier 1 proxy (TILOS) ignores orientations and uses default 'N',
        # so this cannot regress proxy. Tier 2 (OpenROAD WNS/TNS/Area)
        # picks the sidecar up via the TCL generator if present.
        if (os.environ.get("PLACER_V7_ORIENTATION_FLIP", "0") == "1"
                and overlaps == 0):
            try:
                self._write_orientation_sidecar(
                    bench_path, portfolio_pos, save_template, benchmark)
            except Exception as e:
                print(f"  [v7] WARNING: orientation flip failed: {e}",
                      flush=True)
        return portfolio_pos

    def _write_orientation_sidecar(self, bench_path, portfolio_pos,
                                    save_template, benchmark):
        """Run Klein-4 orientation greedy and persist orientations.pt
        sidecar. Path: same as placement save_template with
        ``.orientations.pt`` suffix; falls back to a per-bench path
        under ``orientations/`` next to the benchmarks dir if no
        save_template is set.
        """
        from _orientation_flip import klein4_orient, save_orientation_sidecar
        import importlib.util as _ilu
        bench = Benchmark.load(bench_path)
        n_hard = bench.num_hard_macros
        if n_hard == 0:
            print("  [v7] orientation: no hard macros, skipping",
                  flush=True)
            return
        v1_spec = _ilu.spec_from_file_location(
            "_v1_v7_orient",
            str(_HERE.parent / "vmallela" / "placer.py"))
        v1 = _ilu.module_from_spec(v1_spec)
        v1_spec.loader.exec_module(v1)
        plc = v1._load_plc(bench.name)
        incr = v1.IncrementalEvaluator(plc, bench)
        # Sync incr to final positions
        full_np = portfolio_pos.detach().cpu().numpy()
        incr.macro_pos[:] = full_np
        incr._recompute_pin_positions()
        n_passes = int(os.environ.get("PLACER_V7_ORIENTATION_PASSES", "2"))
        # Build pin_to_net (incr stores net_starts; need flat per-pin id)
        pin_to_net = np.zeros(int(incr.pin_macro.shape[0]), dtype=np.int64)
        net_starts = np.asarray(incr.net_starts)
        for nid in range(net_starts.shape[0] - 1):
            pin_to_net[net_starts[nid]:net_starts[nid + 1]] = nid
        n_nets = int(net_starts.shape[0] - 1)
        orientations, info = klein4_orient(
            macro_pos=np.asarray(incr.macro_pos),
            macro_w=np.asarray(incr.macro_w),
            macro_h=np.asarray(incr.macro_h),
            pin_macro=np.asarray(incr.pin_macro),
            pin_xoff=np.asarray(incr.pin_xoff),
            pin_yoff=np.asarray(incr.pin_yoff),
            pin_to_net=pin_to_net,
            net_weight=np.asarray(incr.net_weight),
            n_hard=n_hard,
            n_nets=n_nets,
            n_passes=n_passes,
            verbose=True,
        )
        if save_template:
            try:
                save_path = save_template.format(name=benchmark.name)
            except Exception:
                save_path = save_template
            sidecar_path = save_path + ".orientations.pt"
        else:
            sidecar_dir = _HERE.parents[1] / "orientations"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = str(sidecar_dir / f"{benchmark.name}.orientations.pt")
        save_orientation_sidecar(orientations, sidecar_path)
        print(f"  [v7] orientations: HPWL "
              f"{info['initial_hpwl']:.1f} → {info['final_hpwl']:.1f} "
              f"(Δ {info['delta_hpwl']:+.1f}); "
              f"flipped {info['n_flipped']}/{info['n_hard']} "
              f"({info['counts']}); saved to {sidecar_path}",
              flush=True)

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
            elif perturb_mode == "levy":
                # Lévy α-stable noise: heavy-tailed, occasional large jumps.
                # α=2 is Gaussian; α=1 is Cauchy. α≈1.5 gives jumps that are
                # mostly small but ~5% of samples make big topology-crossing
                # moves. Provably better mixing for multimodal landscapes
                # than Gaussian (Yang-Deb 2009; Pavlyukevich 2007).
                from scipy.stats import levy_stable
                alpha = float(os.environ.get(
                    "PLACER_V7_LEVY_ALPHA", "1.5"))
                sigma_soft = self.BASIN_HOP_SIGMA0 * (0.6 ** (hop - 1)) * canvas_diag
                sigma_hard = sigma_soft * 0.25
                perturbed = best_pos.cpu().numpy().astype(np.float64).copy()
                seed_h = int(rng.integers(0, 2**31 - 1))
                seed_s = int(rng.integers(0, 2**31 - 1))
                noise = np.zeros_like(perturbed)
                noise[:n_hard] = levy_stable.rvs(
                    alpha=alpha, beta=0.0, scale=sigma_hard,
                    size=(n_hard, 2), random_state=seed_h)
                noise[n_hard:n_total] = levy_stable.rvs(
                    alpha=alpha, beta=0.0, scale=sigma_soft,
                    size=(n_total - n_hard, 2), random_state=seed_s)
                perturbed = perturbed + noise
                perturbed[:, 0] = np.clip(perturbed[:, 0], 0.0, cw)
                perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, ch)
                # Stats on the noise to track tail heaviness in the log
                soft_jumps = np.linalg.norm(noise[n_hard:n_total], axis=1)
                p99 = float(np.percentile(soft_jumps, 99))
                pmax = float(np.max(soft_jumps))
                print(f"  [v7.hop {hop}] LEVY α={alpha} σ_soft={sigma_soft:.2f} "
                      f"(p99={p99:.2f} max={pmax:.2f}), running single-worker "
                      f"pipeline at {hop_budget}s...", flush=True)
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

    def _hessian_escape_phase(self, current_pos, current_cost, bench_path,
                                step_sizes, hop_budget, n_lanczos_iters,
                                net_weight_override=None):
        """Phase 4.6: Hessian negative-eigenvector escape.

        Compute Hessian of smooth surrogate at the post-Laplacian state.
        The smallest-eigenvalue eigenvector v_min is the curvature-down
        direction (transition-state theory). Generate len(step_sizes)
        candidates by perturbing along step·v_min, run a reduced
        pipeline from each in parallel via mp.Pool, take min, strict
        improvement gate.
        """
        import multiprocessing as _mp
        from _hessian_escape import hessian_escape_step
        from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                     cvar_smooth)
        from _cell_window import (build_window_indices, smooth_density_grid,
                                    smooth_macro_blockage,
                                    electrostatic_density_energy,
                                    electrostatic_density_energy_normalized)
        from _hessian_worker import hessian_candidate_worker
        import importlib.util as _ilu

        bench = Benchmark.load(bench_path)
        n_hard = bench.num_hard_macros
        canvas_diag = math.hypot(
            float(bench.canvas_width), float(bench.canvas_height))

        # Build IncrementalEvaluator at the current placement
        v1_spec = _ilu.spec_from_file_location(
            "_v1_v7_hess",
            str(_HERE.parent / "vmallela" / "placer.py"))
        v1 = _ilu.module_from_spec(v1_spec)
        v1_spec.loader.exec_module(v1)
        plc = v1._load_plc(bench.name)
        incr = v1.IncrementalEvaluator(plc, bench)
        incr.macro_pos[:] = current_pos.cpu().numpy()
        incr._recompute_pin_positions()
        incr._full_recompute_wl()
        incr._full_recompute_density()
        incr._full_recompute_congestion()

        # Build smooth-surrogate closure (HPWL + density only; cong is
        # too far from exact, omitted)
        device = (torch.device("mps")
                  if torch.backends.mps.is_available()
                  else torch.device("cpu"))
        macro_pos_t = torch.tensor(
            np.asarray(incr.macro_pos), dtype=torch.float32, device=device)
        pin_macro_t = torch.tensor(
            np.asarray(incr.pin_macro), dtype=torch.long, device=device)
        pin_xoff_t = torch.tensor(
            np.asarray(incr.pin_xoff), dtype=torch.float32, device=device)
        pin_yoff_t = torch.tensor(
            np.asarray(incr.pin_yoff), dtype=torch.float32, device=device)
        net_starts_t = torch.tensor(
            np.asarray(incr.net_starts), dtype=torch.long, device=device)
        # albania2: spectral criticality reweighting. If
        # `net_weight_override` is supplied (computed from the previous
        # Hessian iter's eigvecs), use it; else fall back to incr's own
        # net_weight (which carries the bench's STA-derived path-count
        # weights). This decouples surrogate weights from the exact-cost
        # weights — the leaderboard cost is always evaluated with the
        # original `incr.net_weight` and unaffected by this override.
        _nw_src = (np.asarray(net_weight_override)
                   if net_weight_override is not None
                   else np.asarray(incr.net_weight))
        net_weight_t = torch.tensor(
            _nw_src, dtype=torch.float32, device=device)
        if net_weight_override is not None:
            print(f"  [v7] hessian: using SPECTRAL net_weight override "
                  f"(min={_nw_src.min():.3f}, max={_nw_src.max():.3f}, "
                  f"mean={_nw_src.mean():.3f})", flush=True)
        macro_w_t = torch.tensor(
            np.asarray(incr.macro_w), dtype=torch.float32, device=device)
        macro_h_t = torch.tensor(
            np.asarray(incr.macro_h), dtype=torch.float32, device=device)
        # albania1: halo for surrogate-only density. Inflate macro
        # extents by halo_frac when computing density/cong; HPWL still
        # uses real pin offsets (and thus real macro footprints). Net
        # effect: surrogate prefers placements that leave routing
        # channels around macros, while exact-cost HPWL stays clean.
        # Strict-improvement gate against exact proxy preserves Tier 1.
        halo_frac = float(os.environ.get("PLACER_V7_HALO_FRAC", "0.0"))
        if halo_frac > 0.0:
            macro_w_haloed = macro_w_t * (1.0 + halo_frac)
            macro_h_haloed = macro_h_t * (1.0 + halo_frac)
            print(f"  [v7] halo: surrogate density inflated by "
                  f"{halo_frac*100:.1f}% (HPWL unchanged)", flush=True)
        else:
            macro_w_haloed = macro_w_t
            macro_h_haloed = macro_h_t
        pin_to_net_t = build_pin_to_net(net_starts_t)
        n_nets = int(net_weight_t.shape[0])
        cell_idx_d, _ = build_window_indices(
            macro_pos_t.detach(), macro_w_haloed, macro_h_haloed,
            grid_col=incr.grid_col, grid_row=incr.grid_row,
            grid_w=incr.grid_width, grid_h=incr.grid_height,
            margin_cells=4)
        cw_f, ch_f = float(incr.cw), float(incr.ch)
        net_cnt = float(incr.net_cnt)
        # albania1: PLACER_V7_K_DENS_FRAC also tightens density CVaR
        # in the Hessian surrogate (was hardcoded 0.10). Tighter k
        # concentrates the saddle direction on worst-case pinch points.
        k_dens_frac = float(os.environ.get("PLACER_V7_K_DENS_FRAC", "0.10"))
        K_d = max(1, int(k_dens_frac * incr.n_cells))
        hpwl_weight = float(os.environ.get(
            "PLACER_V7_HESSIAN_HPWL_WEIGHT", "1.0"))
        dens_weight = float(os.environ.get(
            "PLACER_V7_HESSIAN_DENS_WEIGHT", "0.5"))

        # albania1: CONGESTION in the Hessian surrogate. Critical fix —
        # without this term the saddle escape only sees HPWL+density
        # while ~73 % of proxy variance lives in congestion (across the
        # 17 IBM benches; see research/lower_bounds/FINDINGS.md). The
        # math mirrors smooth_proxy_for_v7_v2 in _smooth_proxy.py:
        # combined V/H congestion = V_routing_smooth (frozen, from per-net
        # RUDY) + V_macro(x)/grid_v_routes (differentiable macro blockage),
        # CVaR top-(K_c / 2·n_cells) over union.
        cong_enabled = (os.environ.get("PLACER_V7_HESSIAN_CONG", "1") == "1")
        if cong_enabled:
            cell_idx_c, _ = build_window_indices(
                macro_pos_t.detach(), macro_w_haloed, macro_h_haloed,
                grid_col=incr.grid_col, grid_row=incr.grid_row,
                grid_w=incr.grid_width, grid_h=incr.grid_height,
                margin_cells=4)
            V_smooth_frozen = torch.tensor(
                np.asarray(incr.V_routing_smooth),
                dtype=torch.float32, device=device)
            H_smooth_frozen = torch.tensor(
                np.asarray(incr.H_routing_smooth),
                dtype=torch.float32, device=device)
            v_alloc = float(np.asarray(incr.vrouting_alloc).mean())
            h_alloc = float(np.asarray(incr.hrouting_alloc).mean())
            grid_v_routes = float(incr.grid_v_routes)
            grid_h_routes = float(incr.grid_h_routes)
            k_cong_frac = float(
                os.environ.get("PLACER_V7_K_CONG_FRAC", "0.05"))
            K_c = max(1, int(2 * incr.n_cells * k_cong_frac))
            base_cw = float(
                os.environ.get("PLACER_V7_HESSIAN_CONG_WEIGHT", "0.5"))
            if os.environ.get("PLACER_V7_HESSIAN_AUTO_CONG", "0") == "1":
                cong_weight = _auto_cong_weight(bench.name, base_cw)
                resid = _BENCH_CONG_RESIDUAL.get(bench.name, 0.0)
                print(f"  [v7] hessian: AUTO_CONG bench={bench.name} "
                      f"residual={resid:+.3f} → cong_weight={cong_weight:.2f} "
                      f"(base {base_cw} × {cong_weight/max(base_cw,1e-9):.1f}×)",
                      flush=True)
            else:
                cong_weight = base_cw
            print(f"  [v7] hessian: congestion ENABLED "
                  f"(K_c={K_c}/{2*incr.n_cells}, weight={cong_weight}, "
                  f"v_alloc={v_alloc:.4f}, h_alloc={h_alloc:.4f})",
                  flush=True)
        else:
            cell_idx_c = None
            V_smooth_frozen = H_smooth_frozen = None
            v_alloc = h_alloc = grid_v_routes = grid_h_routes = 0.0
            K_c = 0
            cong_weight = 0.0
            print("  [v7] hessian: congestion DISABLED "
                  "(PLACER_V7_HESSIAN_CONG=0)", flush=True)

        # zeus: Differentiable RUDY routing demand. When enabled, the
        # frozen V_routing_smooth/H_routing_smooth are replaced with a
        # per-call RUDY computation that flows gradients through pin
        # positions to macro_pos. See _rudy_smooth.smooth_rudy_routing.
        rudy_enabled = (cong_enabled and
                         os.environ.get("PLACER_V7_HESSIAN_RUDY", "0") == "1")
        net_cell_idx_t = None
        rudy_scale = 1.0
        pair_net_t = pair_cell_t = None
        if rudy_enabled:
            from _rudy_smooth import (build_net_window_indices_sparse,
                                       smooth_rudy_routing_sparse)
            rudy_margin = int(os.environ.get(
                "PLACER_V7_HESSIAN_RUDY_MARGIN", "4"))
            rudy_max_window = int(os.environ.get(
                "PLACER_V7_HESSIAN_RUDY_MAX_WINDOW", "256"))
            # Compute pin coords at current state for window construction.
            with torch.no_grad():
                _is_port = (pin_macro_t < 0)
                _safe = torch.where(_is_port,
                                     torch.zeros_like(pin_macro_t),
                                     pin_macro_t)
                _macro_xy = macro_pos_t[_safe]
                pin_x_init = torch.where(_is_port, pin_xoff_t,
                                          _macro_xy[:, 0] + pin_xoff_t)
                pin_y_init = torch.where(_is_port, pin_yoff_t,
                                          _macro_xy[:, 1] + pin_yoff_t)
            t_rudy_win = time.time()
            pair_net_t, pair_cell_t, n_pairs, n_dropped = \
                build_net_window_indices_sparse(
                    pin_x_init, pin_y_init, pin_to_net_t, n_nets,
                    incr.grid_col, incr.grid_row,
                    incr.grid_width, incr.grid_height,
                    margin_cells=rudy_margin,
                    max_window_cells=rudy_max_window)
            # Auto-scale so V_rudy's median matches V_smooth_frozen's
            # median at the initial state. Keeps the cong surrogate's
            # absolute magnitude comparable to the verified config —
            # CVaR top-K and cong_weight calibration are preserved.
            rudy_scale_env = float(
                os.environ.get("PLACER_V7_HESSIAN_RUDY_SCALE", "0"))
            with torch.no_grad():
                V_init, H_init = smooth_rudy_routing_sparse(
                    pin_x_init, pin_y_init, pin_to_net_t, net_weight_t,
                    n_nets, pair_net_t, pair_cell_t,
                    incr.grid_col, incr.grid_row,
                    incr.grid_width, incr.grid_height,
                    n_cells=incr.n_cells)
                if rudy_scale_env > 0.0:
                    rudy_scale = rudy_scale_env
                else:
                    Vp = V_init[V_init > 1e-9]
                    Vf = V_smooth_frozen[V_smooth_frozen > 1e-9] if V_smooth_frozen is not None else None
                    v_med_rudy = float(Vp.median().item()) if Vp.numel() > 0 else 1.0
                    v_med_fr = float(Vf.median().item()) if (Vf is not None and Vf.numel() > 0) else 1.0
                    rudy_scale = (v_med_fr * float(grid_v_routes)
                                   / max(v_med_rudy, 1e-9))
            print(f"  [v7] hessian: RUDY differentiable routing ENABLED "
                  f"(n_pairs={n_pairs}, dropped={n_dropped}, "
                  f"margin={rudy_margin}, max_win={rudy_max_window}, "
                  f"scale={rudy_scale:.4f}, "
                  f"V_init.med={float(V_init[V_init>1e-9].median()) if (V_init>1e-9).any() else 0:.4e}, "
                  f"V_fr.med={(float(V_smooth_frozen[V_smooth_frozen>1e-9].median()) if (V_smooth_frozen is not None and (V_smooth_frozen>1e-9).any()) else 0):.4e}, "
                  f"win {time.time()-t_rudy_win:.2f}s)", flush=True)

        electro_enabled = (
            os.environ.get("PLACER_V7_HESSIAN_ELECTROSTATIC", "0") == "1")
        electro_weight = float(
            os.environ.get("PLACER_V7_HESSIAN_ELECTRO_WEIGHT", "1.0"))
        # Hybrid mode: keep both CVaR and electrostatic terms in the
        # surrogate. The Hessian eigvec captures BOTH local hotspots
        # (CVaR) and long-range structure (electrostatic). Default: when
        # electro is enabled, hybrid is OFF (pure electro replaces CVaR).
        # Set hybrid=1 to ADD electro on top of CVaR rather than replace.
        hybrid_density = (
            os.environ.get("PLACER_V7_HESSIAN_HYBRID_DENSITY", "0") == "1")
        if electro_enabled:
            mode = "HYBRID (CVaR + electro)" if hybrid_density else "ELECTRO-ONLY"
            print(f"  [v7] hessian: {mode} density (DREAMPlace-style) "
                  f"ENABLED (electro_weight={electro_weight}, "
                  f"dens_weight={dens_weight if hybrid_density else 0})",
                  flush=True)

        # albania2: HPWL surrogate model selector. "lse" = LSE-HPWL (current
        # default — only bbox-extreme pins drive gradient at τ=50). "b2b" =
        # pairwise-Manhattan / (K-1) where every pin in a net pulls every
        # other pin (denser gradient, equals HPWL exactly at K=2).
        hpwl_model = os.environ.get("PLACER_V7_HPWL_MODEL", "lse").lower()
        if hpwl_model == "b2b":
            from _smooth_proxy import (build_b2b_pair_indices,
                                          b2b_hpwl_vectorized)
            pair_a, pair_b, pair_w = build_b2b_pair_indices(
                pin_to_net_t.cpu(), n_nets)
            pair_a = pair_a.to(device)
            pair_b = pair_b.to(device)
            pair_w = pair_w.to(device).to(torch.float32)
            pair_to_net = pin_to_net_t[pair_a]
            print(f"  [v7] hessian: HPWL_MODEL=b2b "
                  f"(n_pairs={int(pair_a.shape[0])}, "
                  f"avg/net={float(pair_a.shape[0])/max(n_nets,1):.2f})",
                  flush=True)

        def smooth_proxy_call(macro_pos_var):
            is_port = (pin_macro_t < 0)
            safe = torch.where(is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
            macro_xy = macro_pos_var[safe]
            pin_x = torch.where(is_port, pin_xoff_t, macro_xy[:, 0] + pin_xoff_t)
            pin_y = torch.where(is_port, pin_yoff_t, macro_xy[:, 1] + pin_yoff_t)
            if hpwl_model == "b2b":
                hpwl = b2b_hpwl_vectorized(
                    pin_x, pin_y, pair_a, pair_b, pair_w, pair_to_net,
                    net_weight_t, n_nets,
                    cw=cw_f, ch=ch_f, net_cnt=net_cnt)
            else:
                hpwl = lse_hpwl_vectorized(
                    pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
                    cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
            rho = smooth_density_grid(
                macro_pos_var, macro_w_haloed, macro_h_haloed, cell_idx_d,
                incr.grid_col, incr.grid_row,
                incr.grid_width, incr.grid_height,
                n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
            if electro_enabled:
                # DREAMPlace-style: Poisson energy of the density
                # distribution (global structure). Normalized variant
                # divides by var(ρ)·canvas_area for scale-balance with
                # HPWL_LSE (~1.0 typical magnitude).
                use_normalized = (
                    os.environ.get("PLACER_V7_HESSIAN_ELECTRO_NORM", "0") == "1")
                if use_normalized:
                    density_term = electrostatic_density_energy_normalized(
                        rho, incr.grid_row, incr.grid_col,
                        grid_w=float(incr.grid_width),
                        grid_h=float(incr.grid_height))
                else:
                    density_term = electrostatic_density_energy(
                        rho, incr.grid_row, incr.grid_col,
                        grid_w=float(incr.grid_width),
                        grid_h=float(incr.grid_height))
                if hybrid_density:
                    # Hybrid: keep CVaR (local hotspots) AND add electro.
                    with torch.no_grad():
                        t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
                    density_smooth = cvar_smooth(
                        rho.unsqueeze(0), K_d, t_d.detach(),
                        mu=100.0).squeeze()
                    loss = (hpwl_weight * hpwl
                            + dens_weight * density_smooth
                            + electro_weight * density_term)
                else:
                    # Pure electro replaces CVaR.
                    loss = hpwl_weight * hpwl + electro_weight * density_term
            else:
                with torch.no_grad():
                    t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
                density_smooth = cvar_smooth(
                    rho.unsqueeze(0), K_d, t_d.detach(),
                    mu=100.0).squeeze()
                loss = hpwl_weight * hpwl + dens_weight * density_smooth
            if cong_enabled:
                V_macro, H_macro = smooth_macro_blockage(
                    macro_pos_var, macro_w_haloed, macro_h_haloed,
                    cell_idx_c,
                    incr.grid_col, incr.grid_row,
                    incr.grid_width, incr.grid_height,
                    n_cells=incr.n_cells,
                    vrouting_alloc=v_alloc,
                    hrouting_alloc=h_alloc,
                    mu=100.0)
                if rudy_enabled:
                    # zeus: differentiable per-net routing demand (sparse
                    # COO) replaces the frozen V/H_routing_smooth, so the
                    # eigvec direction tracks the live congestion gradient
                    # (not the stale one). See _rudy_smooth.py.
                    from _rudy_smooth import smooth_rudy_routing_sparse
                    V_rudy, H_rudy = smooth_rudy_routing_sparse(
                        pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
                        pair_net_t, pair_cell_t,
                        incr.grid_col, incr.grid_row,
                        incr.grid_width, incr.grid_height,
                        n_cells=incr.n_cells)
                    V_total = (rudy_scale * V_rudy + V_macro) / max(grid_v_routes, 1e-9)
                    H_total = (rudy_scale * H_rudy + H_macro) / max(grid_h_routes, 1e-9)
                else:
                    V_total = V_smooth_frozen + V_macro / max(grid_v_routes, 1e-9)
                    H_total = H_smooth_frozen + H_macro / max(grid_h_routes, 1e-9)
                combined = torch.cat([V_total, H_total], dim=0)
                # zeus B3: cong aggregator. "cvar" (default) | "l1" | "lp"
                # | "linf". L1/Lp/Linf give sparse gradients concentrated on
                # the hottest cells — closer to the true exact cong gradient.
                cong_agg = os.environ.get(
                    "PLACER_V7_HESSIAN_CONG_AGG", "cvar").lower()
                if cong_agg == "cvar":
                    with torch.no_grad():
                        t_c = torch.quantile(
                            combined, 1.0 - K_c / (2 * incr.n_cells))
                    cong_smooth = cvar_smooth(
                        combined.unsqueeze(0), K_c, t_c.detach(),
                        mu=100.0).squeeze()
                else:
                    # Build a per-bench target from the current quantile:
                    # cells exceeding the (1 − K_c / 2n_cells) quantile
                    # are the "hot" cells — same effective active set as CVaR.
                    with torch.no_grad():
                        t_c = torch.quantile(
                            combined, 1.0 - K_c / (2 * incr.n_cells))
                    if cong_agg == "l1":
                        from _smooth_proxy import l1_excess
                        cong_smooth = (l1_excess(
                            combined.unsqueeze(0), t_c.detach(),
                            mu=100.0).squeeze() / K_c)
                    elif cong_agg == "lp":
                        from _smooth_proxy import lp_excess
                        p_lp = float(os.environ.get(
                            "PLACER_V7_HESSIAN_CONG_LP_P", "2.0"))
                        # K_c^{1/p} normalization so this is dimensionally
                        # comparable to cvar's "per-cell mean of excess".
                        cong_smooth = (lp_excess(
                            combined.unsqueeze(0), t_c.detach(),
                            p=p_lp, mu=100.0).squeeze()
                                        / (K_c ** (1.0 / max(p_lp, 1e-3))))
                    elif cong_agg == "linf":
                        from _smooth_proxy import linf_excess
                        tau_linf = float(os.environ.get(
                            "PLACER_V7_HESSIAN_CONG_LINF_TAU", "30.0"))
                        cong_smooth = linf_excess(
                            combined.unsqueeze(0), t_c.detach(),
                            tau=tau_linf).squeeze()
                    else:
                        raise ValueError(
                            f"Unknown CONG_AGG: {cong_agg}")
                loss = loss + cong_weight * cong_smooth
            return loss

        # albania1: AUTO_LAMBDA_SCAN — pre-Hessian sweep over cong_weight
        # candidates to find the value that MAXIMIZES |λ_min| (deepest
        # saddle). Physics: deeper |λ_min| = stronger curvature in
        # eigvec direction = larger optimal step → more effective
        # saddle escape. From the cong-weight sensitivity test on ibm06,
        # |λ_min| varies non-monotonically with cong_weight (peak around
        # the "transition point" where eigvec rotates from HPWL-dominated
        # to cong-dominated). Cost: ~5 quick Lanczos calls × 5s = 25s
        # overhead, well under the Hessian budget.
        if (os.environ.get("PLACER_V7_HESSIAN_AUTO_LAMBDA_SCAN", "0") == "1"
                and cong_enabled):
            from _hessian_escape import hessian_min_eigvec
            scan_str = os.environ.get(
                "PLACER_V7_HESSIAN_LAMBDA_SCAN_WEIGHTS",
                "0.25,0.5,0.75,1.0,1.5,2.0")
            candidate_weights = [float(w) for w in scan_str.split(",")]
            best_w = cong_weight
            best_lam = 0.0
            t_scan = time.time()
            print(f"  [v7] hessian AUTO_LAMBDA_SCAN: testing weights "
                  f"{candidate_weights}", flush=True)
            for w_test in candidate_weights:
                cong_weight = w_test   # closure reads this on each call
                try:
                    lam, _ = hessian_min_eigvec(
                        smooth_proxy_call, macro_pos_t,
                        n_lanczos_iters=50, tikhonov=1e-4, verbose=False)
                except Exception as e:
                    print(f"  [v7]   w={w_test}: ERROR {e}", flush=True)
                    continue
                print(f"  [v7]   w={w_test}: λ_min={lam:+.6e}",
                      flush=True)
                if lam < best_lam:   # most negative wins
                    best_lam = lam
                    best_w = w_test
            cong_weight = best_w
            print(f"  [v7] hessian AUTO_LAMBDA_SCAN: optimal w={best_w} "
                  f"(λ_min={best_lam:+.6e}, scan {time.time()-t_scan:.1f}s)",
                  flush=True)

        # zeus B7: Free-energy / Gaussian-smoothed wrapper.
        # When enabled, replaces smooth_proxy_call with F̂(x) = mean over
        # K Gaussian-perturbed samples, penalizing sharp minima.
        if os.environ.get("PLACER_V7_HESSIAN_FREE_ENERGY", "0") == "1":
            from _free_energy import make_free_energy_proxy
            fe_sigma = float(os.environ.get(
                "PLACER_V7_HESSIAN_FE_SIGMA", "5.0"))
            fe_K = int(os.environ.get("PLACER_V7_HESSIAN_FE_K", "4"))
            fe_state_salt = int((macro_pos_t.detach().sum().item()
                                  * 1e6)) % 10_000_000
            fe_seed = (int(self.seed) + 88888 + fe_state_salt) & 0x7FFFFFFF
            print(f"  [v7] hessian: FREE-ENERGY wrapper "
                  f"σ={fe_sigma:.2f}μm K={fe_K} seed={fe_seed}", flush=True)
            smooth_proxy_call = make_free_energy_proxy(
                smooth_proxy_call, sigma=fe_sigma, K=fe_K, seed=fe_seed,
                soft_only=True, n_hard=n_hard)

        # Compute Hessian eigvec
        t_h = time.time()
        # istanbul: adaptive line search + feasibility filter (env-gated).
        if os.environ.get("PLACER_V7_HESSIAN_ADAPTIVE", "0") == "1":
            from _hessian_escape import (
                adaptive_topk_candidates, feasibility_filter)
            adaptive_k = int(os.environ.get(
                "PLACER_V7_HESSIAN_ADAPTIVE_TOPK", "2"))
            ls_initial = float(os.environ.get(
                "PLACER_V7_HESSIAN_LS_INITIAL", "0.10"))
            ls_steps = int(os.environ.get(
                "PLACER_V7_HESSIAN_LS_STEPS", "10"))
            ls_shrink = float(os.environ.get(
                "PLACER_V7_HESSIAN_LS_SHRINK", "0.6"))
            tikhonov = float(os.environ.get(
                "PLACER_V7_HESSIAN_TIKHONOV", "1e-4"))
            candidates, diag = adaptive_topk_candidates(
                macro_pos_t, smooth_proxy_call,
                k=adaptive_k,
                canvas_diag=canvas_diag,
                n_lanczos_iters=n_lanczos_iters,
                tikhonov=tikhonov,
                n_hard=n_hard,
                soft_only_perturb=True,
                ls_initial=ls_initial,
                ls_n_steps=ls_steps,
                ls_shrink=ls_shrink,
                verbose=True)
            # albania2: K-dim trust-region Newton step in the negative
            # eigenspace, as an ADDITIONAL candidate alongside the per-
            # eigvec line searches. Coordinated multi-direction descent
            # — the per-eigvec candidates only explore axis-aligned
            # cuts of the same K-dim subspace.
            if (os.environ.get("PLACER_V7_HESSIAN_KDIM_NEWTON", "0") == "1"
                    and diag.get("eigvecs") is not None
                    and diag.get("eigvals") is not None):
                from _hessian_escape import kdim_trust_region_step
                kdim_trust = float(os.environ.get(
                    "PLACER_V7_HESSIAN_KDIM_TRUST", "0.10"))
                kdim_backtrack = int(os.environ.get(
                    "PLACER_V7_HESSIAN_KDIM_BACKTRACK", "6"))
                kd_result = kdim_trust_region_step(
                    macro_pos_t, smooth_proxy_call,
                    diag["eigvals"], diag["eigvecs"],
                    canvas_diag=canvas_diag,
                    trust_radius=kdim_trust,
                    n_backtrack=kdim_backtrack,
                    n_hard=n_hard,
                    soft_only=True,
                    verbose=True)
                if kd_result is not None:
                    kd_label, kd_pos, _kd_f = kd_result
                    candidates.append((kd_label, kd_pos))
                    print(f"  [v7] hessian: kdim Newton candidate added "
                          f"({kd_label}); total candidates={len(candidates)}",
                          flush=True)
            # zeus: Subspace Hamiltonian Monte Carlo escape, added as an
            # additional candidate generator alongside the per-eigvec line
            # searches. See _subspace_hmc.py for the math. When enabled,
            # we run a SEPARATE Lanczos call to get a wider K-dim basis
            # (the existing adaptive call may use k=1 only), then run
            # HMC trajectories from random momentum in that subspace.
            # The strict-improvement gate downstream filters bad samples.
            hmc_K = int(os.environ.get("PLACER_V7_HESSIAN_HMC_K", "0"))
            hmc_T = int(os.environ.get("PLACER_V7_HESSIAN_HMC_TRAJ", "0"))
            if hmc_K > 0 and hmc_T > 0:
                from _hessian_escape import hessian_min_eigvecs_topk
                from _subspace_hmc import subspace_hmc_candidates
                hmc_L = int(os.environ.get("PLACER_V7_HESSIAN_HMC_L", "12"))
                hmc_h = float(os.environ.get("PLACER_V7_HESSIAN_HMC_STEP", "0.5"))
                hmc_cap = float(os.environ.get("PLACER_V7_HESSIAN_HMC_CAP", "0.20"))
                # zeus B1: integrator order. "leapfrog" (2nd) or "yoshida4" (4th).
                hmc_integ = os.environ.get(
                    "PLACER_V7_HESSIAN_HMC_INTEGRATOR", "leapfrog")
                # Salt seed with the integer hash of the current macro state
                # so each MAX_ITERS iter draws fresh momenta (and successive
                # benches/seeds are independent).
                state_salt = int((macro_pos_t.detach().sum().item()
                                   * 1e6)) % 10_000_000
                hmc_seed = (int(self.seed) + state_salt) & 0x7FFFFFFF
                t_hmc = time.time()
                # Use the eigeninfo from adaptive call if it has enough
                # eigvecs, otherwise spend a separate Lanczos to get hmc_K.
                ev = diag.get("eigvecs")
                eg = diag.get("eigvals")
                if ev is None or eg is None or ev.shape[1] < hmc_K:
                    eg, ev = hessian_min_eigvecs_topk(
                        smooth_proxy_call, macro_pos_t,
                        k=hmc_K, n_lanczos_iters=n_lanczos_iters,
                        tikhonov=tikhonov, verbose=False)
                hmc_cands, hmc_diag = subspace_hmc_candidates(
                    macro_pos_t, smooth_proxy_call, eg, ev,
                    n_trajectories=hmc_T, n_leapfrog=hmc_L,
                    step_size=hmc_h, canvas_diag=canvas_diag,
                    n_hard=n_hard, soft_only=True, seed=hmc_seed,
                    max_total_step_canvas=hmc_cap,
                    integrator=hmc_integ, verbose=True)
                # zeus B2: replica-overlap diverse selection. If enabled,
                # over-sample HMC trajectories and pick the K most diverse.
                hmc_replica_keep = int(os.environ.get(
                    "PLACER_V7_HESSIAN_HMC_REPLICA_KEEP", "0"))
                if hmc_replica_keep > 0 and len(hmc_cands) > hmc_replica_keep:
                    from _subspace_hmc import replica_diverse_select
                    base_pos_np = macro_pos_t.detach().cpu().numpy()
                    hmc_cands, rep_diag = replica_diverse_select(
                        hmc_cands, base_pos_np,
                        n_select=hmc_replica_keep,
                        candidate_diagnostics=hmc_diag.get("trajectories"))
                    print(f"  [v7] hessian: replica-diverse "
                          f"keep={len(hmc_cands)}/{rep_diag['n_cand']}, "
                          f"all_med={rep_diag['all_pairwise_median_microns']:.1f}μm, "
                          f"sub_min={rep_diag['subset_pairwise_min_microns']:.1f}μm",
                          flush=True)
                # Append HMC candidates with original-label form.
                for lab, pos_np in hmc_cands:
                    candidates.append((lab, pos_np))
                print(f"  [v7] hessian: HMC added {len(hmc_cands)} candidates "
                      f"(K={hmc_K}, T={hmc_T}, L={hmc_L}, h={hmc_h}, "
                      f"{time.time()-t_hmc:.1f}s); total={len(candidates)}",
                      flush=True)

            # zeus B11: NEB minimum-energy path between top-2 diverse
            # candidates. Finds the saddle/transition state separating
            # two basins; adds it (and adjacent images) as candidates.
            neb_enabled = int(os.environ.get(
                "PLACER_V7_HESSIAN_NEB", "0"))
            if neb_enabled > 0 and len(candidates) >= 2:
                from _neb import neb_candidates
                # Pick top-2 candidates by their candidate-list position.
                # (The candidates list is unsorted at this stage; we pick
                # the first 2 with maximally different positions.)
                neb_images = int(os.environ.get(
                    "PLACER_V7_HESSIAN_NEB_IMAGES", "7"))
                neb_iters = int(os.environ.get(
                    "PLACER_V7_HESSIAN_NEB_ITERS", "20"))
                neb_lr = float(os.environ.get(
                    "PLACER_V7_HESSIAN_NEB_LR", "0.3"))
                neb_spring = float(os.environ.get(
                    "PLACER_V7_HESSIAN_NEB_SPRING_K", "0.05"))
                # Compute pairwise distances of existing candidates,
                # pick the 2 with maximal distance.
                cand_arr = [c[1] for c in candidates]
                if len(cand_arr) >= 2:
                    flats = [c.reshape(-1) for c in cand_arr]
                    nn = len(flats)
                    max_d, max_pair = 0.0, (0, 1)
                    for ii in range(nn):
                        for jj in range(ii + 1, nn):
                            d = float(np.linalg.norm(flats[ii] - flats[jj]))
                            if d > max_d:
                                max_d = d
                                max_pair = (ii, jj)
                    seeds_for_neb = [cand_arr[max_pair[0]],
                                      cand_arr[max_pair[1]]]
                    t_neb = time.time()
                    neb_cands, neb_diag = neb_candidates(
                        seeds_for_neb, smooth_proxy_call,
                        n_images=neb_images, n_iters=neb_iters,
                        lr=neb_lr, spring_k=neb_spring,
                        n_hard=n_hard,
                        canvas_w=float(canvas_w), canvas_h=float(canvas_h),
                        barrier_eps_frac=0.001, verbose=False)
                    for lab, pos_np in neb_cands:
                        candidates.append((lab, pos_np))
                    n_pairs = len(neb_diag.get("pairs", []))
                    print(f"  [v7] hessian: NEB added "
                          f"{len(neb_cands)} candidates "
                          f"(pairs={n_pairs}, images={neb_images}, "
                          f"max_d={max_d:.1f}μm, "
                          f"{time.time()-t_neb:.1f}s); "
                          f"total={len(candidates)}", flush=True)

            # zeus B10: Catastrophe-theory fold candidates. For each
            # Lanczos eigvec v_k with λ_k < 0, estimate the cubic term
            # c_k via 4-point finite differences along v_k, then place
            # one candidate at the EXACT fold critical point
            # t_k* = -2λ_k / c_k. Closed-form alternative to line search.
            cata_K = int(os.environ.get(
                "PLACER_V7_HESSIAN_CATASTROPHE_K", "0"))
            if cata_K > 0:
                from _hessian_escape import hessian_min_eigvecs_topk
                from _catastrophe import catastrophe_fold_candidates
                cata_h_frac = float(os.environ.get(
                    "PLACER_V7_HESSIAN_CATASTROPHE_H_FRAC", "0.005"))
                cata_cap = float(os.environ.get(
                    "PLACER_V7_HESSIAN_CATASTROPHE_CAP", "0.15"))
                # Reuse eigeninfo from adaptive call if possible.
                ev_c = diag.get("eigvecs")
                eg_c = diag.get("eigvals")
                if ev_c is None or eg_c is None or ev_c.shape[1] < cata_K:
                    eg_c, ev_c = hessian_min_eigvecs_topk(
                        smooth_proxy_call, macro_pos_t,
                        k=cata_K, n_lanczos_iters=n_lanczos_iters,
                        tikhonov=tikhonov, verbose=False)
                t_cata = time.time()
                cata_cands, cata_diag = catastrophe_fold_candidates(
                    smooth_proxy_call, macro_pos_t, eg_c, ev_c,
                    canvas_diag=canvas_diag,
                    cap_frac=cata_cap, h_frac=cata_h_frac,
                    n_hard=n_hard, soft_only=True, verbose=True,
                )
                for lab, pos_np in cata_cands:
                    candidates.append((lab, pos_np))
                print(f"  [v7] hessian: catastrophe-fold added "
                      f"{len(cata_cands)} candidates "
                      f"(K={cata_K}, h_frac={cata_h_frac:.3f}, "
                      f"{time.time()-t_cata:.1f}s); "
                      f"total={len(candidates)}", flush=True)

            # zeus B8: SMC tempered importance sampler as a candidate
            # generator. Starts N=16 particles jittered from current state,
            # runs T tempering stages with adaptive β-schedule + Gaussian
            # RW Metropolis between stages. Returns top-K final particles.
            smc_N = int(os.environ.get(
                "PLACER_V7_HESSIAN_SMC_N", "0"))
            if smc_N > 0:
                from _smc import smc_sampler
                smc_T = int(os.environ.get(
                    "PLACER_V7_HESSIAN_SMC_STAGES", "10"))
                smc_jitter = float(os.environ.get(
                    "PLACER_V7_HESSIAN_SMC_JITTER", "3.0"))
                smc_mcmc_sigma = float(os.environ.get(
                    "PLACER_V7_HESSIAN_SMC_MCMC_SIGMA", "2.0"))
                smc_mcmc_per = int(os.environ.get(
                    "PLACER_V7_HESSIAN_SMC_MCMC_PER_STAGE", "1"))
                smc_keep = int(os.environ.get(
                    "PLACER_V7_HESSIAN_SMC_KEEP", "8"))
                base_pos_np_smc = macro_pos_t.detach().cpu().numpy()
                # Jitter init (soft macros only).
                rng_init = np.random.default_rng(int(self.seed) + 99999)
                smc_init = np.tile(base_pos_np_smc[None, :, :],
                                     (smc_N, 1, 1)).astype(np.float64)
                if smc_init.shape[1] - n_hard > 0:
                    smc_init[:, n_hard:, :] += (rng_init.standard_normal(
                        (smc_N, smc_init.shape[1] - n_hard, 2)) * smc_jitter)
                    smc_init[:, :, 0] = np.clip(
                        smc_init[:, :, 0], 0.0, float(canvas_w))
                    smc_init[:, :, 1] = np.clip(
                        smc_init[:, :, 1], 0.0, float(canvas_h))

                def smc_U_batch(xs_np):
                    out = np.empty(xs_np.shape[0], dtype=np.float64)
                    with torch.no_grad():
                        for i in range(xs_np.shape[0]):
                            x_t = torch.tensor(xs_np[i], dtype=torch.float32)
                            out[i] = float(smooth_proxy_call(x_t).item())
                    return out

                t_smc = time.time()
                smc_final, smc_diag = smc_sampler(
                    smc_init, smc_U_batch,
                    n_steps=smc_T,
                    target_ess_frac=0.5,
                    n_mcmc_per_step=smc_mcmc_per,
                    mcmc_step_sigma=smc_mcmc_sigma,
                    canvas_w=float(canvas_w), canvas_h=float(canvas_h),
                    n_hard=n_hard,
                    seed=(int(self.seed) + 55555) & 0x7FFFFFFF,
                    verbose=True,
                )
                # Sort particles by final U and keep top-K.
                smc_U_final = smc_U_batch(smc_final)
                order = np.argsort(smc_U_final)
                for i, idx in enumerate(order[:smc_keep]):
                    label = f"smc_p{i:02d}_U{smc_U_final[idx]:+.4f}"
                    candidates.append((label, smc_final[idx]))
                print(f"  [v7] hessian: SMC added "
                      f"{min(smc_keep, len(order))} candidates "
                      f"(N={smc_N}, stages={smc_T}, β_final={smc_diag['final_beta']:.2f}, "
                      f"{time.time()-t_smc:.1f}s); total={len(candidates)}",
                      flush=True)

            # zeus B5: Diffusion Monte Carlo as an additional candidate
            # generator. Walker-based imaginary-time evolution; doesn't
            # need a smooth gradient surface and naturally covers diverse
            # basins. Env: PLACER_V7_HESSIAN_DMC_WALKERS, _STEPS, _TAU, _BETA.
            dmc_walkers = int(os.environ.get(
                "PLACER_V7_HESSIAN_DMC_WALKERS", "0"))
            dmc_steps = int(os.environ.get(
                "PLACER_V7_HESSIAN_DMC_STEPS", "30"))
            if dmc_walkers > 0:
                from _dmc_walker import diffusion_monte_carlo_candidates
                dmc_tau = float(os.environ.get(
                    "PLACER_V7_HESSIAN_DMC_TAU", "0.5"))
                dmc_beta = float(os.environ.get(
                    "PLACER_V7_HESSIAN_DMC_BETA", "1.0"))
                dmc_init_jitter = float(os.environ.get(
                    "PLACER_V7_HESSIAN_DMC_INIT_JITTER", "5.0"))
                dmc_cap = float(os.environ.get(
                    "PLACER_V7_HESSIAN_DMC_STEP_CAP", "50.0"))
                dmc_keep = int(os.environ.get(
                    "PLACER_V7_HESSIAN_DMC_KEEP", "8"))
                dmc_state_salt = int((macro_pos_t.detach().sum().item()
                                       * 1e6)) % 10_000_000
                dmc_seed = (int(self.seed) + 77777 + dmc_state_salt) & 0x7FFFFFFF
                t_dmc = time.time()
                dmc_cands, dmc_diag = diffusion_monte_carlo_candidates(
                    macro_pos_t, smooth_proxy_call,
                    n_walkers=dmc_walkers, n_steps=dmc_steps,
                    tau=dmc_tau, beta=dmc_beta,
                    init_jitter=dmc_init_jitter,
                    n_hard=n_hard,
                    canvas_w=float(canvas_w),
                    canvas_h=float(canvas_h),
                    step_cap_microns=dmc_cap,
                    seed=dmc_seed, verbose=True)
                # Keep only top-K best walkers as candidates.
                for lab, pos_np in dmc_cands[:dmc_keep]:
                    candidates.append((lab, pos_np))
                print(f"  [v7] hessian: DMC added "
                      f"{min(len(dmc_cands), dmc_keep)} candidates "
                      f"(N0={dmc_walkers}, steps={dmc_steps}, τ={dmc_tau:.2f}, "
                      f"survivors={dmc_diag.get('n_walkers_final', 0)}, "
                      f"{time.time()-t_dmc:.1f}s); total={len(candidates)}",
                      flush=True)

            # Convert to (s, pos) tuple shape that the worker code below
            # expects (existing code uses `s` as a sortkey/seed-shifter).
            # We re-encode the ±step from the label into a numeric sortkey.
            # kdim labels (e.g. "kdim_K3_r0.050") get sortkey 0.0; HMC
            # labels (e.g. "hmc_t05_K6_dU-1.23") get a unique fractional
            # sortkey so each worker gets a distinct seed.
            def _label_to_sortkey(lab):
                if not isinstance(lab, str):
                    return 0.0
                if "_ls" in lab:
                    try: return float(lab.split("_ls")[1])
                    except (ValueError, IndexError): return 0.0
                if lab.startswith("hmc_t"):
                    # Extract trajectory index for unique seed offset.
                    try:
                        tnum = int(lab[5:].split("_")[0])
                        return 0.001 * (tnum + 1)
                    except (ValueError, IndexError): return 0.0
                if lab.startswith("dmc_w"):
                    try:
                        wnum = int(lab[5:].split("_")[0])
                        return 0.0001 * (wnum + 1)
                    except (ValueError, IndexError): return 0.0
                if lab.startswith("smc_p"):
                    try:
                        pnum = int(lab[5:].split("_")[0])
                        return 0.00001 * (pnum + 1)
                    except (ValueError, IndexError): return 0.0
                if lab.startswith("cata_"):
                    try:
                        cnum = int(lab[6:].split("_")[0])
                        return 0.000001 * (cnum + 1)
                    except (ValueError, IndexError): return 0.0
                if lab.startswith("neb_"):
                    try:
                        # "neb_p01_img3_..."
                        pidx = lab.split("_img")[1].split("_")[0]
                        return 0.0000001 * (int(pidx) + 1)
                    except (ValueError, IndexError): return 0.0
                return 0.0
            candidates = [(_label_to_sortkey(lab), pos)
                          for lab, pos in candidates]
            # Feasibility filter
            max_overlaps = int(os.environ.get(
                "PLACER_V7_HESSIAN_MAX_OVERLAPS", "200"))
            kept, dropped = feasibility_filter(
                candidates, bench, max_overlaps=max_overlaps)
            if dropped:
                drop_summary = ", ".join(f"e{i}={n}" for i, (_, n) in enumerate(dropped))
                print(f"  [v7] hessian feasibility filter dropped "
                      f"{len(dropped)}/{len(candidates)} "
                      f"(threshold={max_overlaps}): {drop_summary}",
                      flush=True)
            candidates = kept
        else:
            candidates, diag = hessian_escape_step(
                macro_pos_t, smooth_proxy_call,
                step_sizes=step_sizes,
                canvas_diag=canvas_diag,
                n_lanczos_iters=n_lanczos_iters,
                n_hard=n_hard,
                soft_only_perturb=True,
                verbose=True)
        # albania2: stash eigvecs/eigvals so the iter loop can derive
        # spectral net criticality for the next iteration's surrogate.
        # Done here (before feasibility filter / SA workers) so the
        # signal is captured even if all candidates fail validation.
        self._last_hessian_eigeninfo = {
            "eigvals": diag.get("eigvals"),
            "eigvecs": diag.get("eigvecs"),
            "pin_macro": np.asarray(incr.pin_macro),
            "net_starts": np.asarray(incr.net_starts),
            "n_total": int(np.asarray(incr.macro_pos).shape[0]),
            "base_net_weight": np.asarray(incr.net_weight).copy(),
        }
        if not candidates:
            print(f"  [v7] hessian: no candidates (eigvec degenerate); "
                  f"keeping post-Lap", flush=True)
            return current_pos, current_cost
        print(f"  [v7] hessian: λ_min={diag['lambda_min']:.6f}, "
              f"computed eigvec in {time.time()-t_h:.1f}s, "
              f"running {len(candidates)} candidates "
              f"× {hop_budget}s in parallel...", flush=True)

        # Spawn workers, each runs reduced pipeline from a perturbed init
        args = []
        for s, pos_np in candidates:
            pos64 = np.ascontiguousarray(pos_np, dtype=np.float64)
            seed = self.seed + int(abs(s) * 10000) + (1 if s > 0 else 0)
            args.append((bench_path, pos64.tobytes(), pos64.shape,
                          str(pos64.dtype), int(hop_budget), seed,
                          f"step={s:+.3f}"))
        ctx = _mp.get_context("spawn")
        with ctx.Pool(min(len(args), 8)) as pool:
            results = pool.map(hessian_candidate_worker, args)

        best_pos = None
        best_cost = float(current_cost)
        best_label = None
        for label, pos_bytes, shape, dtype_str, cost, ov in results:
            tag = "VALID" if ov == 0 else f"INVALID({ov})"
            print(f"    [v7.hess.{label}] cost={cost:.6f} {tag}",
                  flush=True)
            if ov != 0:
                continue
            if cost < best_cost - 1e-7:
                best_cost = cost
                best_pos = np.frombuffer(
                    pos_bytes, dtype=np.dtype(dtype_str)
                ).reshape(shape).copy()
                best_label = label

        if best_pos is None:
            print(f"  [v7] hessian: no candidate beat post-Lap "
                  f"({current_cost:.6f}); keeping", flush=True)
            return current_pos, current_cost

        print(f"  [v7] HESSIAN WIN: {best_label} cost={best_cost:.6f} < "
              f"{current_cost:.6f} (Δ {current_cost-best_cost:+.4f})",
              flush=True)
        return torch.tensor(best_pos, dtype=torch.float32), best_cost

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
        # Import the worker from a standalone module so multiprocessing's
        # pickle can resolve it. (When this placer.py is loaded via
        # importlib.util.spec_from_file_location, in-file functions can't
        # be pickled.)
        from _sp_worker import sp_hop_worker

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
            results = pool.map(sp_hop_worker, args)

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
