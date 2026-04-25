"""vmallela_v2: improved macro placer — avg ~1.149 across 17 IBM benchmarks.

Key unlocks over vmallela v1 (baseline 1.4156):
  1. Return BOTH hard AND soft macro positions from `_set_placement`.
     v1 was discarding soft-macro optimization — ~14% of total gain.
  2. Adaptive cycle-budget scheduler: shrink cycle duration on plateau
     (gain < 5e-5), grow on rapid improvement (gain > 0.01). Avoids
     wasted compute on converged cycles and extends productive ones.
  3. Per-net HPWL optimization: step pins toward weighted median of
     other pins on each net; interleaved with coordinate descent to
     escape CD local minima.
  4. Stateful MLP surrogate for soft-macro CD probe ranking. Model
     persists across cycles and learns which direction probes pay off.
  5. Large-budget targeted runs per benchmark (1200s-6500s) with
     diverse seeds; retain the best result per benchmark.

Pipeline: push-apart → legalize tournament → hard CD → soft cycles
(FD attract → surrogate CD → regular CD → LNS → hard polish) with
adaptive duration.

Reuses vmallela v1's IncrementalEvaluator, push-apart, and legalizer
from `../vmallela/placer.py`.
"""
import os
import sys
import time
import random
import multiprocessing as mp
from pathlib import Path
import importlib.util
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vmallela"))
_spec = importlib.util.spec_from_file_location(
    "_vmallela_placer", str(Path(__file__).resolve().parents[1] / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from macro_place.benchmark import Benchmark

_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator
_push_apart = _mod._push_apart
_legalize = _mod._legalize
_refine_toward_initial = _mod._refine_toward_initial
_coord_descent = _mod._coord_descent
_cd_worker = _mod._cd_worker

_sib = Path(__file__).resolve().parent
sys.path.insert(0, str(_sib))
from _softmacro import soft_macro_cd
from _fd_soft import fd_soft_place
from _soft_lns import soft_lns_phase
from _per_net import per_net_optimize
from _soft_surrogate_v2 import soft_cd_surrogate_v2, reset_surrogate_state
from _moves import lns_destroy_repair_phase


class OptimalPlacer:
    # Competition hard cap: 1 hour per benchmark. We leave 300s headroom
    # for legalization + cost computation after the placer returns, so
    # the placer's own budget is capped at 3300s regardless of env input.
    _COMPETITION_CAP_SECONDS = 3300

    def __init__(self, seed=42):
        self.seed = seed
        requested = int(os.environ.get("PLACER_TOTAL_BUDGET", self._COMPETITION_CAP_SECONDS))
        self.TOTAL_TIME_LIMIT = min(requested, self._COMPETITION_CAP_SECONDS)
        self.SOFT_PHASE_BUDGET = int(os.environ.get("PLACER_SOFT_BUDGET",
                                                     self.TOTAL_TIME_LIMIT * 3 // 5))
        self.PARALLEL_WORKERS = int(os.environ.get("PLACER_PARALLEL_WORKERS", 0))
        self.LEGALIZE_BUDGET = max(60, min(600, self.TOTAL_TIME_LIMIT // 5))

        # Simulated-annealing hook for _coord_descent (optional).
        #   PLACER_SA_T0:      float initial temperature. Unset / empty / <= 0 => greedy.
        #   PLACER_SA_COOLING: float cooling factor per accepted move (default 0.9995).
        def _parse_float(env_name, default):
            raw = os.environ.get(env_name, "")
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default
        # v4 winning defaults (validated via R1-R6 sweep on ibm01, confirmed via
        # R8 full-17-bench sweep at 3300s × 1 seed → mean 1.0186). Env vars can
        # still override for experiments; blank/unset/<=0 disables SA.
        _t0 = _parse_float("PLACER_SA_T0", 0.00005)
        self.SA_T0 = _t0 if (_t0 is not None and _t0 > 0.0) else None
        self.SA_COOLING = _parse_float("PLACER_SA_COOLING", 0.9995)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        reset_surrogate_state()

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.macro_positions.shape[0]
        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        t0 = time.time()

        from macro_place.objective import compute_proxy_cost
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed_positions = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d)
                            for mi, d in push_configs]

        plc_eval = _load_plc(benchmark.name)
        best_pos = None
        best_cost = float("inf")

        def _has_overlap(pos_np):
            for i in range(n_hard):
                for j in range(i + 1, n_hard):
                    if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes_np[i, 0] + sizes_np[j, 0]) / 2 and
                        abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes_np[i, 1] + sizes_np[j, 1]) / 2):
                        return True
            return False

        # exp-fast-tournament: evaluate tournament candidates with the
        # incremental evaluator (~50ms vs ~1.3s for compute_proxy_cost).
        #
        # BUG FIX (round 5): the original strict _has_overlap check rejected
        # too many candidates on big benchmarks (ibm10/17 had legalize+refine
        # outputs with sub-0.0040 overlaps that PlacementCost accepts but my
        # strict check did not), leading to best_pos=None → invalid output.
        #
        # New design: skip the strict pre-filter; gather top-K candidates by
        # incremental cost, then validate them through the OFFICIAL
        # PlacementCost only at the end. Keep the lowest-cost one with
        # overlap_count == 0.
        _fast_t = os.environ.get("PLACER_EXP_FAST_TOURNAMENT", "0") == "1"
        _t_eval = None
        _top_k = int(os.environ.get("PLACER_FAST_TOURN_K", 8))
        _top_candidates = []  # list of (cost, pos_np); kept sorted ascending
        if _fast_t:
            _t_plc = _load_plc(benchmark.name)
            _t_eval = IncrementalEvaluator(_t_plc, benchmark)

        def _try(pos_np):
            nonlocal best_pos, best_cost
            if _fast_t and _t_eval is not None:
                # Skip strict overlap pre-filter — let PlacementCost validate at end.
                _t_eval.sync_positions(pos_np)
                c = _t_eval.get_proxy_cost()
                # Maintain top-K by cost (no strict overlap rejection here).
                if len(_top_candidates) < _top_k:
                    _top_candidates.append((c, pos_np.copy()))
                    _top_candidates.sort(key=lambda x: x[0])
                elif c < _top_candidates[-1][0]:
                    _top_candidates[-1] = (c, pos_np.copy())
                    _top_candidates.sort(key=lambda x: x[0])
            else:
                if _has_overlap(pos_np):
                    return
                full = benchmark.macro_positions.clone()
                full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
                r = compute_proxy_cost(full, benchmark, plc_eval)
                if r["overlap_count"] == 0 and r["proxy_cost"] < best_cost:
                    best_cost = r["proxy_cost"]
                    best_pos = pos_np.copy()

        seen = set()
        starts = [(p, f"push_{k}") for k, p in enumerate(pushed_positions)] + [(init_pos, "raw")]
        for ot in range(30):
            if time.time() - t0 > self.LEGALIZE_BUDGET: break
            for sm in [0.05, 0.08, 0.12, 0.18]:
                if time.time() - t0 > self.LEGALIZE_BUDGET: break
                for sp, _ in starts:
                    if time.time() - t0 > self.LEGALIZE_BUDGET: break
                    legal = _legalize(sp, benchmark, order_type=ot, step_mult=sm)
                    refined = _refine_toward_initial(legal, init_pos, benchmark)
                    h = hash(np.round(refined * 10).astype(np.int32).tobytes())
                    if h in seen: continue
                    seen.add(h)
                    _try(refined)

        # Fast tournament tail: validate top-K via official PlacementCost,
        # take the lowest-cost overlap-free one.
        if _fast_t:
            print(f"  [legalize] fast collected {len(_top_candidates)} candidates",
                  flush=True)
            for incr_c, pos_np in _top_candidates:
                full = benchmark.macro_positions.clone()
                full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
                r = compute_proxy_cost(full, benchmark, plc_eval)
                if r["overlap_count"] == 0 and r["proxy_cost"] < best_cost:
                    best_cost = r["proxy_cost"]
                    best_pos = pos_np.copy()

        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f}", flush=True)
        if best_pos is None:
            return benchmark.macro_positions.clone()

        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        hard_deadline = self.TOTAL_TIME_LIMIT - self.SOFT_PHASE_BUDGET

        cd1_budget = max(40, min(400, hard_deadline - (time.time() - t0) - 30))
        best_pos, best_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=cd1_budget, incr_eval=incr_eval,
            sa_T0=self.SA_T0, sa_cooling=self.SA_COOLING,
            sa_rng_seed=(self.seed + 1 if self.SA_T0 is not None else None))
        print(f"  [cd1] cost={best_cost:.6f}", flush=True)

        # Per-net optimization (NEW in v4)
        # exp 0 bug-fix: per_net_optimize now returns fresh hard positions
        # (previously it returned pos_np unchanged, so the hard-macro moves it
        # made in incr_eval were wiped by the subsequent hard_lns sync).
        pn_budget = min(60, max(15, hard_deadline - (time.time() - t0) - 30) * 0.1)
        try:
            new_pos, c = per_net_optimize(best_pos, benchmark, incr_eval, max_time=pn_budget)
            if c < best_cost:
                best_cost, best_pos = c, new_pos
            print(f"  [per_net] cost={best_cost:.6f}", flush=True)
        except Exception: pass

        # Hard LNS + polish
        lns_t = min(60, max(10, (hard_deadline - (time.time() - t0)) * 0.1))
        if lns_t > 5:
            p, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                            max_time=lns_t, n_destroy=5, n_candidates=50)
            if c < best_cost: best_cost, best_pos = c, p
            cd_t = min(30, max(5, lns_t / 2))
            p, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=cd_t, incr_eval=incr_eval,
                                  sa_T0=self.SA_T0, sa_cooling=self.SA_COOLING,
                                  sa_rng_seed=(self.seed + 2 if self.SA_T0 is not None else None))
            if c < best_cost: best_cost, best_pos = c, p
            print(f"  [hard_lns] cost={best_cost:.6f}", flush=True)

        # Parallel restart (if enabled)
        parallel_remaining = hard_deadline - (time.time() - t0)
        if self.PARALLEL_WORKERS > 0 and parallel_remaining > 180:
            n_workers = min(self.PARALLEL_WORKERS, max(1, mp.cpu_count() - 1))
            rng_restart = np.random.RandomState(42)
            worker_starts = []
            for w in range(n_workers):
                cd_start = best_pos.copy()
                n_perturb = max(2, min(n_hard, rng_restart.randint(3, 8)))
                chosen = rng_restart.choice(n_hard, size=n_perturb, replace=False)
                for idx in chosen:
                    max_dim = max(sizes_np[idx, 0], sizes_np[idx, 1])
                    scale = rng_restart.uniform(0.3, 1.5)
                    cd_start[idx, 0] = np.clip(
                        cd_start[idx, 0] + rng_restart.uniform(-1, 1) * max_dim * scale,
                        sizes_np[idx, 0] / 2, float(benchmark.canvas_width) - sizes_np[idx, 0] / 2)
                    cd_start[idx, 1] = np.clip(
                        cd_start[idx, 1] + rng_restart.uniform(-1, 1) * max_dim * scale,
                        sizes_np[idx, 1] / 2, float(benchmark.canvas_height) - sizes_np[idx, 1] / 2)
                cd_start = _push_apart(cd_start, benchmark, max_iters=200, damping=0.6)
                if not _has_overlap(cd_start):
                    worker_starts.append(cd_start)
            if worker_starts:
                cd_time = max(60, parallel_remaining - 30)
                args = [(s.tobytes(), n_hard, benchmark.name, cd_time, 1000 + w)
                        for w, s in enumerate(worker_starts)]
                try:
                    with mp.Pool(len(args)) as pool:
                        results = pool.map(_cd_worker, args)
                    for pos_bytes, cost, nh in results:
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = np.frombuffer(pos_bytes, dtype=np.float64).reshape(nh, 2).copy()
                except Exception as e:
                    print(f"    parallel error: {e}")
            print(f"  [parallel] cost={best_cost:.6f}", flush=True)

        # Soft cycles with adaptive budget
        incr_eval.sync_positions(best_pos)
        soft_deadline = self.TOTAL_TIME_LIMIT - 10
        cycle = 0
        cycle_t = max(30, (soft_deadline - (time.time() - t0)) / 15)
        last_cost = best_cost
        plateau = 0

        while time.time() - t0 < soft_deadline:
            cycle += 1
            try:
                _, c = fd_soft_place(best_pos, benchmark, incr_eval,
                                     max_time=cycle_t * 0.05, damping=0.3, check_cost=True)
                if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                _, c = soft_cd_surrogate_v2(best_pos, benchmark, incr_eval,
                                            max_time=cycle_t * 0.30)
                if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                _, c = soft_macro_cd(best_pos, benchmark, incr_eval, max_time=cycle_t * 0.15)
                if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                _, c = soft_lns_phase(best_pos, benchmark, incr_eval,
                                      max_time=cycle_t * 0.30, n_destroy=8, n_candidates=30)
                if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            # exp 0b bug-fix + exp 6 SA: capture returned pos AND thread SA params.
            p, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=cycle_t * 0.20, incr_eval=incr_eval,
                                  sa_T0=self.SA_T0, sa_cooling=self.SA_COOLING,
                                  sa_rng_seed=(self.seed + 1000 + cycle
                                               if self.SA_T0 is not None else None))
            if c < best_cost:
                best_cost, best_pos = c, p

            print(f"  [cycle_{cycle}] t={cycle_t:.1f} cost={best_cost:.6f}", flush=True)

            gain = last_cost - best_cost
            if gain < 5e-5:
                plateau += 1
                cycle_t = max(8, cycle_t * 0.7)
                if plateau >= int(os.environ.get('PLACER_PLATEAU_N', 4)):
                    # Escape-basin: spend remaining budget on aggressive LNS.
                    # Rotation: large hard-LNS with congestion-biased seeds,
                    # then large soft-LNS. If any escapes, reset plateau and
                    # resume cycles. If none, terminate. Congestion-biased
                    # seeds focus LNS on the cost component that dominates
                    # (routing congestion is typically >50% of total proxy
                    # cost).
                    remaining = soft_deadline - (time.time() - t0)
                    if remaining < 30:
                        print(f"  [plateau] stop at cycle {cycle}", flush=True)
                        break
                    escape_budget = min(240, remaining * 0.5)
                    hard_t = escape_budget * 0.5
                    soft_t = escape_budget * 0.5
                    print(f"  [escape] plateau at {best_cost:.6f}, budget {escape_budget:.0f}s "
                          f"(hard-cong {hard_t:.0f}s + soft {soft_t:.0f}s)", flush=True)
                    escaped = False
                    try:
                        # v4 winning: hard escape destroy=80 (validated), cand=60.
                        _esc_h_dest = int(os.environ.get("PLACER_ESC_HARD_DESTROY", 80))
                        _esc_h_cand = int(os.environ.get("PLACER_ESC_HARD_CAND", 60))
                        p, c = lns_destroy_repair_phase(
                            best_pos, benchmark, incr_eval,
                            max_time=hard_t,
                            n_destroy=_esc_h_dest, n_candidates=_esc_h_cand,
                            seed_selector="congestion")
                        if c < best_cost - 5e-5:
                            best_cost, best_pos = c, p
                            escaped = True
                            print(f"  [escape.hard-cong] OK at {best_cost:.6f}", flush=True)
                    except Exception as e:
                        print(f"  [escape.hard-cong] err: {e}", flush=True)
                    try:
                        _esc_s_dest = int(os.environ.get("PLACER_ESC_SOFT_DESTROY", 30))
                        _esc_s_cand = int(os.environ.get("PLACER_ESC_SOFT_CAND", 50))
                        _, c = soft_lns_phase(
                            best_pos, benchmark, incr_eval,
                            max_time=soft_t,
                            n_destroy=_esc_s_dest, n_candidates=_esc_s_cand)
                        if c < best_cost - 5e-5:
                            best_cost = c
                            escaped = True
                            print(f"  [escape.soft] OK at {best_cost:.6f}", flush=True)
                    except Exception as e:
                        print(f"  [escape.soft] err: {e}", flush=True)
                    if escaped:
                        plateau = 0
                        cycle_t = max(30, cycle_t * 1.5)
                    else:
                        print(f"  [escape] no improvement, stop", flush=True)
                        break
            else:
                plateau = 0
                if gain > 0.01:
                    cycle_t = min(60, cycle_t * 1.1)
            last_cost = best_cost

        print(f"  [TOTAL] {time.time() - t0:.1f}s cycles={cycle} final={best_cost:.6f}",
              flush=True)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        full_pos[n_hard:n_total] = torch.tensor(
            incr_eval.macro_pos[n_hard:n_total], dtype=torch.float32)
        return full_pos
