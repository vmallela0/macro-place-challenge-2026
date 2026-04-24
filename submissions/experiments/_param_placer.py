"""Parameterized placer — subclass of sub_v4 with hyperparameters exposed as class attrs.

Variants override attrs (e.g., RATIOS, LNS_N_DESTROY, SHRINK_FACTOR) without copying place().
"""
import os, sys, time, random, importlib.util
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
_load_plc = _m._load_plc
IncrementalEvaluator = _m.IncrementalEvaluator
_push_apart = _m._push_apart
_legalize = _m._legalize
_refine_toward_initial = _m._refine_toward_initial
_coord_descent = _m._coord_descent
soft_macro_cd = _m.soft_macro_cd
fd_soft_place = _m.fd_soft_place
soft_lns_phase = _m.soft_lns_phase
per_net_optimize = _m.per_net_optimize
soft_cd_surrogate_v2 = _m.soft_cd_surrogate_v2
reset_surrogate_state = _m.reset_surrogate_state
lns_destroy_repair_phase = _m.lns_destroy_repair_phase

_Base = _m.OptimalPlacer


class ParameterizedPlacer(_Base):
    """All hyperparameters as overridable class attributes."""
    # Soft-cycle operator fractions (fd, surrogate, regular_cd, lns, hard_polish)
    RATIOS = (0.05, 0.30, 0.15, 0.30, 0.20)
    # LNS params
    LNS_N_DESTROY = 8
    LNS_N_CANDIDATES = 30
    # Adaptive scheduler
    SHRINK_FACTOR = 0.7
    GROW_FACTOR = 1.1
    PLATEAU_THRESHOLD = 5e-5
    GAIN_THRESHOLD = 0.01
    PLATEAU_COUNT = 4
    MIN_CYCLE_T = 8
    MAX_CYCLE_T = 60
    INITIAL_CYCLE_DIVISOR = 15
    # FD damping
    FD_DAMPING = 0.3

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
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed_positions = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d)
                            for mi, d in push_configs]
        print(f"  [push_apart] {time.time() - t0:.1f}s", flush=True)

        best_legal = None
        best_cost = float('inf')
        starts = [(p, f"push_{k}") for k, p in enumerate(pushed_positions)] + [(init_pos, "raw")]
        order_types = ['size_desc', 'conn_desc', 'random']
        step_mults = [0.5, 1.0, 2.0, 4.0]
        for sp, tag in starts:
            for ot in order_types:
                for sm in step_mults:
                    if time.time() - t0 > self.LEGALIZE_BUDGET: break
                    legal = _legalize(sp, benchmark, order_type=ot, step_mult=sm)
                    refined = _refine_toward_initial(legal, init_pos, benchmark)
                    if refined is None: continue
                    plc_eval = _load_plc(benchmark.name)
                    placement = torch.from_numpy(refined).float()
                    placement = torch.cat([placement, benchmark.macro_positions[n_hard:]])
                    cost = compute_proxy_cost(placement, benchmark, plc_eval)["proxy_cost"]
                    if cost < best_cost:
                        best_cost = cost
                        best_legal = refined.copy()
                if time.time() - t0 > self.LEGALIZE_BUDGET: break
            if time.time() - t0 > self.LEGALIZE_BUDGET: break
        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f}", flush=True)
        if best_legal is None:
            best_legal = pushed_positions[0]

        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)
        incr_eval.sync_positions(best_legal)

        cd1_budget = max(40, min(400, self.TOTAL_TIME_LIMIT - (time.time() - t0) - self.SOFT_PHASE_BUDGET - 30))
        best_pos, best_cost = _coord_descent(best_legal, benchmark, plc_cd, max_time=cd1_budget, incr_eval=incr_eval)
        print(f"  [cd1] cost={best_cost:.6f}", flush=True)

        try:
            _, c = per_net_optimize(best_pos, benchmark, incr_eval, max_time=15.0)
            if c < best_cost: best_cost = c
            print(f"  [per_net] cost={best_cost:.6f}", flush=True)
        except Exception: pass

        try:
            _, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                             max_time=min(30.0, max(5, (self.TOTAL_TIME_LIMIT - (time.time() - t0) - self.SOFT_PHASE_BUDGET) * 0.1)))
            if c < best_cost: best_cost = c
            print(f"  [hard_lns] cost={best_cost:.6f}", flush=True)
        except Exception: pass

        incr_eval.sync_positions(best_pos)
        soft_deadline = self.TOTAL_TIME_LIMIT - 10
        cycle = 0
        cycle_t = max(30, (soft_deadline - (time.time() - t0)) / self.INITIAL_CYCLE_DIVISOR)
        last_cost = best_cost
        plateau = 0
        r_fd, r_sur, r_cd, r_lns, r_hard = self.RATIOS

        while time.time() - t0 < soft_deadline:
            cycle += 1
            try:
                if r_fd > 0:
                    _, c = fd_soft_place(best_pos, benchmark, incr_eval,
                                          max_time=cycle_t * r_fd, damping=self.FD_DAMPING, check_cost=True)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_sur > 0:
                    _, c = soft_cd_surrogate_v2(best_pos, benchmark, incr_eval, max_time=cycle_t * r_sur)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_cd > 0:
                    _, c = soft_macro_cd(best_pos, benchmark, incr_eval, max_time=cycle_t * r_cd)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_lns > 0:
                    _, c = soft_lns_phase(best_pos, benchmark, incr_eval,
                                           max_time=cycle_t * r_lns,
                                           n_destroy=self.LNS_N_DESTROY, n_candidates=self.LNS_N_CANDIDATES)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            if r_hard > 0:
                _, c = _coord_descent(best_pos, benchmark, plc_cd,
                                      max_time=cycle_t * r_hard, incr_eval=incr_eval)
                if c < best_cost: best_cost = c

            print(f"  [cycle_{cycle}] t={cycle_t:.1f} cost={best_cost:.6f}", flush=True)

            gain = last_cost - best_cost
            if gain < self.PLATEAU_THRESHOLD:
                plateau += 1
                cycle_t = max(self.MIN_CYCLE_T, cycle_t * self.SHRINK_FACTOR)
                if plateau >= self.PLATEAU_COUNT:
                    print(f"  [plateau] stop at cycle {cycle}", flush=True)
                    break
            else:
                plateau = 0
                if gain > self.GAIN_THRESHOLD:
                    cycle_t = min(self.MAX_CYCLE_T, cycle_t * self.GROW_FACTOR)
            last_cost = best_cost

        print(f"  [TOTAL] {time.time() - t0:.1f}s cycles={cycle} final={best_cost:.6f}", flush=True)

        out = incr_eval.macro_pos.copy()
        return torch.from_numpy(out.astype(np.float32))
