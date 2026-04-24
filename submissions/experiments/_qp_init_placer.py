"""QP-init placer: subclass of ParameterizedPlacer that prepends QP seed to tournament."""
import sys, time, random, importlib.util
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _param_placer import ParameterizedPlacer
import _param_placer as _pp  # for access to module-level helpers

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from _qp_init import qp_init


class QPInitPlacer(ParameterizedPlacer):
    """Adds quadratic-placement seed to tournament. All other knobs inherited."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        _pp.reset_surrogate_state()

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.macro_positions.shape[0]
        plc = _pp._load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        t0 = time.time()
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed_positions = [_pp._push_apart(init_pos, benchmark, max_iters=mi, damping=d)
                            for mi, d in push_configs]
        print(f"  [push_apart] {time.time() - t0:.1f}s", flush=True)

        # === NEW: compute QP seed and prepend to starts ===
        try:
            t_qp = time.time()
            qp_plc = _pp._load_plc(benchmark.name)
            qp_seed = qp_init(init_pos, benchmark, qp_plc)
            print(f"  [qp_init] {time.time() - t_qp:.2f}s", flush=True)
        except Exception as e:
            print(f"  [qp_init] FAILED: {e}", flush=True)
            qp_seed = None

        best_legal = None
        best_cost = float('inf')
        starts = []
        if qp_seed is not None:
            starts.append((qp_seed, "qp"))
        starts += [(p, f"push_{k}") for k, p in enumerate(pushed_positions)] + [(init_pos, "raw")]
        order_types = ['size_desc', 'conn_desc', 'random']
        step_mults = [0.5, 1.0, 2.0, 4.0]
        for sp, tag in starts:
            for ot in order_types:
                for sm in step_mults:
                    if time.time() - t0 > self.LEGALIZE_BUDGET: break
                    legal = _pp._legalize(sp, benchmark, order_type=ot, step_mult=sm)
                    refined = _pp._refine_toward_initial(legal, init_pos, benchmark)
                    if refined is None: continue
                    plc_eval = _pp._load_plc(benchmark.name)
                    placement = torch.from_numpy(refined).float()
                    placement = torch.cat([placement, benchmark.macro_positions[n_hard:]])
                    cost = compute_proxy_cost(placement, benchmark, plc_eval)["proxy_cost"]
                    if cost < best_cost:
                        best_cost = cost
                        best_legal = refined.copy()
                        best_tag = tag
                if time.time() - t0 > self.LEGALIZE_BUDGET: break
            if time.time() - t0 > self.LEGALIZE_BUDGET: break
        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f} winner={best_tag if best_legal is not None else 'NONE'}", flush=True)
        if best_legal is None:
            best_legal = pushed_positions[0]

        plc_cd = _pp._load_plc(benchmark.name)
        incr_eval = _pp.IncrementalEvaluator(plc_cd, benchmark)
        incr_eval.sync_positions(best_legal)

        cd1_budget = max(40, min(400, self.TOTAL_TIME_LIMIT - (time.time() - t0) - self.SOFT_PHASE_BUDGET - 30))
        best_pos, best_cost = _pp._coord_descent(best_legal, benchmark, plc_cd, max_time=cd1_budget, incr_eval=incr_eval)
        print(f"  [cd1] cost={best_cost:.6f}", flush=True)

        try:
            _, c = _pp.per_net_optimize(best_pos, benchmark, incr_eval, max_time=15.0)
            if c < best_cost: best_cost = c
            print(f"  [per_net] cost={best_cost:.6f}", flush=True)
        except Exception: pass

        try:
            _, c = _pp.lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
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
                    _, c = _pp.fd_soft_place(best_pos, benchmark, incr_eval,
                                              max_time=cycle_t * r_fd, damping=self.FD_DAMPING, check_cost=True)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_sur > 0:
                    _, c = _pp.soft_cd_surrogate_v2(best_pos, benchmark, incr_eval, max_time=cycle_t * r_sur)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_cd > 0:
                    _, c = _pp.soft_macro_cd(best_pos, benchmark, incr_eval, max_time=cycle_t * r_cd)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            try:
                if r_lns > 0:
                    _, c = _pp.soft_lns_phase(best_pos, benchmark, incr_eval,
                                               max_time=cycle_t * r_lns,
                                               n_destroy=self.LNS_N_DESTROY, n_candidates=self.LNS_N_CANDIDATES)
                    if c < best_cost: best_cost = c
            except Exception: pass
            if time.time() - t0 > soft_deadline: break

            if r_hard > 0:
                _, c = _pp._coord_descent(best_pos, benchmark, plc_cd,
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
