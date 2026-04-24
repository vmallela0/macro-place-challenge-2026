"""QP init + UCB bandit scheduler."""
import sys, time, random, math, importlib.util
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bandit_placer import BanditPlacer
import _param_placer as _pp
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from _qp_init import qp_init


class QPBanditPlacer(BanditPlacer):
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

        try:
            qp_seed = qp_init(init_pos, benchmark, _pp._load_plc(benchmark.name))
        except Exception as e:
            print(f"  [qp_init] FAILED: {e}", flush=True); qp_seed = None

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
                if time.time() - t0 > self.LEGALIZE_BUDGET: break
            if time.time() - t0 > self.LEGALIZE_BUDGET: break
        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f}", flush=True)
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

        arm_names = ["fd", "sur", "cd", "lns", "hard"]
        n_arms = 5
        pulls = np.array(self.INIT_PULLS, dtype=np.float64)
        sum_rewards = np.array(self.INIT_REWARDS, dtype=np.float64)

        while time.time() - t0 < soft_deadline:
            cycle += 1
            total_pulls = pulls.sum()
            means = sum_rewards / np.maximum(pulls, 1.0)
            ucb_bonus = self.UCB_C * np.sqrt(np.log(max(total_pulls, 2)) / np.maximum(pulls, 1.0))
            ucb = means + ucb_bonus
            ucb_pos = np.maximum(ucb, 1e-9)
            raw = ucb_pos / ucb_pos.sum()
            clamped = np.clip(raw, self.RATIO_MIN, self.RATIO_MAX)
            ratios = clamped / clamped.sum()

            if self.BANDIT_LOG:
                ratio_str = " ".join(f"{arm_names[i]}={ratios[i]:.2f}" for i in range(n_arms))
                print(f"  [bandit c{cycle}] ratios: {ratio_str}", flush=True)

            for arm_idx in range(n_arms):
                if time.time() - t0 > soft_deadline: break
                t_budget = cycle_t * ratios[arm_idx]
                if t_budget < 0.5:
                    continue
                cost_before = best_cost
                new_cost, wall = self._run_arm(arm_idx, best_pos, benchmark, incr_eval, plc_cd, t_budget)
                if new_cost is not None and new_cost < best_cost:
                    best_cost = new_cost
                gain = max(0.0, cost_before - best_cost)
                reward = gain / max(wall, 1e-3)
                pulls[arm_idx] += 1
                sum_rewards[arm_idx] += reward

            print(f"  [cycle_{cycle}] t={cycle_t:.1f} cost={best_cost:.6f}", flush=True)

            gain_overall = last_cost - best_cost
            if gain_overall < self.PLATEAU_THRESHOLD:
                plateau += 1
                cycle_t = max(self.MIN_CYCLE_T, cycle_t * self.SHRINK_FACTOR)
                if plateau >= self.PLATEAU_COUNT:
                    print(f"  [plateau] stop at cycle {cycle}", flush=True); break
            else:
                plateau = 0
                if gain_overall > self.GAIN_THRESHOLD:
                    cycle_t = min(self.MAX_CYCLE_T, cycle_t * self.GROW_FACTOR)
            last_cost = best_cost

        print(f"  [TOTAL] {time.time() - t0:.1f}s cycles={cycle} final={best_cost:.6f}", flush=True)
        out = incr_eval.macro_pos.copy()
        return torch.from_numpy(out.astype(np.float32))
