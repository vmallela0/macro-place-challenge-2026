"""exp_v48: v36 + HUGE LNS (destroy 20-30 hards) early, then cycles."""
import sys
import time
import random
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

_sib = Path(__file__).resolve().parent
sys.path.insert(0, str(_sib))
from _softmacro import soft_macro_cd
from _fd_soft import fd_soft_place
from _soft_lns import soft_lns_phase
from _moves import lns_destroy_repair_phase


class OptimalPlacer:
    def __init__(self, seed=42):
        self.seed = seed
        self.TOTAL_BUDGET = 220
        self.LEGALIZE_BUDGET = 75

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

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
        pushed = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d) for mi, d in push_configs]

        plc_eval = _load_plc(benchmark.name)
        best_pos, best_cost = None, float("inf")

        def _has_overlap(pos_np):
            for i in range(n_hard):
                for j in range(i + 1, n_hard):
                    if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes_np[i, 0] + sizes_np[j, 0]) / 2 and
                        abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes_np[i, 1] + sizes_np[j, 1]) / 2):
                        return True
            return False

        def _try(pos_np):
            nonlocal best_pos, best_cost
            if _has_overlap(pos_np):
                return
            full = benchmark.macro_positions.clone()
            full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
            r = compute_proxy_cost(full, benchmark, plc_eval)
            if r["overlap_count"] == 0 and r["proxy_cost"] < best_cost:
                best_cost = r["proxy_cost"]
                best_pos = pos_np.copy()

        seen = set()
        starts = [(p, f"push_{k}") for k, p in enumerate(pushed)] + [(init_pos, "raw")]
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

        if best_pos is None:
            return benchmark.macro_positions.clone()

        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        best_pos, best_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=25, incr_eval=incr_eval)

        # HUGE LNS: destroy 25 hards, 150 candidates
        _, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                        max_time=25, n_destroy=25,
                                        n_candidates=150)
        if c < best_cost: best_cost = c
        _, c = _coord_descent(best_pos, benchmark, plc_cd, max_time=10, incr_eval=incr_eval)
        if c < best_cost: best_cost = c

        cycle = 0
        cycle_t = 30
        last_cost = best_cost
        while time.time() - t0 < self.TOTAL_BUDGET - 5:
            cycle += 1
            try:
                _, c = fd_soft_place(best_pos, benchmark, incr_eval,
                                     max_time=cycle_t * 0.05, damping=0.3, check_cost=True)
                if c < best_cost: best_cost = c
            except Exception: pass
            try:
                _, c = soft_macro_cd(best_pos, benchmark, incr_eval, max_time=cycle_t * 0.4)
                if c < best_cost: best_cost = c
            except Exception: pass
            try:
                _, c = soft_lns_phase(best_pos, benchmark, incr_eval,
                                      max_time=cycle_t * 0.4, n_destroy=8, n_candidates=30)
                if c < best_cost: best_cost = c
            except Exception: pass
            _, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=cycle_t * 0.15, incr_eval=incr_eval)
            if c < best_cost: best_cost = c

            gain = last_cost - best_cost
            if gain < 1e-4:
                cycle_t = max(8, cycle_t * 0.7)
            elif gain > 0.01:
                cycle_t = min(40, cycle_t * 1.1)
            last_cost = best_cost

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        full_pos[n_hard:n_total] = torch.tensor(
            incr_eval.macro_pos[n_hard:n_total], dtype=torch.float32)
        return full_pos
