"""Fast-budget baseline: mirrors vmallela/placer.py but caps total time at ~3min on ibm01.

Short-budget iteration baseline. This file just imports the real placer and
overrides the TOTAL_TIME_LIMIT / LEGALIZE_TIME_BUDGET via subclass injection.

Usage:
    uv run evaluate submissions/experiments/placer_base_fast.py -b ibm01
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vmallela"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_vmallela_placer",
    str(Path(__file__).resolve().parents[1] / "vmallela" / "placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

import time
import random
import numpy as np
import torch
import multiprocessing as mp
from macro_place.benchmark import Benchmark

_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator
_push_apart = _mod._push_apart
_legalize = _mod._legalize
_refine_toward_initial = _mod._refine_toward_initial
_coord_descent = _mod._coord_descent
_cd_worker = _mod._cd_worker


class OptimalPlacer:
    """Short-budget variant: ~3 min on ibm01 for fast iteration."""
    # Tunables (exposed so subclasses can override)
    LEGALIZE_BUDGET = 45      # was 600
    TOTAL_BUDGET = 180        # was 3300
    N_RESTART_WORKERS = 0     # disable parallel restart for speed
    GD_PHASE_SEC = 0          # disable GD for speed

    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        init_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        t0 = time.time()

        from macro_place.objective import compute_proxy_cost

        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        # Phase 1: push-apart (3 configs)
        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d) for mi, d in push_configs]

        # Phase 2: legalization tournament (short budget)
        plc_eval = _load_plc(benchmark.name)
        best_pos, best_cost = None, float("inf")

        def _has_overlap(pos_np):
            for i in range(n_hard):
                for j in range(i + 1, n_hard):
                    if (abs(pos_np[i, 0] - pos_np[j, 0]) < (sizes_np[i, 0] + sizes_np[j, 0]) / 2 and
                        abs(pos_np[i, 1] - pos_np[j, 1]) < (sizes_np[i, 1] + sizes_np[j, 1]) / 2):
                        return True
            return False

        def _try_candidate(pos_np):
            nonlocal best_pos, best_cost
            if _has_overlap(pos_np):
                return
            full = benchmark.macro_positions.clone()
            full[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
            result = compute_proxy_cost(full, benchmark, plc_eval)
            if result["overlap_count"] == 0 and result["proxy_cost"] < best_cost:
                best_cost = result["proxy_cost"]
                best_pos = pos_np.copy()

        seen = set()
        step_sizes = [0.05, 0.08, 0.12, 0.18]
        starts = [(p, f"push_{k}") for k, p in enumerate(pushed)] + [(init_pos, "raw")]
        for ot in range(30):
            if time.time() - t0 > self.LEGALIZE_BUDGET:
                break
            for sm in step_sizes:
                if time.time() - t0 > self.LEGALIZE_BUDGET:
                    break
                for sp, _ in starts:
                    if time.time() - t0 > self.LEGALIZE_BUDGET:
                        break
                    legal = _legalize(sp, benchmark, order_type=ot, step_mult=sm)
                    refined = _refine_toward_initial(legal, init_pos, benchmark)
                    h = hash(np.round(refined * 10).astype(np.int32).tobytes())
                    if h in seen:
                        continue
                    seen.add(h)
                    _try_candidate(refined)

        # Phase 3: single CD run
        if best_pos is not None:
            plc_cd = _load_plc(benchmark.name)
            incr_eval = IncrementalEvaluator(plc_cd, benchmark)
            cd_budget = self.TOTAL_BUDGET - (time.time() - t0) - 5
            if cd_budget > 10:
                cd_pos, cd_cost = _coord_descent(
                    best_pos, benchmark, plc_cd, max_time=cd_budget, incr_eval=incr_eval)
                if cd_cost < best_cost:
                    best_cost = cd_cost
                    best_pos = cd_pos

        full_pos = benchmark.macro_positions.clone()
        if best_pos is not None:
            full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        return full_pos
