"""Experimental placer v1: adds cluster-translate + LNS + gradient-step phases.

Pipeline:
  Phase 1-3 (base_fast): push-apart, legalization, single CD run
  Phase 4: Cluster translation (novel — rigid multi-macro moves)
  Phase 5: Gradient-step CD (novel — continuous direction, not 8-axis)
  Phase 6: LNS destroy+repair (novel — coupled re-insertion)
  Phase 7: Final CD polish
"""

import sys
import time
import random
from pathlib import Path
import importlib.util
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vmallela"))

_spec = importlib.util.spec_from_file_location(
    "_vmallela_placer",
    str(Path(__file__).resolve().parents[1] / "vmallela" / "placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from macro_place.benchmark import Benchmark

_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator
_push_apart = _mod._push_apart
_legalize = _mod._legalize
_refine_toward_initial = _mod._refine_toward_initial
_coord_descent = _mod._coord_descent

# Import move operators from sibling module
_sib = Path(__file__).resolve().parent
sys.path.insert(0, str(_sib))
from _moves import cluster_translate_phase, lns_destroy_repair_phase, gradient_step_phase


class OptimalPlacer:
    # Budget (seconds) — short for fast iteration on ibm01.
    # Total target: ~180s
    LEGALIZE_BUDGET = 45
    CD1_BUDGET = 60          # initial CD run
    CLUSTER_BUDGET = 20      # cluster translation phase
    GRAD_BUDGET = 15         # gradient-step phase
    LNS_BUDGET = 25          # LNS phase
    CD2_BUDGET = 15          # final CD polish

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
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        # Phase 1: push-apart
        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d) for mi, d in push_configs]

        # Phase 2: legalization tournament
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
            if time.time() - t0 > self.LEGALIZE_BUDGET:
                break
            for sm in [0.05, 0.08, 0.12, 0.18]:
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
                    _try(refined)

        legalize_cost = best_cost
        print(f"  [legalize] {time.time() - t0:.1f}s cost={legalize_cost:.6f}")

        if best_pos is None:
            return benchmark.macro_positions.clone()

        # Phase 3: initial CD run
        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        cd1_start = time.time()
        cd_pos, cd_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=self.CD1_BUDGET, incr_eval=incr_eval)
        if cd_cost < best_cost:
            best_cost = cd_cost
            best_pos = cd_pos
        print(f"  [cd1] {time.time() - cd1_start:.1f}s cost={best_cost:.6f}")

        # Phase 4: Cluster translation
        cluster_start = time.time()
        try:
            ct_pos, ct_cost = cluster_translate_phase(
                best_pos, benchmark, incr_eval, max_time=self.CLUSTER_BUDGET, verbose=True)
            if ct_cost < best_cost:
                best_cost = ct_cost
                best_pos = ct_pos
        except Exception as e:
            print(f"    cluster error: {e}")
        print(f"  [cluster] {time.time() - cluster_start:.1f}s cost={best_cost:.6f}")

        # Phase 5: Gradient-step
        grad_start = time.time()
        try:
            gs_pos, gs_cost = gradient_step_phase(
                best_pos, benchmark, incr_eval, max_time=self.GRAD_BUDGET, verbose=True)
            if gs_cost < best_cost:
                best_cost = gs_cost
                best_pos = gs_pos
        except Exception as e:
            print(f"    grad error: {e}")
        print(f"  [grad] {time.time() - grad_start:.1f}s cost={best_cost:.6f}")

        # Phase 6: LNS
        lns_start = time.time()
        try:
            ln_pos, ln_cost = lns_destroy_repair_phase(
                best_pos, benchmark, incr_eval, max_time=self.LNS_BUDGET, verbose=True)
            if ln_cost < best_cost:
                best_cost = ln_cost
                best_pos = ln_pos
        except Exception as e:
            print(f"    lns error: {e}")
        print(f"  [lns] {time.time() - lns_start:.1f}s cost={best_cost:.6f}")

        # Phase 7: Final CD polish
        cd2_remaining = max(5, self.CD2_BUDGET)
        cd2_start = time.time()
        cd2_pos, cd2_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=cd2_remaining, incr_eval=incr_eval)
        if cd2_cost < best_cost:
            best_cost = cd2_cost
            best_pos = cd2_pos
        print(f"  [cd2] {time.time() - cd2_start:.1f}s cost={best_cost:.6f}")
        print(f"  [TOTAL] {time.time() - t0:.1f}s final_cost={best_cost:.6f}")

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        return full_pos
