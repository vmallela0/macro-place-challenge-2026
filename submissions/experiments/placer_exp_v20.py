"""exp_v20: Kitchen sink — combines best-performing phases.

  - Full legalization tournament (80s)
  - CD1 (25s) + LNS (10s) + CD (5s)
  - Cycles with: FD + surrogate-v2 soft + targeted soft LNS + soft SA + hard CD
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
from _moves import lns_destroy_repair_phase
from _softmacro import soft_macro_cd
from _fd_soft import fd_soft_place
from _targeted_soft_lns import targeted_soft_lns_phase
from _soft_sa import soft_cd_sa
from _soft_surrogate_v2 import soft_cd_surrogate_v2, reset_surrogate_state


class OptimalPlacer:
    LEGALIZE_BUDGET = 75
    CD1_BUDGET = 25
    LNS_BUDGET = 10
    CD_LNS = 5
    # Per-cycle: FD 2s / surrogate-soft 8s / tgt-soft-LNS 8s / soft-SA 4s / hard-CD 5s = 27s
    FD_T = 2
    SURR_SOFT_T = 8
    TGT_LNS_T = 8
    SA_T = 4
    HARD_T = 5
    TOTAL_BUDGET = 220

    def __init__(self, seed=42):
        self.seed = seed

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

        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f}", flush=True)
        if best_pos is None:
            return benchmark.macro_positions.clone()

        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        s = time.time()
        best_pos, best_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=self.CD1_BUDGET, incr_eval=incr_eval)
        print(f"  [cd1] {time.time() - s:.1f}s cost={best_cost:.6f}", flush=True)

        s = time.time()
        p, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                        max_time=self.LNS_BUDGET, n_destroy=5,
                                        n_candidates=50)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [lns] {time.time() - s:.1f}s cost={best_cost:.6f}", flush=True)
        s = time.time()
        p, c = _coord_descent(best_pos, benchmark, plc_cd, max_time=self.CD_LNS, incr_eval=incr_eval)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [cd_lns] {time.time() - s:.1f}s cost={best_cost:.6f}", flush=True)

        cycle = 0
        while time.time() - t0 < self.TOTAL_BUDGET - 5:
            cycle += 1
            try:
                _, c = fd_soft_place(best_pos, benchmark, incr_eval,
                                     max_time=self.FD_T, damping=0.3)
                if c < best_cost:
                    best_cost = c
            except Exception: pass

            if time.time() - t0 > self.TOTAL_BUDGET - 5: break
            try:
                _, c = soft_cd_surrogate_v2(best_pos, benchmark, incr_eval,
                                            max_time=self.SURR_SOFT_T, verbose=False)
                if c < best_cost:
                    best_cost = c
            except Exception: pass

            if time.time() - t0 > self.TOTAL_BUDGET - 5: break
            try:
                _, c = targeted_soft_lns_phase(best_pos, benchmark, incr_eval,
                                               max_time=self.TGT_LNS_T,
                                               n_destroy=8, n_candidates=30)
                if c < best_cost:
                    best_cost = c
            except Exception: pass

            if time.time() - t0 > self.TOTAL_BUDGET - 5: break
            try:
                _, c = soft_cd_sa(best_pos, benchmark, incr_eval,
                                  max_time=self.SA_T, t_start=0.008, t_end=1e-5)
                if c < best_cost:
                    best_cost = c
            except Exception: pass

            if time.time() - t0 > self.TOTAL_BUDGET - 5: break
            p, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=self.HARD_T, incr_eval=incr_eval)
            if c < best_cost:
                best_cost, best_pos = c, p
            print(f"  [cycle_{cycle}] cost={best_cost:.6f}", flush=True)

        print(f"  [TOTAL] {time.time() - t0:.1f}s cycles={cycle} final={best_cost:.6f}",
              flush=True)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        full_pos[n_hard:n_total] = torch.tensor(
            incr_eval.macro_pos[n_hard:n_total], dtype=torch.float32)
        return full_pos
