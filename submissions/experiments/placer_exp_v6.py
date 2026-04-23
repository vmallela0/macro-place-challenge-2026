"""exp_v6: more soft-macro budget, cascade with hard CD.

v5 result: 180s → 0.88 on ibm01 with only 30s on soft.
v6 shifts more time to soft + hard interleaving.

Pipeline (target 180s on ibm01):
  legalize (45s) → cd1 (30s) → LNS+cd (15s) → tgt_lns+cd (10s)
  soft_A (20s) → hard_cd (8s) → soft_B (20s) → hard_cd (8s)
  → soft_C (15s) → hard_cd (9s)
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
from _targeted_lns import targeted_lns_phase
from _softmacro import soft_macro_cd


class OptimalPlacer:
    LEGALIZE_BUDGET = 45
    CD1_BUDGET = 30
    LNS_A = 12
    CD_A = 6
    TGT_LNS = 10
    CD_T = 5
    SOFT_A = 20
    HARD_A = 8
    SOFT_B = 20
    HARD_B = 8
    SOFT_C = 15
    HARD_C = 8

    def __init__(self, seed=42):
        self.seed = seed

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

        print(f"  [legalize] {time.time() - t0:.1f}s cost={best_cost:.6f}")
        if best_pos is None:
            return benchmark.macro_positions.clone()

        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        s = time.time()
        best_pos, best_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=self.CD1_BUDGET, incr_eval=incr_eval)
        print(f"  [cd1] {time.time() - s:.1f}s cost={best_cost:.6f}")

        s = time.time()
        p, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                        max_time=self.LNS_A, n_destroy=5,
                                        n_candidates=50, verbose=True)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [lns_A] {time.time() - s:.1f}s cost={best_cost:.6f}")
        s = time.time()
        p, c = _coord_descent(best_pos, benchmark, plc_cd, max_time=self.CD_A, incr_eval=incr_eval)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [cd_A] {time.time() - s:.1f}s cost={best_cost:.6f}")

        s = time.time()
        p, c = targeted_lns_phase(best_pos, benchmark, incr_eval,
                                  max_time=self.TGT_LNS, n_destroy=5,
                                  n_candidates=50, verbose=True)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [tgt_lns] {time.time() - s:.1f}s cost={best_cost:.6f}")
        s = time.time()
        p, c = _coord_descent(best_pos, benchmark, plc_cd, max_time=self.CD_T, incr_eval=incr_eval)
        if c < best_cost:
            best_cost, best_pos = c, p
        print(f"  [cd_T] {time.time() - s:.1f}s cost={best_cost:.6f}")

        # Soft/hard cascade
        for tag, soft_t, hard_t in [
            ("A", self.SOFT_A, self.HARD_A),
            ("B", self.SOFT_B, self.HARD_B),
            ("C", self.SOFT_C, self.HARD_C),
        ]:
            s = time.time()
            try:
                _, c = soft_macro_cd(best_pos, benchmark, incr_eval,
                                     max_time=soft_t, verbose=True)
                if c < best_cost:
                    best_cost = c
            except Exception as e:
                print(f"    soft_{tag} error: {e}")
            print(f"  [soft_{tag}] {time.time() - s:.1f}s cost={best_cost:.6f}")

            s = time.time()
            p, c = _coord_descent(best_pos, benchmark, plc_cd, max_time=hard_t,
                                  incr_eval=incr_eval)
            if c < best_cost:
                best_cost, best_pos = c, p
            print(f"  [hard_{tag}] {time.time() - s:.1f}s cost={best_cost:.6f}")

        print(f"  [TOTAL] {time.time() - t0:.1f}s final_internal_cost={best_cost:.6f}")

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        full_pos[n_hard:n_total] = torch.tensor(
            incr_eval.macro_pos[n_hard:n_total], dtype=torch.float32)
        return full_pos
