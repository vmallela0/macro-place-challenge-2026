"""exp_v3: Memetic crossover between two CD solutions.

Pipeline:
  1-2. push-apart + legalize (45s)
  3. CD1 (50s) → solution A
  4. Perturb A, run CD2 (40s) → solution B
  5. Crossover A + B → several candidates → legalize overlaps → pick best
  6. Final CD (40s) on the best candidate
  Total ~180s.
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
from _memetic import crossover_candidates


class OptimalPlacer:
    LEGALIZE_BUDGET = 45
    CD1_BUDGET = 50
    PERTURB_CD_BUDGET = 40
    FINAL_CD_BUDGET = 40

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

        sizes_np = benchmark.macro_sizes[:n_hard].numpy()
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]

        # Phase 1-2: push-apart + legalization
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

        # Phase 3: CD1 → parent A
        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)
        cd1_start = time.time()
        parent_a, cost_a = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=self.CD1_BUDGET, incr_eval=incr_eval)
        print(f"  [cd1 → A] {time.time() - cd1_start:.1f}s cost={cost_a:.6f}")

        best_pos, best_cost = parent_a, cost_a

        # Phase 4: perturb A, run CD2 → parent B
        rng = np.random.RandomState(9999)
        cd2_seed = parent_a.copy()
        n_perturb = max(5, min(len(movable_idx), rng.randint(8, 15)))
        chosen = rng.choice(movable_idx, size=n_perturb, replace=False)
        for idx in chosen:
            max_dim = max(sizes_np[idx, 0], sizes_np[idx, 1])
            scale = rng.uniform(0.4, 1.8)
            cd2_seed[idx, 0] = np.clip(
                cd2_seed[idx, 0] + rng.uniform(-1, 1) * max_dim * scale,
                sizes_np[idx, 0] / 2, float(benchmark.canvas_width) - sizes_np[idx, 0] / 2)
            cd2_seed[idx, 1] = np.clip(
                cd2_seed[idx, 1] + rng.uniform(-1, 1) * max_dim * scale,
                sizes_np[idx, 1] / 2, float(benchmark.canvas_height) - sizes_np[idx, 1] / 2)
        cd2_seed = _push_apart(cd2_seed, benchmark, max_iters=200, damping=0.6)
        if _has_overlap(cd2_seed):
            # Legalize harder
            for ot in range(8):
                sm = rng.choice([0.05, 0.08, 0.12, 0.18])
                cd2_seed = _legalize(cd2_seed, benchmark, order_type=ot, step_mult=sm)
                if not _has_overlap(cd2_seed):
                    break
            cd2_seed = _refine_toward_initial(cd2_seed, init_pos, benchmark)

        parent_b = None
        cost_b = float("inf")
        if not _has_overlap(cd2_seed):
            # Need a fresh incr_eval for divergent state
            plc_cd2 = _load_plc(benchmark.name)
            incr_eval2 = IncrementalEvaluator(plc_cd2, benchmark)
            cd2_start = time.time()
            parent_b, cost_b = _coord_descent(
                cd2_seed, benchmark, plc_cd2, max_time=self.PERTURB_CD_BUDGET, incr_eval=incr_eval2)
            print(f"  [cd2 → B] {time.time() - cd2_start:.1f}s cost={cost_b:.6f}")
            if cost_b < best_cost:
                best_cost = cost_b
                best_pos = parent_b

        # Phase 5: Crossover parents
        if parent_b is not None:
            # Ordered so parents[0] is best
            if cost_a <= cost_b:
                parents = [parent_a, parent_b]
            else:
                parents = [parent_b, parent_a]

            children = crossover_candidates(parents, benchmark, incr_eval, n_candidates=8)
            # Evaluate each (after legalization of overlaps)
            for ci, child in enumerate(children):
                child = _push_apart(child, benchmark, max_iters=200, damping=0.5)
                if _has_overlap(child):
                    continue
                full = benchmark.macro_positions.clone()
                full[:n_hard] = torch.tensor(child, dtype=torch.float32)
                r = compute_proxy_cost(full, benchmark, plc_eval)
                if r["overlap_count"] == 0:
                    if r["proxy_cost"] < best_cost:
                        best_cost = r["proxy_cost"]
                        best_pos = child
                        print(f"  [crossover_{ci}] child improves: {best_cost:.6f}")

        # Phase 6: final CD
        cd3_budget = max(10, self.FINAL_CD_BUDGET - (time.time() - t0 - 145))
        cd3_start = time.time()
        plc_cd3 = _load_plc(benchmark.name)
        incr_eval3 = IncrementalEvaluator(plc_cd3, benchmark)
        cd3_pos, cd3_cost = _coord_descent(
            best_pos, benchmark, plc_cd3, max_time=cd3_budget, incr_eval=incr_eval3)
        if cd3_cost < best_cost:
            best_cost = cd3_cost
            best_pos = cd3_pos
        print(f"  [final_cd] {time.time() - cd3_start:.1f}s cost={best_cost:.6f}")
        print(f"  [TOTAL] {time.time() - t0:.1f}s final_cost={best_cost:.6f}")

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        return full_pos
