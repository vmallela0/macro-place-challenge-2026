"""FINAL SUBMISSION CANDIDATE: vmallela full pipeline + soft-macro tail.

Modifications from vmallela/placer.py:
  1. Reserve SOFT_PHASE_BUDGET seconds at the END for soft-macro optimization.
  2. Return soft macro positions (from incr_eval) in the final tensor.

TOTAL_TIME_LIMIT controls overall budget. SOFT_PHASE_BUDGET is how much of
the tail goes to soft-macro phases.

Runs can be tuned via environment variables:
  PLACER_TOTAL_BUDGET      — full budget in seconds (default 3300)
  PLACER_SOFT_BUDGET       — soft phase budget (default 600)
  PLACER_PARALLEL_WORKERS  — parallel restart workers (default 15)
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
_gradient_descent_exact = _mod._gradient_descent_exact
_cd_worker = _mod._cd_worker

_sib = Path(__file__).resolve().parent
sys.path.insert(0, str(_sib))
from _softmacro import soft_macro_cd
from _fd_soft import fd_soft_place
from _soft_lns import soft_lns_phase
from _moves import lns_destroy_repair_phase


class OptimalPlacer:
    def __init__(self, seed=42):
        self.seed = seed
        # Configurable budgets
        self.TOTAL_TIME_LIMIT = int(os.environ.get("PLACER_TOTAL_BUDGET", 3300))
        self.SOFT_PHASE_BUDGET = int(os.environ.get("PLACER_SOFT_BUDGET",
                                                     self.TOTAL_TIME_LIMIT // 5))
        self.PARALLEL_WORKERS = int(os.environ.get("PLACER_PARALLEL_WORKERS", 15))
        # Legalize budget scales with total (but minimum 60 for safety)
        self.LEGALIZE_BUDGET = max(60, min(600, self.TOTAL_TIME_LIMIT // 5))

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

        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        movable_idx = np.where(movable)[0]
        sizes_np = benchmark.macro_sizes[:n_hard].numpy()

        # Phase 1: Push-apart
        push_configs = [(300, 0.4), (500, 0.6), (800, 0.8)]
        pushed_positions = [
            _push_apart(init_pos, benchmark, max_iters=mi, damping=d)
            for mi, d in push_configs
        ]

        # Phase 2: Legalization tournament
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
        starts = [(p, f"push_{k}") for k, p in enumerate(pushed_positions)] + [(init_pos, "raw")]
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
            print(f"  [legalize] FAILED — fallback to raw init")
            return benchmark.macro_positions.clone()

        # Phase 3+4: CD + parallel restart
        plc_cd = _load_plc(benchmark.name)
        incr_eval = IncrementalEvaluator(plc_cd, benchmark)

        hard_deadline = self.TOTAL_TIME_LIMIT - self.SOFT_PHASE_BUDGET

        first_cd_budget = max(60, min(
            2000, hard_deadline - (time.time() - t0) - 100))
        cd_pos, cd_cost = _coord_descent(
            best_pos, benchmark, plc_cd, max_time=first_cd_budget, incr_eval=incr_eval)
        if cd_cost < best_cost:
            best_cost = cd_cost
            best_pos = cd_pos
        print(f"  [cd1] {time.time() - t0:.1f}s cost={best_cost:.6f}")

        # Short LNS+CD
        lns_t = min(60, max(10, (hard_deadline - (time.time() - t0)) * 0.05))
        if lns_t > 5:
            p, c = lns_destroy_repair_phase(best_pos, benchmark, incr_eval,
                                            max_time=lns_t, n_destroy=5,
                                            n_candidates=50)
            if c < best_cost:
                best_cost, best_pos = c, p
            print(f"  [lns] cost={best_cost:.6f}")

            cd_t = min(30, max(5, lns_t / 2))
            p, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=cd_t, incr_eval=incr_eval)
            if c < best_cost:
                best_cost, best_pos = c, p
            print(f"  [cd_lns] cost={best_cost:.6f}")

        # Parallel restart (optional if PARALLEL_WORKERS > 0)
        parallel_remaining = hard_deadline - (time.time() - t0)
        if self.PARALLEL_WORKERS > 0 and parallel_remaining > 180:
            n_workers = min(self.PARALLEL_WORKERS, max(1, mp.cpu_count() - 1))
            rng_restart = np.random.RandomState(42)

            worker_starts = []
            for w in range(n_workers):
                cd_start = best_pos.copy()
                n_perturb = max(2, min(len(movable_idx), rng_restart.randint(3, 8)))
                chosen = rng_restart.choice(movable_idx, size=n_perturb, replace=False)
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
                if _has_overlap(cd_start):
                    for ot in range(8):
                        sm = rng_restart.choice([0.05, 0.08, 0.12, 0.18])
                        cd_start = _legalize(cd_start, benchmark, order_type=ot, step_mult=sm)
                        if not _has_overlap(cd_start):
                            break
                    cd_start = _refine_toward_initial(cd_start, init_pos, benchmark)
                    if _has_overlap(cd_start):
                        continue
                worker_starts.append(cd_start)

            if worker_starts:
                cd_time_per_worker = max(60, parallel_remaining - 30)
                worker_args = [
                    (s.tobytes(), n_hard, benchmark.name, cd_time_per_worker, 1000 + w)
                    for w, s in enumerate(worker_starts)
                ]
                try:
                    with mp.Pool(len(worker_starts)) as pool:
                        results = pool.map(_cd_worker, worker_args)
                    for pos_bytes, cost, nh in results:
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = np.frombuffer(pos_bytes, dtype=np.float64).reshape(nh, 2).copy()
                except Exception as e:
                    print(f"    parallel error: {e}")
            print(f"  [parallel] cost={best_cost:.6f}")

        # Phase 5: Soft-macro optimization tail
        soft_deadline = self.TOTAL_TIME_LIMIT - 10
        incr_eval.sync_positions(best_pos)

        cycle = 0
        last_cost = best_cost
        plateau = 0

        # Per-cycle budget based on remaining time and target cycle count
        remaining = soft_deadline - (time.time() - t0)
        target_cycles = 15
        per_cycle = max(10, int(remaining / target_cycles))

        while time.time() - t0 < soft_deadline:
            cycle += 1

            fd_t = max(2, int(per_cycle * 0.05))
            try:
                _, c = fd_soft_place(best_pos, benchmark, incr_eval,
                                     max_time=fd_t, damping=0.3,
                                     check_cost=True, verbose=False)
                if c < best_cost:
                    best_cost = c
            except Exception as e:
                print(f"    fd_{cycle} error: {e}")
            if time.time() - t0 > soft_deadline:
                break

            scd_t = max(8, int(per_cycle * 0.45))
            try:
                _, c = soft_macro_cd(best_pos, benchmark, incr_eval,
                                     max_time=scd_t, verbose=False)
                if c < best_cost:
                    best_cost = c
            except Exception as e:
                print(f"    scd_{cycle} error: {e}")
            if time.time() - t0 > soft_deadline:
                break

            slns_t = max(8, int(per_cycle * 0.35))
            try:
                _, c = soft_lns_phase(best_pos, benchmark, incr_eval,
                                      max_time=slns_t, n_destroy=8,
                                      n_candidates=30, verbose=False)
                if c < best_cost:
                    best_cost = c
            except Exception as e:
                print(f"    slns_{cycle} error: {e}")
            if time.time() - t0 > soft_deadline:
                break

            hard_t = max(5, int(per_cycle * 0.15))
            p, c = _coord_descent(best_pos, benchmark, plc_cd,
                                  max_time=hard_t, incr_eval=incr_eval)
            if c < best_cost:
                best_cost, best_pos = c, p

            print(f"  [cycle_{cycle}] {time.time() - t0:.1f}s cost={best_cost:.6f}")

            if abs(last_cost - best_cost) < 5e-5:
                plateau += 1
                if plateau >= 3:
                    print(f"  [plateau] stop at cycle {cycle}")
                    break
            else:
                plateau = 0
            last_cost = best_cost

        print(f"  [TOTAL] {time.time() - t0:.1f}s cycles={cycle} final={best_cost:.6f}")

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        full_pos[n_hard:n_total] = torch.tensor(
            incr_eval.macro_pos[n_hard:n_total], dtype=torch.float32)
        return full_pos
