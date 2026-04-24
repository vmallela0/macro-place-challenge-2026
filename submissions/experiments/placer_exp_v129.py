"""exp_v129 (idea IP10): 8 push-apart damping configs instead of 3.

Trivial tournament-expansion test: does adding more diverse push-apart
seeds change the legalize outcome?
"""
import os, sys, time, random
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_s = importlib.util.spec_from_file_location("_sv4", str(Path(__file__).resolve().parent / "placer_submission_v4.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

from macro_place.benchmark import Benchmark
_load_plc = _m._load_plc
IncrementalEvaluator = _m.IncrementalEvaluator
_push_apart = _m._push_apart
_legalize = _m._legalize
_refine_toward_initial = _m._refine_toward_initial
_coord_descent = _m._coord_descent
_cd_worker = _m._cd_worker
soft_macro_cd = _m.soft_macro_cd
fd_soft_place = _m.fd_soft_place
soft_lns_phase = _m.soft_lns_phase
per_net_optimize = _m.per_net_optimize
soft_cd_surrogate_v2 = _m.soft_cd_surrogate_v2
reset_surrogate_state = _m.reset_surrogate_state
lns_destroy_repair_phase = _m.lns_destroy_repair_phase

_Base = _m.OptimalPlacer


class OptimalPlacer(_Base):
    """Same pipeline but 8 push-apart variants feeding the legalize tournament."""
    # Expanded set: 8 damping values × 1 iter-count each (cheaper per variant)
    _PUSH_CONFIGS = [
        (200, 0.3), (300, 0.4), (400, 0.5), (500, 0.6),
        (600, 0.7), (700, 0.75), (800, 0.8), (1000, 0.85),
    ]

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Monkey-patch push_configs inside _Base.place by replicating the relevant prelude
        # Simplest path: copy the Base.place method's first step (generate pushed_positions)
        # and then invoke the rest via calling super with a pre-populated attribute.
        # Since the base method reconstructs push_configs locally, the cleanest override
        # is to fully replicate place() here — but we want to stay thin.
        #
        # Trick: monkey-patch the local name `push_configs` doesn't work. Instead we
        # monkey-patch `_push_apart` to produce a pre-computed list indexed by call order.
        # But _Base.place only calls _push_apart 3 times.
        # Simpler: rewrite the first block inline.
        #
        # For this test we copy the whole place() body, with push_configs substituted.
        # Keep in sync with placer_submission_v4.py.
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

        # === CHANGED: 8 push configs instead of 3 ===
        push_configs = self._PUSH_CONFIGS
        pushed_positions = [_push_apart(init_pos, benchmark, max_iters=mi, damping=d)
                            for mi, d in push_configs]
        print(f"  [push_apart] {time.time() - t0:.1f}s ({len(pushed_positions)} variants)", flush=True)

        # === rest is identical to placer_submission_v4.py's place() ===
        hard_deadline = self.TOTAL_TIME_LIMIT
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

        # ... delegate the rest to super by temporarily patching init to best_legal
        # Simpler: from here, we replicate what Base.place does from phase 3 onward.
        # But that's 80+ more lines of replication. Compromise: accept that we're only
        # testing the tournament-width effect, and the rest of the pipeline runs the
        # same way. We hook via a tiny wrapper: save best_legal as "init" and call super.
        benchmark.macro_positions[:n_hard] = torch.from_numpy(best_legal).float()
        # Save the already-spent time via env (not persistent), we just let Base re-run
        # push (fast at this point since legal is close), legalize (fast, already near-best),
        # and consume the rest of the budget on CD+cycles. This is wasteful but simple.
        return super().place(benchmark)
