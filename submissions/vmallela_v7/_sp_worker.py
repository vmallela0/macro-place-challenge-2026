"""Module-level worker for the SP-multi-worker basin-hop pool.

Lives in its own module so multiprocessing's pickle (used by spawn-context
Pool) can resolve it by qualified name. When placer.py is loaded via
`importlib.util.spec_from_file_location`, functions defined inside it can't
be pickled because the module name doesn't match the loader's lookup path.
A standalone module imported normally (sys.path entry → import _sp_worker)
gets a clean __module__ attribute that pickle can use.

Used by Option C: 4 worker processes spawn from the same SP-perturbed init
position with different LNS/SA seeds; min across workers as the hop result.
"""
from __future__ import annotations
import os
import sys
import numpy as np


def sp_hop_worker(args):
    """Multi-worker SP basin-hop worker. See module docstring.

    args: (bench_path, init_pos_bytes, init_pos_shape, init_pos_dtype,
           budget, seed)
    Returns: (final_pos_bytes, shape, dtype_str, cost, overlaps, seed)
    """
    (bench_path, init_pos_bytes, init_pos_shape, init_pos_dtype,
     budget, seed) = args

    # Hardware-portable determinism (mirror v6 worker setup).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from pathlib import Path as _Path
    ROOT = _Path(bench_path).resolve().parents[3]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

    import torch
    import importlib.util as _ilu
    v1_spec = _ilu.spec_from_file_location(
        "_v1_sp_hop_w", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = _ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)

    from macro_place.benchmark import Benchmark as _Benchmark
    from macro_place.objective import compute_proxy_cost as _cpc
    from _soft_laplacian import apply_laplacian_refine as _lap

    bench = _Benchmark.load(bench_path)
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    n_hard = bench.num_hard_macros
    n_total = bench.macro_positions.shape[0]

    init_pos = np.frombuffer(
        init_pos_bytes, dtype=np.dtype(init_pos_dtype)
    ).reshape(init_pos_shape).copy()

    # Reduced pipeline: push-apart → legalize → refine_toward_initial.
    init_hard = init_pos[:n_hard].astype(np.float64)
    pushed = v1._push_apart(init_hard, bench, max_iters=300, damping=0.4)
    legal = v1._legalize(pushed, bench, order_type=0, step_mult=0.05)
    refined_hard = v1._refine_toward_initial(legal, init_hard, bench)

    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.sync_positions(refined_hard)
    incr.macro_pos[n_hard:] = init_pos[n_hard:n_total].astype(
        incr.macro_pos.dtype)
    np.clip(incr.macro_pos[n_hard:, 0], 0.0, cw,
            out=incr.macro_pos[n_hard:, 0])
    np.clip(incr.macro_pos[n_hard:, 1], 0.0, ch,
            out=incr.macro_pos[n_hard:, 1])
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    _lap(incr, bench, verbose=False)

    cd_budget = budget * 0.5
    sa_T0 = float(os.environ.get("PLACER_SA_T0", "0.00005"))
    sa_cooling = float(os.environ.get("PLACER_SA_COOLING", "0.9995"))
    try:
        best_hard, best_cost = v1._coord_descent(
            refined_hard, bench, plc, max_time=cd_budget,
            incr_eval=incr, sa_T0=sa_T0, sa_cooling=sa_cooling,
            sa_rng_seed=seed)
    except Exception:
        best_hard = refined_hard
        best_cost = float(incr.get_proxy_cost())

    try:
        from _per_net import per_net_optimize as _pn
        new_pos, c = _pn(best_hard, bench, incr, max_time=budget * 0.1)
        if c < best_cost:
            best_cost, best_hard = c, new_pos
    except Exception:
        pass

    try:
        from _moves import lns_destroy_repair_phase as _lns
        p, c = _lns(best_hard, bench, incr, max_time=budget * 0.15,
                    n_destroy=5, n_candidates=50)
        if c < best_cost:
            best_cost, best_hard = c, p
    except Exception:
        pass

    try:
        _lap(incr, bench, verbose=False)
    except Exception:
        pass

    full_out = np.zeros((n_total, 2), dtype=np.float64)
    full_out[:n_hard] = np.asarray(incr.macro_pos[:n_hard])
    full_out[n_hard:] = np.asarray(incr.macro_pos[n_hard:n_total])

    full_tensor = torch.tensor(full_out, dtype=torch.float32)
    r = _cpc(full_tensor, bench, plc)
    final_cost = float(r["proxy_cost"])
    final_overlaps = int(r["overlap_count"])

    return (full_out.tobytes(), full_out.shape, str(full_out.dtype),
            final_cost, final_overlaps, seed)
