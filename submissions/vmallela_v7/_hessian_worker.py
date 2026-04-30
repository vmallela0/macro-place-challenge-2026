"""Module-level worker for Hessian-escape multi-worker phase.

Lives in its own module so multiprocessing's pickle (used by spawn-
context Pool) can resolve it by qualified name. Mirror of _sp_worker.py
for the SP basin-hop.

Each worker takes a Hessian-perturbed init position and runs the v4
pipeline at the given budget. Returns the final placement + cost +
overlap count for the parent's strict-improvement gate.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np


def hessian_candidate_worker(args):
    """args: (bench_path, init_pos_bytes, shape, dtype_str, budget,
              seed, label).
    Returns: (label, final_pos_bytes, shape, dtype_str, cost, overlaps).
    """
    (bench_path, init_pos_bytes, shape, dtype_str, budget, seed,
     label) = args

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PLACER_TOTAL_BUDGET"] = str(budget)
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
    spec = _ilu.spec_from_file_location(
        "_v4_pl_hess_w", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    v4 = _ilu.module_from_spec(spec); spec.loader.exec_module(v4)
    spec1 = _ilu.spec_from_file_location(
        "_v1_pl_hess_w", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = _ilu.module_from_spec(spec1); spec1.loader.exec_module(v1)

    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost

    bench = Benchmark.load(bench_path)
    init_pos = np.frombuffer(
        init_pos_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    bench.macro_positions = torch.tensor(
        init_pos, dtype=bench.macro_positions.dtype)

    plc = v1._load_plc(bench.name)
    placer = v4.OptimalPlacer(seed=seed)
    pos = placer.place(bench)
    if isinstance(pos, torch.Tensor):
        pos_np = np.asarray(pos.detach().cpu().numpy(), dtype=np.float64)
    else:
        pos_np = np.asarray(pos, dtype=np.float64)
    pos_t = torch.tensor(pos_np, dtype=torch.float32)
    r = compute_proxy_cost(pos_t, bench, plc)
    final_cost = float(r["proxy_cost"])
    final_overlaps = int(r["overlap_count"])

    pos_out = np.ascontiguousarray(pos_np, dtype=np.float64)
    return (label, pos_out.tobytes(), pos_out.shape, str(pos_out.dtype),
            final_cost, final_overlaps)
