"""Generate the v6-gpu vs CPU CD convergence plot on ibm01.

Runs both CD variants from the same legalized starting point, polls the
current proxy cost every 0.25s via a daemon thread, plots both curves on
the same axes. Output: assets/v6_gpu_vs_cpu_ibm01.png

The polling reads `IncrementalEvaluator.get_proxy_cost()` (a pure-numpy
reduction over already-maintained per-net HPWL / per-cell density / per-
cell congestion arrays). Numpy releases the GIL on these reductions, so
the poller does not block CD progress.
"""
from __future__ import annotations
import importlib.util
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

_spec = importlib.util.spec_from_file_location(
    "_v1", str(ROOT / "submissions" / "vmallela" / "placer.py"))
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

_load_plc = _v1._load_plc
IncrementalEvaluator = _v1.IncrementalEvaluator
_push_apart = _v1._push_apart
_legalize = _v1._legalize
_refine_toward_initial = _v1._refine_toward_initial
_coord_descent = _v1._coord_descent

from macro_place.benchmark import Benchmark
from _torch_eval import TorchBatchEvaluator
from _gpu_cd import gpu_mass_cd


BUDGET = 60.0
SA_T0 = 0.00005
SA_COOLING = 0.9995
POLL_EVERY = 0.25


def _polled_run(label, cd_fn, cd_kwargs, incr):
    """Run cd_fn(**cd_kwargs) while a daemon thread polls incr.get_proxy_cost()."""
    log: list[tuple[float, float]] = []
    stop = threading.Event()
    t0 = time.time()

    def poller():
        while not stop.is_set():
            try:
                c = float(incr.get_proxy_cost())
                log.append((time.time() - t0, c))
            except Exception:
                pass
            stop.wait(POLL_EVERY)

    th = threading.Thread(target=poller, daemon=True)
    th.start()
    result = cd_fn(**cd_kwargs)
    stop.set()
    th.join(timeout=2.0)
    elapsed = time.time() - t0
    final_cost = float(result[1]) if isinstance(result, tuple) else None
    print(f"  [{label}] elapsed={elapsed:.1f}s  final={final_cost:.6f}  "
          f"polled samples={len(log)}", flush=True)
    return log, final_cost


def main():
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / "ibm01.pt"))
    plc = _load_plc("ibm01")
    init_pos = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    pushed = _push_apart(init_pos, bench, max_iters=300, damping=0.4)
    legal = _legalize(pushed, bench, order_type=0, step_mult=0.05)
    refined = _refine_toward_initial(legal, init_pos, bench)

    # CPU run
    incr_cpu = IncrementalEvaluator(_load_plc("ibm01"), bench)
    incr_cpu.sync_positions(refined.copy())
    init_cost = float(incr_cpu.get_proxy_cost())
    print(f"init cost: {init_cost:.6f}")

    cpu_kwargs = dict(
        pos_np=refined.copy(),
        benchmark=bench,
        plc_eval=plc,
        max_time=BUDGET,
        incr_eval=incr_cpu,
        sa_T0=SA_T0,
        sa_cooling=SA_COOLING,
        sa_rng_seed=42,
    )
    cpu_log, cpu_final = _polled_run("CPU CD (v4 baseline)",
                                      _coord_descent, cpu_kwargs, incr_cpu)

    # GPU run (same starting point)
    incr_gpu = IncrementalEvaluator(_load_plc("ibm01"), bench)
    incr_gpu.sync_positions(refined.copy())
    gpu = TorchBatchEvaluator(incr_gpu, bench)
    print(f"  torch device: {gpu.device}")

    gpu_kwargs = dict(
        pos_np=refined.copy(),
        benchmark=bench,
        plc_eval=plc,
        max_time=BUDGET,
        incr_eval=incr_gpu,
        gpu_eval=gpu,
        K=32,
        sa_T0=SA_T0,
        sa_cooling=SA_COOLING,
        seed=42,
    )
    gpu_log, gpu_final = _polled_run("GPU CD (v6-gpu)",
                                      gpu_mass_cd, gpu_kwargs, incr_gpu)

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))

    if cpu_log:
        t_cpu = np.array([p[0] for p in cpu_log])
        c_cpu = np.minimum.accumulate(np.array([p[1] for p in cpu_log]))
        ax.plot(t_cpu, c_cpu, color="#cc4444", lw=2.0,
                label=f"CPU CD (v4 baseline)  final = {cpu_final:.4f}")

    if gpu_log:
        t_gpu = np.array([p[0] for p in gpu_log])
        c_gpu = np.minimum.accumulate(np.array([p[1] for p in gpu_log]))
        ax.plot(t_gpu, c_gpu, color="#1f77b4", lw=2.0,
                label=f"GPU CD (v6-gpu, torch.MPS)  final = {gpu_final:.4f}")

    ax.axhline(y=init_cost, color="#888888", ls=":", lw=1.0,
               label=f"start (after legalize+refine)  = {init_cost:.4f}")

    delta = cpu_final - gpu_final
    ax.set_xlabel("wall-clock seconds (single seed, SA T0=5e-5)", fontsize=11)
    ax.set_ylabel("proxy cost (best-so-far)", fontsize=11)
    ax.set_title(
        f"v6-gpu vs v4 CPU coordinate descent — ibm01, {int(BUDGET)} s budget\n"
        f"GPU wins by {delta:+.4f} (cross-macro batched torch evaluator, MPS / RTX 6000 Ada)",
        fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out_path = ROOT / "assets" / "v6_gpu_vs_cpu_ibm01.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
