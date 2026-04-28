"""vmallela_v6 — GPU + multi-process portfolio placer.

Algorithm
---------
1.  Spawn N worker processes in parallel (default: N=8 on a 16+core machine).
2.  Each worker runs the full v4 pipeline (push-apart → legalize tournament
    → hard CD → per-net step → hard LNS → soft cycles → escape basin) at a
    different RNG seed.
3.  One worker (the "GPU worker") swaps the hard-CD step for `gpu_mass_cd`
    backed by `TorchBatchEvaluator` — exact HPWL+density (matches PlacementCost
    to <=1e-7) plus an approximate frozen-routing congestion (~6e-3 ranking
    error, validated on CPU on commit). Cross-macro batched: one GPU dispatch
    per delta level covers all movable macros × K candidates each.
4.  After all workers finish, take the lowest-cost overlap-free result.

Why this maps to the grader machine
-----------------------------------
Grader: 16-core AMD EPYC 9655P + 100 GB RAM + NVIDIA RTX 6000 Ada 48 GB
(per `COMPETITION.md`). `TorchBatchEvaluator` auto-selects ``cuda`` on the
grader (~91 TFLOPS FP32) and ``mps`` on M-series Macs (~7.4 TFLOPS via
torch.MPS). The default `PLACER_V6_WORKERS=8` saturates 8 cores per
benchmark and leaves 8 for the OS + grader harness + GPU driver. The MLX
backend (`_mlx_eval.py`) is kept for reference but no longer in the
submission path.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Make sibling imports work whether invoked from the repo root or directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vmallela"))
sys.path.insert(0, str(_HERE.parent / "vmallela_v2"))

from macro_place.benchmark import Benchmark


def _benchmark_pt_path(bench_name: str) -> str:
    root = _HERE.parents[1]  # repo root
    p = root / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"
    if p.exists():
        return str(p)
    # Fallback: maybe Benchmark.name was already a path
    if Path(bench_name).exists():
        return str(Path(bench_name).resolve())
    return str(p)  # let Benchmark.load raise


class OptimalPlacer:
    """Multi-process portfolio with GPU-augmented hard-CD on one worker."""

    _COMPETITION_CAP_SECONDS = 3300

    def __init__(self, seed: int = 42):
        self.seed = seed
        requested = int(os.environ.get("PLACER_TOTAL_BUDGET",
                                       self._COMPETITION_CAP_SECONDS))
        self.TOTAL_TIME_LIMIT = min(requested, self._COMPETITION_CAP_SECONDS)
        self.N_WORKERS = int(os.environ.get("PLACER_V6_WORKERS", 8))
        self.GPU_WORKERS = int(os.environ.get("PLACER_V6_GPU_WORKERS", 1))
        if self.GPU_WORKERS > self.N_WORKERS:
            self.GPU_WORKERS = self.N_WORKERS

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Prefer the file-system path so the workers can re-load the
        # benchmark from scratch (subprocess).
        bench_path = _benchmark_pt_path(benchmark.name)
        # Each worker takes the full budget (they run in parallel).
        per_worker_budget = self.TOTAL_TIME_LIMIT

        # Late import to avoid pulling in mp at module load time.
        from _portfolio import run_portfolio
        log_prefix = "  "
        result_pos, best_cost, overlaps, best_seed = run_portfolio(
            bench_path,
            total_budget=per_worker_budget,
            n_workers=self.N_WORKERS,
            gpu_workers=self.GPU_WORKERS,
            base_seed=self.seed,
            log_prefix=log_prefix,
        )
        if overlaps != 0:
            # Defensive: caller validates again, but if every worker came back
            # invalid we still want to surface what we have.
            print(f"  [v6] WARNING: best result has {overlaps} overlaps "
                  f"(seed={best_seed} cost={best_cost:.6f})", flush=True)
        else:
            print(f"  [v6] DONE: cost={best_cost:.6f} seed={best_seed} "
                  f"overlaps=0", flush=True)
        return result_pos
