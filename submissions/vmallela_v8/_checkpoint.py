"""Phase checkpointing for v8. Resume-on-restart support.

API:
    save_phase(name, state, gate_result) — pickle state + metadata to
        checkpoints/<name>.pkl.
    load_phase(name) -> (state, gate_result) | None — restore. None if
        no checkpoint or it was marked failed.
    is_phase_passed(name) -> bool — quick check; if True, skip the phase.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
_CKPT_DIR.mkdir(exist_ok=True)


def _ckpt_path(name: str) -> Path:
    return _CKPT_DIR / f"{name}.pkl"


def save_phase(name: str, state: Any, gate_result: dict) -> None:
    """Persist a phase's output and gate result.

    state: arbitrary picklable. Typically the placement np.ndarray.
    gate_result: dict with at least {"passed": bool, "proxy": float, ...}.
    """
    payload = {"state": state, "gate_result": gate_result}
    with _ckpt_path(name).open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_phase(name: str) -> tuple[Any, dict] | None:
    p = _ckpt_path(name)
    if not p.exists():
        return None
    with p.open("rb") as f:
        payload = pickle.load(f)
    return payload["state"], payload["gate_result"]


def is_phase_passed(name: str) -> bool:
    rec = load_phase(name)
    if rec is None:
        return False
    _, gate = rec
    return bool(gate.get("passed", False))
