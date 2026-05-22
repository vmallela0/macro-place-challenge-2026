# `vmallela_v2` — shared infrastructure (legacy v4 baseline)

This directory contains the v2 / v4 baseline placer (`placer.py`), the SA / LNS
/ CD / per-net / soft-cycle move primitives, and the legacy launcher. Mean
**1.0186** on 17 IBM benchmarks.

It is **not** the active submission, but it provides Phase 1 of the v7
pipeline — the v7 placer imports `placer.py`, `_moves.py`, `_per_net.py`,
`_soft_lns.py`, `_fd_soft.py`, `_softmacro.py`, and `_surrogate.py` from
here at runtime.

A single commit on this directory in the submission branch (`91e9e87`)
fixes a recovery-path bug: when the legalize phase cannot find an
overlap-free placement within its budget, the placer now falls back to
the best partial pushed placement instead of returning the unmodified
`.plc` init. See [`../../results/RESULTS.md`](../../results/RESULTS.md)
for details and verification.

**The current submission is `submissions/vmallela_v7/`.**
See [`../vmallela_v7/README.md`](../vmallela_v7/README.md) for the algorithm,
math derivation, and grader-verified results.
