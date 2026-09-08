"""Context-length curriculum: 1K -> 2K -> 4K (baseline §5.3 / plan_40m §F).

The corpus is packed once at the maximum ``seq_len``. Each curriculum phase feeds the
first ``phase.seq_len`` tokens of every packed window, so no repacking is needed and the
window index stream (``BatchSampler``) is unchanged — resume stays bit-exact because the
phase is a pure function of the global step.

The LR schedule is deliberately *not* phase-aware: it stays a pure function of the global
step (continuous cosine), so a resume mid-phase reproduces the same curve.

Config form (``TrainConfig.curriculum``): a list of ``[seq_len, until_fraction]`` pairs,
fractions strictly increasing and ending at 1.0, e.g.::

    curriculum: [[1024, 0.40], [2048, 0.70], [4096, 1.0]]
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    seq_len: int
    until_frac: float


def parse_curriculum(spec, max_seq_len: int) -> list[Phase]:
    """Validate a curriculum spec against the packed ``max_seq_len``. ``None`` -> a single
    full-length phase."""
    if not spec:
        return [Phase(max_seq_len, 1.0)]
    phases = [Phase(int(sl), float(f)) for sl, f in spec]
    if any(p.seq_len > max_seq_len for p in phases):
        raise ValueError(f"curriculum seq_len exceeds packed max_seq_len {max_seq_len}")
    if [p.seq_len for p in phases] != sorted(p.seq_len for p in phases):
        raise ValueError("curriculum seq_len must be non-decreasing")
    fracs = [p.until_frac for p in phases]
    if fracs != sorted(fracs) or len(set(fracs)) != len(fracs):
        raise ValueError("curriculum fractions must be strictly increasing")
    if abs(fracs[-1] - 1.0) > 1e-9:
        raise ValueError("last curriculum fraction must be 1.0")
    return phases


def phase_at(step: int, max_steps: int, phases: list[Phase]) -> Phase:
    frac = step / max(1, max_steps)
    for p in phases:
        if frac < p.until_frac:
            return p
    return phases[-1]


def seq_len_at(step: int, max_steps: int, phases: list[Phase]) -> int:
    return phase_at(step, max_steps, phases).seq_len


def phase_boundaries(max_steps: int, phases: list[Phase]) -> list[int]:
    """Global steps at which the phase changes (for logging)."""
    out, prev = [], None
    for s in range(max_steps):
        sl = seq_len_at(s, max_steps, phases)
        if sl != prev:
            out.append(s)
            prev = sl
    return out
