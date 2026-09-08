"""Context-length curriculum schedule (plan_40m §F)."""
from __future__ import annotations

import pytest

from food_slm.train.curriculum import (
    parse_curriculum,
    phase_boundaries,
    seq_len_at,
)


def test_none_is_single_full_phase():
    ph = parse_curriculum(None, 4096)
    assert len(ph) == 1
    assert seq_len_at(0, 100, ph) == 4096
    assert seq_len_at(99, 100, ph) == 4096


def test_three_phase_schedule():
    ph = parse_curriculum([[1024, 0.4], [2048, 0.7], [4096, 1.0]], 4096)
    assert seq_len_at(0, 1000, ph) == 1024
    assert seq_len_at(399, 1000, ph) == 1024
    assert seq_len_at(400, 1000, ph) == 2048
    assert seq_len_at(699, 1000, ph) == 2048
    assert seq_len_at(700, 1000, ph) == 4096
    assert seq_len_at(999, 1000, ph) == 4096
    assert seq_len_at(10_000, 1000, ph) == 4096  # past the end


def test_phase_boundaries():
    ph = parse_curriculum([[1024, 0.5], [2048, 1.0]], 2048)
    assert phase_boundaries(1000, ph) == [0, 500]


@pytest.mark.parametrize("spec", [
    [[1024, 0.4], [2048, 0.4], [4096, 1.0]],   # non-increasing fractions
    [[1024, 0.4], [2048, 0.7], [4096, 0.9]],   # last != 1.0
    [[2048, 0.4], [1024, 1.0]],                # decreasing seq_len
    [[8192, 1.0]],                             # exceeds packed max
])
def test_invalid_specs_rejected(spec):
    with pytest.raises(ValueError):
        parse_curriculum(spec, 4096)
