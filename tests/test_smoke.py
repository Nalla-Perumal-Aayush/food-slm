"""Scaffold smoke tests — no GPU, no data required."""
from __future__ import annotations

from pathlib import Path

import pytest

from food_slm.model.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


_LADDER = [
    "configs/model_15m.yaml",
    "configs/model_40m.yaml",
    "configs/model_89m.yaml",
    "configs/model_150m.yaml",
]


@pytest.mark.parametrize(
    ("cfg_path", "lo", "hi"),
    [
        ("configs/model_15m.yaml", 14_000_000, 17_000_000),
        # 40M rung redefined d=512/L=14 -> ~53M (Rev 8: depth de-risk, 89M skipped if clean)
        ("configs/model_40m.yaml", 50_000_000, 56_000_000),
        ("configs/model_89m.yaml", 86_000_000, 93_000_000),
        # Config B — release target is strictly 150M, accepted tolerance +/- 5%
        ("configs/model_150m.yaml", 142_500_000, 157_500_000),
    ],
)
def test_param_counts_in_range(cfg_path: str, lo: int, hi: int) -> None:
    cfg = ModelConfig.from_yaml(ROOT / cfg_path)
    total = cfg.param_count()["total"]
    assert lo <= total <= hi, f"{cfg.name}: {total:,}"


def test_release_config_within_5pct_of_150m() -> None:
    total = ModelConfig.from_yaml(ROOT / "configs/model_150m.yaml").param_count()["total"]
    assert abs(total / 150_000_000 - 1.0) <= 0.05, f"{total:,} is >5% off 150M"


def test_head_dim_consistent() -> None:
    for p in _LADDER:
        cfg = ModelConfig.from_yaml(ROOT / p)
        assert cfg.hidden_dim % cfg.num_heads == 0
        assert cfg.head_dim == 64


def test_schema_imports() -> None:
    from food_slm.schema import Dish, Ingredient, Recipe, Source  # noqa: F401
