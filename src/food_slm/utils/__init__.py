"""Shared helpers: config loading, seeding, logging."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed python / numpy / torch. Required for reproducibility (baseline §7)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch optional for pure-data tooling
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
