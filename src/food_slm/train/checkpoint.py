"""Bit-exact checkpoint / resume (baseline §7).

A checkpoint carries everything needed to continue training such that the loss
curve after resume is identical to an uninterrupted run: model + optimizer state,
the optimizer-step counter, and the python / numpy / torch RNG states. The data
order is a pure function of ``(seed, step)`` via :class:`BatchSampler`, and the LR
is a pure function of the step, so nothing else needs to be stored.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BatchSampler:
    """Deterministic, resumable micro-batch index stream over ``n_windows`` windows.

    Batch *b* (0-indexed, global) is the same array of indices for a given ``seed``
    regardless of where iteration started — so resume just sets ``start_batch``.
    Partial final batches of each epoch are dropped for constant shapes.
    """

    n_windows: int
    batch_size: int
    seed: int
    start_batch: int = 0

    def __post_init__(self) -> None:
        self.per_epoch = self.n_windows // self.batch_size
        if self.per_epoch == 0:
            raise ValueError(f"batch_size {self.batch_size} > n_windows {self.n_windows}")

    def _epoch_perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.seed + epoch).permutation(self.n_windows)

    def __iter__(self):
        b = self.start_batch
        while True:
            epoch, within = divmod(b, self.per_epoch)
            perm = self._epoch_perm(epoch)
            for k in range(within, self.per_epoch):
                yield perm[k * self.batch_size:(k + 1) * self.batch_size]
                b += 1


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu() if torch.is_tensor(s["torch"]) else s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        with contextlib.suppress(Exception):  # device-count mismatch etc.
            torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float | None = None,
    config: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
            "config": config or {},
            "format": 1,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore_rng and ckpt.get("rng"):
        _set_rng_state(ckpt["rng"])
    return {"step": ckpt["step"], "best_val": ckpt.get("best_val"), "config": ckpt.get("config", {})}


def prune_checkpoints(ckpt_dir: str | Path, keep_last_n: int, *, keep_names=("best.pt",)) -> None:
    d = Path(ckpt_dir)
    steps = sorted(
        (p for p in d.glob("step_*.pt")),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for p in steps[:-keep_last_n] if keep_last_n > 0 else []:
        if p.name not in keep_names:
            p.unlink()
