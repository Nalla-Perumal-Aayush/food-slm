"""Held-out loss / perplexity (baseline §9)."""
from __future__ import annotations

import math

import numpy as np
import torch

from food_slm.train.packing import PackedDataset


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    ds: PackedDataset,
    *,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Mean causal-LM loss (ce + z-loss) over the first ``batch_size*n_batches`` windows."""
    was_training = model.training
    model.eval()
    limit = min(len(ds), batch_size * n_batches)
    losses: list[float] = []
    for start in range(0, limit - batch_size + 1, batch_size):
        x = torch.from_numpy(ds.batch(np.arange(start, start + batch_size))).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            losses.append(model(x, labels=x).loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses)) if losses else float("nan")


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 30 else float("inf")


def perplexity_report(
    model: torch.nn.Module,
    ds: PackedDataset,
    *,
    batch_size: int = 16,
    n_batches: int = 50,
    device: torch.device | None = None,
) -> dict:
    device = device or next(model.parameters()).device
    loss = evaluate_loss(model, ds, batch_size=batch_size, n_batches=n_batches,
                         device=device, use_amp=device.type == "cuda")
    return {
        "windows_scored": min(len(ds), batch_size * n_batches),
        "loss": round(loss, 5),
        "perplexity": round(perplexity(loss), 3),
    }
