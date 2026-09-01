"""Config A decoder-only Transformer.

TODO (first code milestone, after the schema/tokenizer artifacts):
  - token embedding (tied to LM head)
  - RoPE cache
  - RMSNorm + optional QK-norm
  - MHA via torch.nn.functional.scaled_dot_product_attention  (no flash-attn on Windows)
  - SwiGLU FFN
  - pre-norm residual blocks x num_layers
  - scaled-normal init (std 0.02, residual proj / sqrt(2L))
  - forward() returning logits + optional z-loss term
"""
from __future__ import annotations

import torch
import torch.nn as nn

from food_slm.model.config import ModelConfig


class FoodSLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "FoodSLM forward pass not implemented yet — this is the first code "
            "milestone. Scaffold only."
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
