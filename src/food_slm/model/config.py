"""Typed model config + parameter-count accounting."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from food_slm.utils import load_yaml


class ModelConfig(BaseModel):
    name: str
    arch: str = "decoder_only_transformer"

    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    ffn_inner_dim: int
    activation: str = "silu"
    ffn_type: str = "swiglu"

    positional: str = "rope"
    rope_theta: float = 10000.0

    norm: str = "rmsnorm"
    norm_position: str = "pre"
    qk_norm: bool = True
    z_loss_weight: float = 1.0e-4

    tie_embeddings: bool = True
    vocab_size: int = 16384

    dropout: float = 0.1
    attention_dropout: float = 0.0

    max_seq_len: int = 4096
    init: str = "scaled_normal"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        return cls(**load_yaml(path))

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0
        return self.hidden_dim // self.num_heads

    def param_count(self) -> dict[str, int]:
        """Analytic parameter count — matches ``FoodSLM.num_params`` within rounding."""
        d, L, v = self.hidden_dim, self.num_layers, self.vocab_size
        h = self.ffn_inner_dim
        attn = 4 * d * d                              # q, k, v, o  (MHA, no bias)
        ffn = 3 * d * h                               # gate, up, down (SwiGLU, no bias)
        norms = 2 * d                                 # 2 pre-norm RMSNorm gains
        qk = 2 * self.head_dim if self.qk_norm else 0  # per-head q/k RMSNorm gains
        per_layer = attn + ffn + norms + qk
        non_embed = L * per_layer + d                 # + final norm
        embed = v * d                                 # tied -> counted once
        return {
            "non_embedding": non_embed,
            "embedding": embed,
            "total": non_embed + embed,
        }
