"""Self-contained copy of the South Indian Cuisine SLM's architecture, with zero
dependency on the rest of the food_slm project -- just torch + pyyaml. This is
NOT a HuggingFace or GGUF model: it's a from-scratch decoder-only Transformer
with a custom (but small, standard-building-blocks) implementation, so it needs
this actual code to load the weights, not just a weights file.

Architecture (frozen/"locked" in the source project, unchanged since training):
pre-norm RMSNorm + QK-norm, RoPE, SwiGLU FFN, standard multi-head attention via
PyTorch's scaled_dot_product_attention (causal), tied input/output embeddings,
151,225,600 total parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    ffn_inner_dim: int
    vocab_size: int = 16384
    rope_theta: float = 10000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0           # 0 for inference regardless of the training-time value
    attention_dropout: float = 0.0
    max_seq_len: int = 4096
    z_loss_weight: float = 1.0e-4  # unused at inference (no labels passed), kept for parity

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**keep)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def _rope_tables(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv
        self.attn_dropout = cfg.attention_dropout
        d = cfg.hidden_dim
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rope(q, k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        o = o.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.hidden_dim, cfg.ffn_inner_dim
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class FoodSLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.final_norm = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = _rope_tables(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
                                device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        if s > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {s} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(input_ids)
        cos = self.rope_cos[:s].to(x.dtype)
        sin = self.rope_sin[:s].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)


def load_model(config_path: str | Path, weights_path: str | Path, device: str = "cpu") -> tuple[FoodSLM, ModelConfig]:
    """Weights file is the fp16-stripped state_dict (model.load_state_dict's
    keys only, no optimizer/rng/step) -- cast to fp32 for CPU inference, since
    most ARM CPUs (including the Raspberry Pi 5's Cortex-A76) have no fast
    fp16 matmul path the way a GPU does; fp16 here is purely for file size on
    disk/transfer, not a speed optimization for this hardware."""
    cfg = ModelConfig.from_yaml(config_path)
    model = FoodSLM(cfg).to(device)
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model, cfg
