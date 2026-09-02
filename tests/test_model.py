"""FoodSLM transformer — shape, param-count, and learning-signal tests (CPU, no data)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from food_slm.model import FoodSLM, ModelConfig
from food_slm.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name: str = "configs/model_15m.yaml", **update) -> ModelConfig:
    cfg = ModelConfig.from_yaml(ROOT / name)
    return cfg.model_copy(update=update) if update else cfg


@pytest.mark.parametrize("name", [
    "configs/model_15m.yaml", "configs/model_40m.yaml", "configs/model_150m.yaml",
])
def test_param_count_matches_analytic(name: str) -> None:
    cfg = _cfg(name)
    model = FoodSLM(cfg)
    assert model.num_params() == cfg.param_count()["total"]
    assert model.num_params(non_embedding=True) == cfg.param_count()["non_embedding"]


def test_tie_embeddings_shares_storage() -> None:
    model = FoodSLM(_cfg())
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_forward_shapes_and_finite_loss() -> None:
    set_seed(0, deterministic=False)
    cfg = _cfg()
    model = FoodSLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(ids, labels=ids)
    assert tuple(out.logits.shape) == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert out.z_loss.item() >= 0
    # logits-only path
    assert model(ids).loss is None


def test_backward_produces_finite_grads_everywhere() -> None:
    set_seed(0, deterministic=False)
    cfg = _cfg()
    model = FoodSLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids, labels=ids).loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n


def test_rejects_sequence_longer_than_max() -> None:
    cfg = _cfg(max_seq_len=8)
    model = FoodSLM(cfg)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_rope_handles_varied_sequence_lengths() -> None:
    cfg = _cfg(max_seq_len=64)
    model = FoodSLM(cfg).eval()
    for s in (1, 7, 64):
        out = model(torch.randint(0, cfg.vocab_size, (1, s)))
        assert out.logits.shape[1] == s


def test_grad_checkpointing_gives_same_loss() -> None:
    cfg = _cfg(dropout=0.0, attention_dropout=0.0)
    set_seed(3, deterministic=False)
    model = FoodSLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 48))
    plain = model(ids, labels=ids).loss.item()
    model.grad_checkpoint = True
    ckpt = model(ids, labels=ids).loss.item()
    assert abs(plain - ckpt) < 1e-4


def test_overfits_a_single_batch() -> None:
    cfg = _cfg(dropout=0.0, attention_dropout=0.0)
    set_seed(1337, deterministic=False)
    model = FoodSLM(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    first = last = None
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < 0.6 * first, f"{first:.3f} -> {last:.3f}"
