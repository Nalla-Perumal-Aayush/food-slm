"""Training loop: LR schedule, deterministic sampler, checkpoint round-trip,
and bit-exact resume (CPU, tiny model, no network)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from food_slm.model import FoodSLM, ModelConfig
from food_slm.tokenizer import TokenizerSpec, load_tokenizer, train_tokenizer
from food_slm.train import (
    BatchSampler,
    load_checkpoint,
    lr_at,
    pack_documents,
    save_checkpoint,
    train,
)

_SPEC_KW = dict(
    byte_level=True, algorithm="bpe", lowercase=False, nfkc_normalize=True,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    structural_tokens=["<DISH>"], chat_tokens=["<|user|>"],
)
_TEXT = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu.\n\n"
         "quick brown fox over lazy dog while five boxing wizards jump.\n\n") * 400


# -- LR schedule ----------------------------------------------------------

def test_lr_schedule_warmup_peak_and_floor() -> None:
    kw = dict(peak=1.0, warmup=10, total=100, min_ratio=0.1)
    assert lr_at(0, **kw) == pytest.approx(0.1)
    assert lr_at(9, **kw) == pytest.approx(1.0)
    assert lr_at(10, **kw) == pytest.approx(1.0)            # first post-warmup step ~ peak
    assert lr_at(55, **kw) == pytest.approx(0.55, abs=0.05)  # mid cosine
    assert lr_at(100, **kw) == pytest.approx(0.1)           # floor
    assert lr_at(200, **kw) == pytest.approx(0.1)


# -- deterministic / resumable sampler ---------------------------------

def test_batch_sampler_is_position_independent_and_shuffles() -> None:
    full = [b.copy() for _, b in zip(range(12), BatchSampler(50, 4, seed=1), strict=False)]
    mid = [b.copy() for _, b in zip(range(7), BatchSampler(50, 4, seed=1, start_batch=5), strict=False)]
    for a, b in zip(full[5:12], mid, strict=False):
        assert np.array_equal(a, b)
    # different seed -> different order
    other = next(iter(BatchSampler(50, 4, seed=2)))
    assert not np.array_equal(full[0], other)


# -- checkpoint round-trip -------------------------------------------

def test_checkpoint_restores_model_and_optimizer(tmp_path: Path) -> None:
    cfg = ModelConfig.from_yaml(Path("configs/model_15m.yaml")).model_copy(
        update={"hidden_dim": 64, "num_layers": 2, "num_heads": 2, "num_kv_heads": 2,
                "ffn_inner_dim": 128, "vocab_size": 300, "max_seq_len": 32})
    m1 = FoodSLM(cfg)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    x = torch.randint(0, 300, (2, 16))
    m1(x, labels=x).loss.backward()
    opt1.step()
    save_checkpoint(tmp_path / "c.pt", model=m1, optimizer=opt1, step=1)

    m2 = FoodSLM(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    info = load_checkpoint(tmp_path / "c.pt", model=m2, optimizer=opt2)
    assert info["step"] == 1
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=False):
        assert torch.equal(p1, p2)
    assert opt2.state_dict()["state"].keys() == opt1.state_dict()["state"].keys()


# -- end-to-end tiny run + bit-exact resume --------------------------

@pytest.fixture(scope="module")
def poc_env(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("train")
    (d / "c.txt").write_text(_TEXT, encoding="utf-8")
    tokdir = train_tokenizer([d / "c.txt"], TokenizerSpec(vocab_size=320, **_SPEC_KW),
                             d / "tok", min_frequency=1)
    tok = load_tokenizer(tokdir)
    eos = tok.token_to_id("<|eos|>")
    pack_documents(_TEXT.split("\n\n"), tok, 32, d / "shard", eos_id=eos, val_fraction=0.2)

    (d / "model.yaml").write_text(json.dumps({
        "name": "tiny", "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "num_kv_heads": 2,
        "ffn_inner_dim": 128, "vocab_size": tok.get_vocab_size(), "max_seq_len": 32,
    }), encoding="utf-8")
    return d


def _train_cfg(d: Path, max_steps: int, **over) -> Path:
    cfg = {
        "model_config": str(d / "model.yaml"),
        "shard_dir": str(d / "shard"),
        "checkpoint_dir": str(d / over.pop("ckpt", "ck")),
        "seed": 1337, "sequence_length": 32,
        "lr": 3.0e-3, "warmup_ratio": 0.1, "micro_batch_size": 4, "grad_accum_steps": 2,
        "max_steps": max_steps, "checkpoint_every_steps": 4, "eval_every_steps": 4,
        "eval_batches": 3, "log_every_steps": 100, "tracker": "none",
        **over,
    }
    p = d / f"train_{max_steps}_{cfg['checkpoint_dir'][-4:]}.yaml"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_short_run_learns_and_writes_checkpoints(poc_env: Path) -> None:
    res = train(_train_cfg(poc_env, 12, ckpt="ck_a"), repo_root=poc_env)
    assert res.final_step == 12
    assert res.history[-1]["loss"] < res.history[0]["loss"]
    assert res.eval_history and res.eval_history[-1]["val_loss"] > 0
    assert (poc_env / "ck_a" / "step_12.pt").exists()
    assert (poc_env / "ck_a" / "best.pt").exists()


def test_checkpoint_at_stop_is_exact(poc_env: Path) -> None:
    """Two identical runs stopped at step 4 produce byte-identical model weights,
    and a fresh model loading that checkpoint matches bit-for-bit."""
    train(_train_cfg(poc_env, 8, ckpt="ck_e1"), repo_root=poc_env, stop_after=4)
    train(_train_cfg(poc_env, 8, ckpt="ck_e2"), repo_root=poc_env, stop_after=4)
    a = load_checkpoint  # alias for brevity below

    cfg = ModelConfig.from_yaml(poc_env / "model.yaml")
    m1, m2 = FoodSLM(cfg), FoodSLM(cfg)
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    a(poc_env / "ck_e1" / "step_4.pt", model=m1, optimizer=o1)
    a(poc_env / "ck_e2" / "step_4.pt", model=m2, optimizer=o2)
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=False):
        assert torch.equal(p1, p2)


def test_resume_bit_exact_in_deterministic_mode(poc_env: Path) -> None:
    """With deterministic=True on CPU the resumed loss trajectory is bit-identical."""
    kw = dict(repo_root=poc_env, deterministic=True, device="cpu")
    straight = train(_train_cfg(poc_env, 8, ckpt="ck_ds"), **kw)
    train(_train_cfg(poc_env, 8, ckpt="ck_dr"), stop_after=4, **kw)
    resumed = train(_train_cfg(poc_env, 8, ckpt="ck_dr"), resume="auto", **kw)
    tail = [h["loss"] for h in straight.history if h["step"] >= 4]
    assert [h["loss"] for h in resumed.history] == tail


def test_curriculum_run_over_sharded_pack(tmp_path_factory) -> None:
    """A short run over a multi-source sharded pack with a 1K->2K->4K-style curriculum:
    it runs end to end, logs per-source val, writes metrics.jsonl, and deterministic
    resume stays bit-exact across a phase boundary."""
    from food_slm.train.shards import ShardSource, pack_sharded

    d = tmp_path_factory.mktemp("cur")
    (d / "c.txt").write_text(_TEXT, encoding="utf-8")
    tokdir = train_tokenizer([d / "c.txt"], TokenizerSpec(vocab_size=320, **_SPEC_KW),
                             d / "tok", min_frequency=1)
    gen = d / "gen.txt"
    dom = d / "dom.txt"
    gen.write_text("\n\n".join(f"general document {i} plain words here now then" * 4
                               for i in range(300)), encoding="utf-8")
    dom.write_text("\n\n".join(f"<DISH>Dish {i}</DISH> tiffin Tamil Nadu vegetarian"
                               for i in range(40)), encoding="utf-8")
    pack_sharded([ShardSource("gen", [gen], 1), ShardSource("dom", [dom], 6)],
                 tokdir, seq_len=32, out_dir=d / "shard",
                 val_docs_per_source=8, shard_windows=60, seed=1337)
    (d / "model.yaml").write_text(json.dumps({
        "name": "tiny", "hidden_dim": 64, "num_layers": 2, "num_heads": 2, "num_kv_heads": 2,
        "ffn_inner_dim": 128, "vocab_size": 320, "max_seq_len": 32,
    }), encoding="utf-8")

    def cfg(ckpt: str) -> Path:
        p = d / f"train_{ckpt}.yaml"
        p.write_text(json.dumps({
            "model_config": str(d / "model.yaml"), "shard_dir": str(d / "shard"),
            "checkpoint_dir": str(d / ckpt), "seed": 1337, "sequence_length": 32,
            "lr": 3.0e-3, "warmup_ratio": 0.1, "micro_batch_size": 4, "grad_accum_steps": 2,
            "max_steps": 8, "checkpoint_every_steps": 4, "eval_every_steps": 4,
            "eval_batches": 2, "log_every_steps": 100, "tracker": "none",
            "curriculum": [[16, 0.5], [32, 1.0]],
        }), encoding="utf-8")
        return p

    kw = dict(repo_root=d, deterministic=True, device="cpu")
    straight = train(cfg("ck_cs"), **kw)
    assert straight.final_step == 8
    assert straight.history[3]["seq_len"] == 16 and straight.history[4]["seq_len"] == 32
    assert "val_gen" in straight.eval_history[-1] and "val_dom" in straight.eval_history[-1]
    assert (d / "ck_cs" / "metrics.jsonl").exists()

    train(cfg("ck_cr"), stop_after=4, **kw)
    resumed = train(cfg("ck_cr"), resume="auto", **kw)
    tail = [h["loss"] for h in straight.history if h["step"] >= 4]
    assert [h["loss"] for h in resumed.history] == tail


def test_resume_does_not_diverge(poc_env: Path) -> None:
    """Resume continues the same trajectory: the deterministic (no-dropout) val-loss
    matches to float noise, and the training-loss tail stays within noise."""
    straight = train(_train_cfg(poc_env, 8, ckpt="ck_s"), repo_root=poc_env)
    train(_train_cfg(poc_env, 8, ckpt="ck_r"), repo_root=poc_env, stop_after=4)
    resumed = train(_train_cfg(poc_env, 8, ckpt="ck_r"), repo_root=poc_env, resume="auto")

    assert resumed.final_step == 8
    # val loss is computed in eval mode (no dropout) -> essentially deterministic
    sv = {e["step"]: e["val_loss"] for e in straight.eval_history}
    rv = {e["step"]: e["val_loss"] for e in resumed.eval_history}
    assert rv[8] == pytest.approx(sv[8], abs=5e-3)
    # training-loss tail: no divergence
    tail_s = [h["loss"] for h in straight.history if h["step"] >= 4]
    tail_r = [h["loss"] for h in resumed.history]
    for a_, b_ in zip(tail_s, tail_r, strict=False):
        assert abs(a_ - b_) < 2e-2
