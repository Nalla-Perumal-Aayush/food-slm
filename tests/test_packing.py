"""Sequence packing + PackedDataset (small fixture, no network)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from food_slm.tokenizer import TokenizerSpec, load_tokenizer, train_tokenizer
from food_slm.train import PackedDataset, iter_text_documents, pack_documents
from food_slm.train.packing import DTYPE

_SPEC_KW = dict(
    byte_level=True, algorithm="bpe", lowercase=False, nfkc_normalize=True,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    structural_tokens=["<DISH>", "<STEP>"], chat_tokens=["<|user|>"],
)
_TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi.\n\n"
    "one two three four five six seven eight nine ten eleven twelve thirteen.\n\n"
) * 200


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("pk")
    (d / "c.txt").write_text(_TEXT, encoding="utf-8")
    train_tokenizer([d / "c.txt"], TokenizerSpec(vocab_size=400, **_SPEC_KW),
                    d / "tk", min_frequency=1)
    return load_tokenizer(d / "tk")


def _eos(tok) -> int:
    return tok.token_to_id("<|eos|>")


def test_iter_text_documents_splits_on_blank_lines_and_skips_comments(tmp_path: Path) -> None:
    (tmp_path / "c.txt").write_text("# comment\ndoc one line a\ndoc one line b\n\ndoc two\n",
                                    encoding="utf-8")
    docs = list(iter_text_documents(tmp_path / "c.txt"))
    assert docs == ["doc one line a\ndoc one line b", "doc two"]


def test_pack_window_shape_and_dtype(tok, tmp_path: Path) -> None:
    man = pack_documents(_TEXT.split("\n\n"), tok, 64, tmp_path / "s", eos_id=_eos(tok))
    assert man.seq_len == 64
    assert man.n_windows > 0
    assert man.dtype == "uint16"
    data = np.fromfile(tmp_path / "s" / "tokens.bin", dtype=DTYPE)
    assert data.size == man.n_windows * 64
    assert man.dropped_tail_tokens < 64


def test_eos_present_and_token_budget_conserved(tok, tmp_path: Path) -> None:
    docs = [d for d in _TEXT.split("\n\n") if d.strip()]
    man = pack_documents(docs, tok, 64, tmp_path / "s", eos_id=_eos(tok))
    data = np.fromfile(tmp_path / "s" / "tokens.bin", dtype=DTYPE)
    assert (data == _eos(tok)).sum() >= 1
    # packed + dropped == raw (every raw token is either packed or in the dropped tail)
    assert man.n_tokens_packed + man.dropped_tail_tokens == man.n_tokens_raw


def test_add_bos_per_window(tok, tmp_path: Path) -> None:
    bos = tok.token_to_id("<|bos|>")
    man = pack_documents(_TEXT.split("\n\n"), tok, 32, tmp_path / "s",
                         eos_id=_eos(tok), bos_id=bos, add_bos_per_window=True)
    rows = np.fromfile(tmp_path / "s" / "tokens.bin", dtype=DTYPE).reshape(man.n_windows, 32)
    assert (rows[:, 0] == bos).all()


def test_val_split_and_packed_dataset(tok, tmp_path: Path) -> None:
    man = pack_documents(_TEXT.split("\n\n"), tok, 48, tmp_path / "s",
                         eos_id=_eos(tok), val_fraction=0.25)
    assert man.val_windows > 0
    assert (tmp_path / "s" / "val.bin").exists()

    train = PackedDataset(tmp_path / "s", "train")
    val = PackedDataset(tmp_path / "s", "val")
    assert len(train) == man.n_windows
    assert len(val) == man.val_windows
    x, y = train[0]
    assert x.shape == (48,) and np.array_equal(x, y) and x.dtype == np.int64


def test_packing_is_deterministic(tok, tmp_path: Path) -> None:
    a = pack_documents(_TEXT.split("\n\n"), tok, 40, tmp_path / "a", eos_id=_eos(tok))
    b = pack_documents(_TEXT.split("\n\n"), tok, 40, tmp_path / "b", eos_id=_eos(tok))
    assert a.n_windows == b.n_windows
    assert (tmp_path / "a" / "tokens.bin").read_bytes() == (tmp_path / "b" / "tokens.bin").read_bytes()
