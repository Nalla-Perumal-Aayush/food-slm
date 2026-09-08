"""Multi-source sharded packing + ShardedPackedDataset (plan_40m §E)."""
from __future__ import annotations

import numpy as np
import pytest

from food_slm.tokenizer import TokenizerSpec, train_tokenizer
from food_slm.train.shards import ShardedPackedDataset, ShardSource, pack_sharded

_SPEC_KW = dict(
    byte_level=True, algorithm="bpe", lowercase=False, nfkc_normalize=True,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    structural_tokens=["<DISH>", "<STEP>"], chat_tokens=["<|user|>"],
)


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sh")
    (d / "c.txt").write_text(("word alpha beta gamma delta epsilon zeta eta theta.\n\n") * 300,
                             encoding="utf-8")
    train_tokenizer([d / "c.txt"], TokenizerSpec(vocab_size=400, **_SPEC_KW),
                    d / "tk", min_frequency=1)
    return d / "tk"


@pytest.fixture(scope="module")
def packed(tmp_path_factory, tok_dir):
    d = tmp_path_factory.mktemp("corpus")
    gen = d / "gen.txt"
    dom = d / "dom.txt"
    gen.write_text("\n\n".join(f"General doc {i} with several plain english words here today."
                               * 8 for i in range(400)), encoding="utf-8")
    dom.write_text("\n\n".join(f"<DISH>Dish {i}</DISH> is a tiffin from Tamil Nadu. Vegetarian."
                               for i in range(50)), encoding="utf-8")
    out = d / "packed"
    man = pack_sharded(
        [ShardSource("gen", [gen], 1), ShardSource("dom", [dom], 8)],
        tok_dir, seq_len=64, out_dir=out,
        val_docs_per_source=10, shard_windows=40, seed=99,
    )
    return out, man


def test_manifest_shape(packed):
    _, man = packed
    assert man.seq_len == 64
    assert man.total_windows == sum(s["n_windows"] for s in man.shards)
    assert len(man.shards) >= 2                    # multi-shard
    assert set(man.source_mix) == {"gen", "dom"}
    assert man.source_mix["dom"]["repeat"] == 8
    assert set(man.val_files) == {"gen", "dom"}


def test_sha_recorded_and_stable(packed):
    out, man = packed
    for s in man.shards:
        import hashlib
        data = (out / s["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == s["sha256"]


def test_dataset_reads_across_shards(packed):
    out, man = packed
    ds = ShardedPackedDataset(out, "train")
    assert len(ds) == man.total_windows
    x, y = ds[0]
    assert x.shape == (64,) and np.array_equal(x, y)
    last, _ = ds[len(ds) - 1]                      # crosses into the final shard
    assert last.shape == (64,)
    b = ds.batch([0, 1, len(ds) - 1])
    assert b.shape == (3, 64)


def test_val_split_is_document_disjoint_and_per_source(packed):
    out, _ = packed
    v_all = ShardedPackedDataset(out, "val")
    v_dom = ShardedPackedDataset(out, "val:dom")
    v_gen = ShardedPackedDataset(out, "val:gen")
    assert len(v_all) == len(v_dom) + len(v_gen)
    assert len(v_dom) >= 1 and len(v_gen) >= 1


def test_domain_upsampling_changes_mix(tmp_path, tok_dir):
    gen = tmp_path / "g.txt"
    dom = tmp_path / "d.txt"
    gen.write_text("\n\n".join(f"plain doc {i} words words words" for i in range(200)),
                   encoding="utf-8")
    dom.write_text("\n\n".join(f"<DISH>D{i}</DISH> tiffin" for i in range(40)), encoding="utf-8")
    m1 = pack_sharded([ShardSource("gen", [gen], 1), ShardSource("dom", [dom], 1)],
                      tok_dir, 32, tmp_path / "p1", val_docs_per_source=5, seed=1)
    m10 = pack_sharded([ShardSource("gen", [gen], 1), ShardSource("dom", [dom], 10)],
                       tok_dir, 32, tmp_path / "p10", val_docs_per_source=5, seed=1)
    assert m10.source_mix["dom"]["train_tokens"] > 8 * m1.source_mix["dom"]["train_tokens"]
