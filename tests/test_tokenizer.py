"""Tokenizer trainer + analyzer + canonical serialization (small fixture, no network)."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from food_slm.data.serialize import iter_documents, store_to_text
from food_slm.schema import load_store
from food_slm.tokenizer import TokenizerSpec, analyze_tokenizer, load_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"

_SPEC_KW = dict(
    byte_level=True, algorithm="bpe", lowercase=False, nfkc_normalize=True,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    structural_tokens=["<DISH>", "<INGREDIENT>", "<STEP>", "<QTY>", "<TIME>"],
    chat_tokens=["<|user|>", "<|assistant|>"],
)

_FIXTURE = (
    "The quick brown fox jumps over the lazy dog. " * 40
    + "\n\nVen Pongal (ven pongal, வெண் பொங்கல்) is a soft rice and dal tiffin. " * 20
    + "\n\n<DISH>Ven Pongal</DISH> <STEP>1. Dry-roast the dal.</STEP> "
    + "<INGREDIENT><QTY>1 cup</QTY> raw rice</INGREDIENT>. " * 20
    + "\n\nPuttu (puttu, പുട്ട്) steamed rice cake from Kerala. " * 15
)


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "corpus.txt"
    corpus.write_text(_FIXTURE, encoding="utf-8")
    spec = TokenizerSpec(vocab_size=800, **_SPEC_KW)
    return train_tokenizer([corpus], spec, d / "poc", min_frequency=1)


def test_low_ids_are_the_special_tokens(tok_dir: Path) -> None:
    tok = load_tokenizer(tok_dir)
    assert tok.token_to_id("<|pad|>") == 0
    assert tok.token_to_id("<|bos|>") == 1
    assert tok.token_to_id("<|eos|>") == 2
    assert tok.token_to_id("<|unk|>") == 3


def test_structural_and_chat_tokens_are_atomic(tok_dir: Path) -> None:
    tok = load_tokenizer(tok_dir)
    for t in ("<DISH>", "<INGREDIENT>", "<STEP>", "<QTY>", "<|user|>", "<|assistant|>"):
        ids = tok.encode(t).ids
        assert ids == [tok.token_to_id(t)], f"{t} -> {ids} (not atomic)"


@pytest.mark.parametrize("text", [
    "Plain ASCII with digits 12345 and punctuation: (a/b) 1.5 tsp!",
    "வெண் பொங்கல் — a Tamil dish name in native script",
    "പുട്ട് and ಬಿಸಿ ಬೇಳೆ ಬಾತ್ and పప్పు — Malayalam, Kannada, Telugu",
    "emoji-free but weird: café résumé — dashes … ellipsis",
])
def test_byte_level_round_trip_no_loss(tok_dir: Path, text: str) -> None:
    tok = load_tokenizer(tok_dir)
    norm = unicodedata.normalize("NFKC", text)
    assert tok.decode(tok.encode(norm).ids) == norm


def test_no_unk_token_in_model(tok_dir: Path) -> None:
    # <|unk|> exists as a reserved *special* token but is never emitted by encoding
    tok = load_tokenizer(tok_dir)
    unk_id = tok.token_to_id("<|unk|>")
    for s in ("\x00\x01\x02 raw control bytes", "ﷺ ﷽ rare arabic ligatures", "🍚🌶️"):
        assert unk_id not in tok.encode(unicodedata.normalize("NFKC", s)).ids


def test_actual_vocab_within_target(tok_dir: Path) -> None:
    tok = load_tokenizer(tok_dir)
    assert tok.get_vocab_size() <= 800


def test_analyze_report_shape(tok_dir: Path) -> None:
    tok = load_tokenizer(tok_dir)
    rep = analyze_tokenizer(
        tok,
        [b for b in _FIXTURE.split("\n\n") if b.strip()],
        dish_names=["Ven Pongal", "Puttu"],
        native_names=["வெண் பொங்கல்", "പുട്ട്"],
    )
    assert rep["token_fertility"] > 0
    assert rep["compression_ratio"] > 1.0
    assert 0.0 < rep["vocab_utilization"] <= 1.0
    assert rep["culinary_term_fragmentation"]["mean"] >= 1
    assert "number_unit_efficiency" in rep
    assert "native_name_fragmentation" in rep
    assert rep["native_script_vocab_share"] >= 0.0


# -- canonical serialization ------------------------------------------------

def test_serialize_seed_store_produces_structured_text() -> None:
    store = load_store(CANON, strict=True)
    docs = list(iter_documents(store))
    assert len(docs) >= 5
    blob = store_to_text(store)
    assert "<DISH>" in blob and "<STEP>" in blob and "<INGREDIENT>" in blob
    assert "ven pongal" in blob.lower()
