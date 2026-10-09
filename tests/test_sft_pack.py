"""sft_pack.py: loss-masking index correctness (the most error-prone part --
off-by-one here means training on the wrong span, silently) + entity-disjoint
split/exclusion logic."""
from __future__ import annotations

from food_slm.data.sft_pack import (
    entity_disjoint_split,
    load_golden_dish_slugs,
    normalize_dish_slug,
    resolve_entity_key,
    tokenize_example,
)

_SPECIAL_IDS = {"pad": 0, "bos": 1, "eos": 2, "user": 3, "assistant": 4}


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    """Maps each word to a distinct id >= 10 deterministically, so we can predict
    the exact token sequence tokenize_example() should produce."""

    def encode(self, text):
        words = text.split()
        return _FakeEncoding([10 + i for i, _ in enumerate(words)])


def test_tokenize_example_masks_exactly_the_response_plus_eos():
    tok = _FakeTok()
    # user="a b" -> [10, 11]; assistant="c d" -> [10, 11] (fresh index per call)
    input_ids, labels = tokenize_example(tok, "a b", "c d", max_seq_len=12,
                                         special_ids=_SPECIAL_IDS)
    # full = [bos, user, 10, 11, assistant, 10, 11, eos]
    #  idx:    0    1    2   3      4        5   6   7
    expected_input = [1, 3, 10, 11, 4, 10, 11, 2] + [0] * 4  # padded with pad=0
    assert input_ids.tolist() == expected_input
    # resp_start = 1 + len(user_ids) = 1 + 2 = 3 -- wait, user_ids includes the
    # marker: user_ids = [user, 10, 11], len=3, so resp_start = 1+3 = 4 (the
    # <|assistant|> marker's own index) -- labels[4] stays masked, labels[5:8]
    # (the two response tokens + eos) carry the real target.
    expected_labels = [-100, -100, -100, -100, -100, 10, 11, 2] + [-100] * 4
    assert labels.tolist() == expected_labels


def test_tokenize_example_returns_none_when_too_long():
    tok = _FakeTok()
    assert tokenize_example(tok, "a b", "c d", max_seq_len=4, special_ids=_SPECIAL_IDS) is None


def test_normalize_dish_slug():
    assert normalize_dish_slug("fish_curry") == "fish-curry"
    assert normalize_dish_slug("dish:fish-curry") == "fish-curry"
    assert normalize_dish_slug("Fish_Curry") == "fish-curry"


class _FakeRecipeRecord:
    def __init__(self, dish_id):
        self.structure = type("S", (), {"dish_id": dish_id})()


class _FakeStore:
    def __init__(self, records):
        self.records = records


def test_resolve_entity_key_recipe_resolves_to_dish_id():
    store = _FakeStore({"recipe:idli-src1": _FakeRecipeRecord("dish:idli")})
    assert resolve_entity_key("recipe:idli-src1", store) == "dish:idli"


def test_resolve_entity_key_non_recipe_is_identity():
    store = _FakeStore({})
    assert resolve_entity_key("ing:tamarind", store) == "ing:tamarind"
    assert resolve_entity_key("tech:tempering", store) == "tech:tempering"


def _row(question, answer, backing, task_type="definition"):
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}],
           "provenance": {"task_type": task_type, "backing_record_ids": backing}}


def test_split_excludes_golden_overlap_entirely():
    store = _FakeStore({"recipe:idli-src1": _FakeRecipeRecord("dish:idli"),
                       "recipe:dosa-src1": _FakeRecipeRecord("dish:dosa")})
    rows = [
        _row("Q1", "A1", ["recipe:idli-src1"]),
        _row("Q2", "A2", ["recipe:dosa-src1"]),
        _row("Q3", "A3", []),  # ungrouped (e.g. abstention) -- always kept
    ]
    golden_slugs = {"idli"}  # golden QA references idli
    res = entity_disjoint_split(rows, store, golden_slugs, val_frac=0.0, seed=0)
    assert len(res.excluded_golden_overlap) == 1
    assert res.excluded_golden_overlap[0]["messages"][0]["content"] == "Q1"
    kept_questions = {r["messages"][0]["content"] for r in res.train + res.val}
    assert kept_questions == {"Q2", "Q3"}


def test_split_keeps_same_dish_examples_together():
    """Two examples backed by the SAME dish must land on the same side of the
    train/val split -- splitting at the example level (not entity level) would
    leak that dish's facts across both sides."""
    store = _FakeStore({"recipe:idli-src1": _FakeRecipeRecord("dish:idli"),
                       "recipe:idli-src2": _FakeRecipeRecord("dish:idli")})
    rows = [_row("Q1", "A1", ["recipe:idli-src1"]), _row("Q2", "A2", ["recipe:idli-src2"])]
    res = entity_disjoint_split(rows, store, set(), val_frac=1.0, seed=0)
    all_val_qs = {r["messages"][0]["content"] for r in res.val}
    all_train_qs = {r["messages"][0]["content"] for r in res.train}
    assert all_val_qs == {"Q1", "Q2"} or all_train_qs == {"Q1", "Q2"}
    assert not (all_val_qs and all_train_qs)  # never split across both


def test_load_golden_dish_slugs(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text(
        '{"dish_ids": ["fish_curry", "idli"]}\n{"dish_ids": []}\n', encoding="utf-8")
    assert load_golden_dish_slugs(p) == {"fish-curry", "idli"}


def test_load_golden_dish_slugs_missing_file(tmp_path):
    assert load_golden_dish_slugs(tmp_path / "missing.jsonl") == set()
