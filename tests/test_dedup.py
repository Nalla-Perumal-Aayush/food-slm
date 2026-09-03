"""Dedup: MinHash near-exact + local semantic embeddings."""
from __future__ import annotations

import pytest

from food_slm.data.dedup import (
    MinHashDeduper,
    RecordDeduper,
    SemanticDeduper,
    record_signature,
)


def test_minhash_catches_verbatim_and_lightly_edited_copies() -> None:
    passage = ("Heat ghee in a pan and add cumin seeds and crushed black pepper. "
               "When they crackle add the cashews and fry until golden. Add curry leaves "
               "and a pinch of asafoetida, then pour the whole tempering over the pongal.")
    d = MinHashDeduper(threshold=0.5)
    d.add("src-a", passage)
    dup, hits = d.is_duplicate(passage)                       # verbatim
    assert dup and "src-a" in hits
    dup2, _ = d.is_duplicate(passage.replace("golden", "golden brown").replace("pan", "kadai"))
    assert dup2                                               # lightly edited
    dup3, _ = d.is_duplicate("An unrelated paragraph about transplanting rice seedlings.")
    assert not dup3


def test_record_signature_uses_names_and_ingredients() -> None:
    rec = {
        "id": "dish:x", "type": "dish", "labels": ["Bisi Bele Bath"],
        "structure": {"names": [{"transliteration": "bisi bele bath"}],
                      "cuisine_state": "karnataka",
                      "primary_ingredients": ["ing:rice", "ing:toor-dal"]},
    }
    sig = record_signature(rec)
    assert "bisi bele bath" in sig and "karnataka" in sig and "ing rice" in sig


@pytest.mark.slow
def test_semantic_dedup_same_dish_different_words() -> None:
    d = SemanticDeduper(threshold=0.80)
    d.add("pongal", "ven pongal soft rice and moong dal tiffin with pepper cumin ghee tempering")
    dup, key, score = d.is_duplicate(
        "khara pongal creamy rice lentil breakfast seasoned with black pepper and cumin in ghee")
    assert dup and key == "pongal" and score >= 0.80
    dup2, _, s2 = d.is_duplicate("chocolate cake with buttercream frosting and sprinkles")
    assert not dup2 and s2 < 0.6


@pytest.mark.slow
def test_record_deduper_merge_vs_new() -> None:
    dd = RecordDeduper()
    dd.index([{
        "id": "dish:ven-pongal", "type": "dish", "labels": ["Ven Pongal", "Khara Pongal"],
        "structure": {"names": [{"transliteration": "ven pongal"}], "cuisine_state": "tamil_nadu",
                      "primary_ingredients": ["ing:rice-raw", "ing:moong-dal"]},
    }])
    verdict, mid, _ = dd.match({"id": "dish:khara-pongal", "type": "dish",
                                "labels": ["Khara Pongal"], "structure": {"names": []}})
    assert verdict == "merge" and mid == "dish:ven-pongal"
    v2, _, _ = dd.match({"id": "dish:rava-kesari", "type": "dish", "labels": ["Rava Kesari"],
                         "structure": {"names": [{"transliteration": "rava kesari"}],
                                       "cuisine_state": "karnataka",
                                       "primary_ingredients": ["ing:rava", "ing:sugar"]}})
    assert v2 == "new"
