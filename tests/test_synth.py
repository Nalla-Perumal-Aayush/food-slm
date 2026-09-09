"""Template synthetic generation from the canonical store (150M plan, D-series)."""
from __future__ import annotations

from pathlib import Path

from food_slm.data.forbidden import load_forbidden
from food_slm.data.synth import abstention_items, phrasings, qa_pairs
from food_slm.schema import load_store

ROOT = Path(__file__).resolve().parents[1]
STORE = load_store(ROOT / "data" / "canonical", strict=False)


def test_phrasings_are_nonempty_prose_with_provenance():
    items = list(phrasings(STORE))
    assert len(items) >= len(STORE.records) // 2          # ~>= 1 per renderable record
    for x in items[:50]:
        assert x["text"].strip() and len(x["text"]) > 15
        assert x["provenance"] and x["provenance"][0] in STORE.records
        assert x["kind"].startswith("phrasing:")
    # deterministic count
    assert len(items) == len(list(phrasings(STORE)))


def test_phrasings_render_diet_correctly():
    texts = " ".join(x["text"] for x in phrasings(STORE))
    assert "vegan" in texts.lower()
    assert "non-vegetarian" in texts.lower()          # chicken/fish dishes exist in the seed


def test_qa_pairs_cover_dish_questions():
    qa = list(qa_pairs(STORE))
    qs = {x["question"] for x in qa}
    assert any(q.startswith("What is ") for q in qs)
    assert any(q.startswith("Is ") and "vegetarian" in q for q in qs)
    for x in qa:
        assert x["answer"].strip() and x["provenance"][0] in STORE.records


def test_abstention_items_use_forbidden_surface_forms():
    fb = load_forbidden(ROOT / "data" / "eval")
    ab = list(abstention_items(fb))
    assert ab
    fe_ids = {e.id for e in fb.entities}
    for x in ab:
        assert x["provenance"][0] in fe_ids
        assert "not familiar" in x["answer"].lower() or "don't have reliable" in x["answer"].lower() \
            or "can't confidently" in x["answer"].lower()
