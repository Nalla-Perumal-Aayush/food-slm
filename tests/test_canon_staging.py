"""select_promotable(): trust-anchored closure, including the recipe-specific
confidence anchor (corroboration doesn't apply to recipes the way it does to flat
entities -- see canon_staging.py's module comment)."""
from __future__ import annotations

from food_slm.data.canon_staging import select_promotable


def _recipe(rid: str, *, confidence: float, synthetic: bool = False,
           corroboration: int = 1, dish_id: str = "dish:idli") -> dict:
    return {
        "id": rid, "type": "recipe", "schema_version": "2", "labels": ["Idli"],
        "content": {"summary": "x"},
        "structure": {"dish_id": dish_id, "cuisine_state": "tamil_nadu",
                      "diet": {"vegetarian": True}, "ingredients": [{"ingredient_id": "ing:rice"}],
                      "steps": [{"step_id": "step-1", "order": 1, "phase": "cook",
                                "action": "cook", "action_verb": "add",
                                "input_ingredient_ids": [], "depends_on": []}]},
        "provenance": ({"source_ids": [], "transformations": ["generated"], "synthetic": True,
                       "generator": "test", "seed_record_ids": [dish_id]} if synthetic
                      else {"source_ids": ["src:x"], "transformations": ["extracted"],
                           "synthetic": False}),
        "quality": {"confidence": confidence, "validation_status": "unverified",
                   "corroboration_count": corroboration},
        "version": {"revision": 1, "created_on": "2026-01-01", "updated_on": "2026-01-01",
                   "status": "active"},
    }


def _source(rid: str = "src:x") -> dict:
    return {"id": rid, "type": "source", "schema_version": "2", "labels": ["X"],
           "content": {}, "structure": {}, "provenance": {"source_ids": []},
           "quality": {"confidence": 0.9, "validation_status": "auto_validated",
                      "corroboration_count": 1},
           "version": {"revision": 1, "created_on": "2026-01-01", "updated_on": "2026-01-01",
                      "status": "active"}}


def test_unanimous_accept_recipe_promotes_despite_corroboration_1():
    staged = [_source(), _recipe("recipe:idli-x", confidence=0.85, corroboration=1)]
    ids, _ = select_promotable(staged, already_canonical=set())
    assert "recipe:idli-x" in ids


def test_split_vote_recipe_at_0_55_does_not_promote():
    staged = [_source(), _recipe("recipe:idli-x", confidence=0.55, corroboration=1)]
    ids, _ = select_promotable(staged, already_canonical=set())
    assert "recipe:idli-x" not in ids


def test_synthetic_recipe_never_promotes_even_at_high_confidence():
    """The exact real-world case this guards against: 73 generated recipes in the
    live staging dir have confidence=0.85 (panel "accepted" the plausibility check)
    -- without the explicit synthetic exclusion these would wrongly auto-promote."""
    staged = [_recipe("recipe:idli-gen", confidence=0.85, synthetic=True, corroboration=0)]
    ids, _ = select_promotable(staged, already_canonical=set())
    assert "recipe:idli-gen" not in ids


def test_corroboration_2_still_promotes_regardless_of_type():
    staged = [_source(), _recipe("recipe:idli-x", confidence=0.55, corroboration=2)]
    ids, _ = select_promotable(staged, already_canonical=set())
    assert "recipe:idli-x" in ids
