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


def test_nested_step_equipment_ref_to_nonexistent_equipment_gets_stripped():
    """_ID_REF_FIELDS only ever covered recipe's TOP-level equipment_ids -- a step's
    own equipment_id (nested) wasn't checked, so a step could keep pointing at
    equipment that doesn't exist anywhere, surviving promotion and failing strict
    load_store() later. Confirmed live: 796 such dangling refs across
    already-promoted recipes."""
    rec = _recipe("recipe:idli-x", confidence=0.85, corroboration=1)
    rec["structure"]["equipment_ids"] = []
    rec["structure"]["ingredients"] = [{"ingredient_id": "ing:rice"}]
    rec["structure"]["steps"] = [
        {"step_id": "step-1", "order": 1, "phase": "cook", "action": "heat",
         "action_verb": "heat", "equipment_id": "equip:nonexistent-pan",
         "input_ingredient_ids": ["ing:rice", "ing:not-in-recipe-ingredients"], "depends_on": []},
    ]
    staged = [_source(), rec]
    ids, patches = select_promotable(staged, already_canonical=set())
    assert "recipe:idli-x" in ids
    step = patches["recipe:idli-x"]["structure"]["steps"][0]
    assert "equipment_id" not in step
    # input_ingredient_ids is self-consistency against the recipe's own
    # ingredients list, not global existence -- "ing:rice" IS declared so it
    # survives even though no standalone ingredient:rice record exists anywhere.
    assert step["input_ingredient_ids"] == ["ing:rice"]
    # already [] and stays [] -- no patch entry needed for an unchanged field
    assert "equipment_ids" not in patches["recipe:idli-x"].get("structure", {})


def test_nested_step_ref_to_resolvable_equipment_is_backfilled_into_top_level():
    """The self-consistency half of the same fix: a step's equipment_id that DOES
    resolve must also end up in the recipe's own top-level equipment_ids."""
    rec = _recipe("recipe:idli-x", confidence=0.85, corroboration=1)
    rec["structure"]["equipment_ids"] = []
    rec["structure"]["steps"] = [
        {"step_id": "step-1", "order": 1, "phase": "cook", "action": "heat",
         "action_verb": "heat", "equipment_id": "equip:idli-stand",
         "input_ingredient_ids": [], "depends_on": []},
    ]
    staged = [_source(), rec]
    ids, patches = select_promotable(staged, already_canonical={"equip:idli-stand"})
    assert "recipe:idli-x" in ids
    assert patches["recipe:idli-x"]["structure"]["equipment_ids"] == ["equip:idli-stand"]
