"""Canonical-store schema v2 tests — no GPU, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from food_slm.schema import load_store
from food_slm.schema.models import (
    RECORD_FILES,
    RECORD_MODELS,
    DietProfile,
    LocalName,
    Measurement,
    Relationship,
    parse_record,
)
from food_slm.schema.store import ValidationError

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"


# -- the shipped seed store validates end to end -----------------------------

def test_seed_store_loads_and_validates() -> None:
    store = load_store(CANON, strict=True)
    assert len(store) >= 25
    recipe = store.get("recipe:ven-pongal--classic")
    assert recipe is not None
    assert store.get(recipe.structure.dish_id) is not None
    assert all(r.schema_version == "2" for r in store.records.values())


def test_seed_dish_has_cuisine_state_and_diet() -> None:
    store = load_store(CANON, strict=True)
    dish = store.get("dish:ven-pongal")
    assert dish.structure.cuisine_state == "tamil_nadu"
    assert dish.structure.diet.vegetarian is True
    assert dish.structure.diet.contains_dairy is True  # ghee


def test_every_record_file_has_the_right_type() -> None:
    store = load_store(CANON, strict=True)
    by_file: dict[str, set[str]] = {}
    for rec in store.records.values():
        by_file.setdefault(RECORD_FILES[rec.type], set()).add(rec.type)
    for fname, types in by_file.items():
        assert len(types) == 1, f"{fname} mixes types {types}"


# -- envelope / id rules ----------------------------------------------------

def test_id_prefix_must_match_type() -> None:
    assert parse_record(_dish(id="dish:x")).id == "dish:x"
    with pytest.raises(PydanticValidationError):
        parse_record(_dish(id="ing:x"))


def test_id_slug_must_be_lowercase_ascii() -> None:
    with pytest.raises(PydanticValidationError):
        parse_record(_dish(id="dish:Ven-Pongal"))


def test_community_or_religion_term_rejected_in_label() -> None:
    with pytest.raises(PydanticValidationError, match="community/religion"):
        parse_record(_dish(labels=["Iyengar sambar"]))


def test_community_term_rejected_in_structure() -> None:
    with pytest.raises(PydanticValidationError, match="community/religion"):
        parse_record(_dish(style="brahmin style"))


def test_community_term_in_content_prose_caught_by_store(tmp_path: Path) -> None:
    store_dir = tmp_path / "canonical"
    store_dir.mkdir()
    d = _dish(id="dish:x")
    d["content"]["history"] = "A staple in Muslim households of the region."
    (store_dir / "dishes.jsonl").write_text(json.dumps(d) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="community/religion term"):
        load_store(store_dir, strict=True)


# -- multi-language names (v2) --------------------------------------------

def test_localname_requires_transliteration() -> None:
    LocalName(lang="ml", native="പുട്ട്", transliteration="puttu")
    with pytest.raises(PydanticValidationError):
        LocalName(lang="ml", native="പുട്ട്", transliteration="  ")


def test_names_reject_duplicate_language_and_multiple_primary() -> None:
    with pytest.raises(PydanticValidationError):
        parse_record(_dish(names=[
            {"lang": "ta", "native": "அ", "transliteration": "a", "primary": True},
            {"lang": "ta", "native": "ஆ", "transliteration": "aa"},
        ]))
    with pytest.raises(PydanticValidationError):
        parse_record(_dish(names=[
            {"lang": "ta", "native": "அ", "transliteration": "a", "primary": True},
            {"lang": "ml", "native": "ആ", "transliteration": "aa", "primary": True},
        ]))


# -- diet profile (v2) --------------------------------------------------

def test_dietprofile_rejects_vegetarian_with_meat() -> None:
    DietProfile(vegetarian=False, contains_seafood=True)
    with pytest.raises(PydanticValidationError):
        DietProfile(vegetarian=True, contains_poultry=True)


def test_dietprofile_vegan_rules() -> None:
    DietProfile(vegetarian=True, vegan=True)
    with pytest.raises(PydanticValidationError):
        DietProfile(vegetarian=True, vegan=True, contains_dairy=True)
    with pytest.raises(PydanticValidationError):
        DietProfile(vegetarian=False, vegan=True)


# -- region hierarchy (v2) --------------------------------------------

def test_region_level_parent_rule() -> None:
    with pytest.raises(PydanticValidationError):
        parse_record(_region(level="sub_region", parent_region=None))
    with pytest.raises(PydanticValidationError):
        parse_record(_region(level="state", parent_region="region:x"))
    parse_record(_region(level="state", parent_region=None, cuisine_state="kerala"))


# -- measurement normalization --------------------------------------------

def test_qualitative_measurement_rejects_numeric_value() -> None:
    with pytest.raises(PydanticValidationError):
        Measurement(original_text="to taste", dimension="qualitative",
                    qualitative="to_taste", value=1.0)


def test_numeric_measurement_requires_value() -> None:
    with pytest.raises(PydanticValidationError):
        Measurement(original_text="1 cup", dimension="volume", unit="cup")


def test_range_lo_hi_ordering() -> None:
    with pytest.raises(PydanticValidationError):
        Measurement(original_text="x", dimension="mass", value=5.0, range=(9.0, 1.0))


# -- relationship predicate typing ---------------------------------------

def test_relationship_rejects_illegal_object_type() -> None:
    obj = {
        "id": "rel:99999", "type": "rel", "schema_version": "2",
        "structure": {"predicate": "used_in", "subject_id": "ing:x", "object_id": "region:y"},
        "provenance": {"source_ids": ["src:s"], "transformations": ["hand_authored"]},
        "quality": {"confidence": 0.9, "validation_status": "human_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }
    with pytest.raises(PydanticValidationError):
        parse_record(obj)


# -- recipe DAG ---------------------------------------------------------

def test_recipe_dag_rejects_forward_dependency(tmp_path: Path) -> None:
    store_dir = tmp_path / "canonical"
    store_dir.mkdir()
    (store_dir / "recipes.jsonl").write_text(_bad_dag_recipe_line(), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_store(store_dir, strict=True)


def test_dangling_reference_is_caught(tmp_path: Path) -> None:
    store_dir = tmp_path / "canonical"
    store_dir.mkdir()
    (store_dir / "dishes.jsonl").write_text(
        json.dumps(_dish(id="dish:x", primary_ingredients=["ing:does-not-exist"])) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="dangling"):
        load_store(store_dir, strict=True)


# -- helpers ----------------------------------------------------------------

_VEG = {"vegetarian": True, "vegan": False, "contains_dairy": False, "contains_egg": False,
        "contains_seafood": False, "contains_poultry": False, "contains_red_meat": False}


def _dish(**over) -> dict:
    d = {
        "id": "dish:x", "type": "dish", "schema_version": "2",
        "labels": over.pop("labels", ["X"]),
        "content": {"summary": "s"},
        "structure": {"category": "tiffin", "cuisine_state": "tamil_nadu", "diet": dict(_VEG),
                      "primary_ingredients": []},
        "provenance": {"source_ids": ["src:s"], "transformations": ["hand_authored"]},
        "quality": {"confidence": 0.9, "validation_status": "human_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }
    if "id" in over:
        d["id"] = over.pop("id")
    d["structure"].update(over)
    return d


def _region(**over) -> dict:
    return {
        "id": over.pop("id", "region:x"), "type": "region", "schema_version": "2",
        "labels": ["X"],
        "content": {"summary": "s"},
        "structure": {"level": over.pop("level", "state"),
                      "parent_region": over.pop("parent_region", None),
                      "cuisine_state": over.pop("cuisine_state", "tamil_nadu"), **over},
        "provenance": {"source_ids": ["src:s"], "transformations": ["hand_authored"]},
        "quality": {"confidence": 0.9, "validation_status": "human_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }


def _bad_dag_recipe_line() -> str:
    rec = {
        "id": "recipe:x--y", "type": "recipe", "schema_version": "2", "labels": ["X"],
        "content": {"summary": "s"},
        "structure": {
            "dish_id": "dish:x", "cuisine_state": "tamil_nadu", "diet": dict(_VEG),
            "ingredients": [{"ingredient_id": "ing:a"}],
            "steps": [
                {"step_id": "s1", "order": 1, "phase": "cook", "action": "a",
                 "action_verb": "boil", "depends_on": ["s2"]},
                {"step_id": "s2", "order": 2, "phase": "cook", "action": "b",
                 "action_verb": "boil", "depends_on": []},
            ],
            "states": [],
        },
        "provenance": {"source_ids": ["src:s"], "transformations": ["hand_authored"]},
        "quality": {"confidence": 0.9, "validation_status": "human_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }
    return json.dumps(rec) + "\n"


def test_all_record_models_registered() -> None:
    from food_slm.schema.models import RecordType

    assert set(RECORD_MODELS) == set(RecordType.__args__)
    assert set(RECORD_FILES) == set(RECORD_MODELS)
    assert issubclass(RECORD_MODELS["rel"], Relationship)
