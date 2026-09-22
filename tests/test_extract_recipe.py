"""extract_recipe.py: LLM's simple recipe JSON -> schema-v2 `recipe` record."""
from __future__ import annotations

import json

from food_slm.data.extract_recipe import extract_recipe, parse_measurement
from food_slm.data.llm import FakeProvider, ProviderSpec

VEG_DIET = {"vegetarian": True, "vegan": True, "non_veg": False}
NONVEG_DIET = {"vegetarian": False, "vegan": False, "non_veg": True, "contains": ["poultry"]}

_FULL_RECIPE = {
    "recipes": [{
        "variant_label": None, "style": "udupi", "servings": 4,
        "total_time_minutes": 40, "active_time_minutes": 20, "difficulty": "moderate",
        "ingredients": [
            {"name": "idli rice", "quantity_text": "2 cups", "role": "base"},
            {"name": "urad dal", "quantity_text": "1/2 cup", "role": "base"},
            {"name": "salt", "quantity_text": "to taste", "role": "tempering"},
        ],
        "equipment": ["idli stand", "wet grinder"],
        "steps": [
            {"action": "Soak rice and dal separately for 4 hours.", "phase": "soak",
             "action_verb": "soak", "ingredients_used": ["idli rice", "urad dal"],
             "duration_minutes": 240},
            {"action": "Grind into a smooth batter.", "phase": "grind",
             "action_verb": "grind", "ingredients_used": ["idli rice", "urad dal"],
             "equipment": "wet grinder"},
            {"action": "Ferment overnight.", "phase": "ferment", "action_verb": "ferment",
             "ingredients_used": [], "duration_minutes": 480},
            {"action": "Add salt and mix.", "phase": "assemble", "action_verb": "mix",
             "ingredients_used": ["salt"]},
            {"action": "Steam in an idli stand until cooked through.", "phase": "cook",
             "action_verb": "steam", "ingredients_used": [], "equipment": "idli stand",
             "heat_level": "medium", "duration_minutes": 12, "doneness_cue": "toothpick clean"},
        ],
        "tips": ["Use old rice for better fermentation."],
        "common_mistakes": ["Not fermenting long enough in cold weather."],
        "serving_suggestions": "Serve hot with sambar and chutney.",
    }],
    "notes": "",
}


def _provider(payload: dict) -> FakeProvider:
    return FakeProvider(spec=ProviderSpec(name="fake", kind="fake", model="fake", family="open"),
                        responder=lambda s, p: json.dumps(payload))


def test_full_recipe_builds_and_validates():
    res = extract_recipe("page text", "dish:idli", "Idli", "karnataka", "src:example-idli",
                         VEG_DIET, _provider(_FULL_RECIPE))
    assert res.invalid == []
    assert len(res.drafts) == 1
    rec = res.drafts[0]
    assert rec["id"] == "recipe:idli-example-idli"
    struct = rec["structure"]
    assert struct["dish_id"] == "dish:idli"
    assert struct["diet"] == {
        "vegetarian": True, "vegan": True, "contains_dairy": False, "contains_egg": False,
        "contains_seafood": False, "contains_poultry": False, "contains_red_meat": False,
    }
    assert len(struct["ingredients"]) == 3
    assert len(struct["steps"]) == 5
    # linear chain: each step depends only on the previous one
    ids = [s["step_id"] for s in struct["steps"]]
    assert struct["steps"][0]["depends_on"] == []
    for prev, s in zip(ids, struct["steps"][1:]):
        assert s["depends_on"] == [prev]
    salt = next(i for i in struct["ingredients"] if i["ingredient_id"] == "ing:salt")
    assert salt["quantity"]["dimension"] == "qualitative"
    assert salt["quantity"]["qualitative"] == "to_taste"


def test_nonveg_diet_flows_through():
    payload = {"recipes": [{
        "servings": 4, "difficulty": "easy",
        "ingredients": [{"name": "chicken", "quantity_text": "500 g", "role": "protein"}],
        "steps": [{"action": "Cook the chicken.", "phase": "cook", "action_verb": "boil",
                   "ingredients_used": ["chicken"]}],
    }], "notes": ""}
    res = extract_recipe("page text", "dish:chicken-curry", "Chicken Curry", "andhra_pradesh",
                         "src:example-chicken", NONVEG_DIET, _provider(payload))
    assert len(res.drafts) == 1
    diet = res.drafts[0]["structure"]["diet"]
    assert diet["vegetarian"] is False
    assert diet["contains_poultry"] is True


def test_no_recipe_found_yields_no_drafts():
    res = extract_recipe("irrelevant page", "dish:idli", "Idli", "karnataka", "src:x",
                         VEG_DIET, _provider({"recipes": [], "notes": "no recipe found"}))
    assert res.drafts == []
    assert res.invalid == []
    assert res.notes == "no recipe found"


def test_synonym_coercion_and_variant_id():
    payload = {"recipes": [{
        "variant_label": "Hotel-style", "servings": "500",
        "ingredients": [{"name": "rice", "quantity_text": "2 cups"}],
        "steps": [{"action": "Boil rice.", "phase": "COOKING", "action_verb": "Boiling",
                   "ingredients_used": ["rice"]}],
    }], "notes": ""}
    res = extract_recipe("page", "dish:idli", "Idli", "karnataka", "src:hotel",
                         VEG_DIET, _provider(payload))
    assert len(res.drafts) == 1
    rec = res.drafts[0]
    assert rec["id"] == "recipe:idli-hotel-style-hotel"
    step = rec["structure"]["steps"][0]
    assert step["phase"] == "cook"
    assert step["action_verb"] == "boil"


def test_parse_measurement_variants():
    assert parse_measurement("2 cups") == {
        "original_text": "2 cups", "value": 2.0, "unit": "cup", "value_canonical": 2.0,
        "unit_canonical": "cup", "dimension": "volume", "approximate": False,
    }
    rng = parse_measurement("2-3 tsp")
    assert rng["dimension"] == "volume" and rng["range"] == [2.0, 3.0]
    half_kg = parse_measurement("1/2 kg")
    assert half_kg["value"] == 0.5 and half_kg["unit"] == "kg" and half_kg["dimension"] == "mass"
    mixed = parse_measurement("1 1/2 cups")
    assert mixed["value"] == 1.5 and mixed["unit"] == "cup"
    frac_range = parse_measurement("1/2-3/4 cup")
    assert frac_range["range"] == [0.5, 0.75]
    assert parse_measurement("a pinch")["qualitative"] == "a_pinch"
    assert parse_measurement("500")["dimension"] == "count"
    assert parse_measurement(None) is None
    assert parse_measurement("") is None
