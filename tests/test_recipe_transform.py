"""Recipe dataset normalisation + rules-first transform (150M data plan, Level 1/2)."""
from __future__ import annotations

from food_slm.data.recipe_sources import normalise, relevance_tier
from food_slm.data.recipe_transform import classify_step, parse_ingredient, to_prose

# -- normalise ------------------------------------------------------------

def test_normalise_food_com_r_vectors():
    raw = {
        "RecipeId": 42, "Name": "Coconut Chutney",
        "RecipeCategory": "South Indian",
        "RecipeIngredientParts": 'c("coconut", "green chili", "urad dal")',
        "RecipeIngredientQuantities": 'c("1", "2", "1 tsp")',
        "RecipeInstructions": 'c("Grind coconut with chili.", "Temper with mustard seeds.")',
        "Keywords": 'c("Indian", "Breakfast")', "TotalTime": "PT15M", "RecipeServings": 4,
    }
    row = normalise("food_com", raw)
    assert row["title"] == "Coconut Chutney"
    assert row["ingredients"] == ["1 coconut", "2 green chili", "1 tsp urad dal"]
    assert len(row["instructions"]) == 2
    assert row["total_min"] == 15


def test_normalise_archanas_and_epicurious():
    a = normalise("archanas", {
        "TranslatedRecipeName": "Kerala Sambar", "Cuisine": "Kerala Recipes",
        "TranslatedIngredients": "1 cup Toor dal, 2 Drumstick (cut), 1 lime Tamarind",
        "TranslatedInstructions": "Cook the dal. Add vegetables and tamarind. Simmer.",
        "URL": "https://x/kerala-sambar", "TotalTimeInMins": 40})
    assert a["cuisine_raw"] == "Kerala Recipes"
    assert len(a["ingredients"]) == 3

    e = normalise("epicurious", {
        "Unnamed: 0": 7, "Title": "Lentil Soup",
        "Cleaned_Ingredients": "['1 cup lentils', '1 onion', '2 cups water']",
        "Instructions": "Rinse lentils.\nSimmer with onion and water for 30 minutes."})
    assert e["ingredients"] == ["1 cup lentils", "1 onion", "2 cups water"]
    assert len(e["instructions"]) == 2


def test_normalise_rejects_thin_rows():
    assert normalise("epicurious", {"Title": "X", "Cleaned_Ingredients": "['a']",
                                     "Instructions": ""}) is None


# -- relevance tier -----------------------------------------------------

def test_tier_core_si_by_cuisine_dish_and_ingredients():
    assert relevance_tier({"title": "Onion Uttapam", "cuisine_raw": "South Indian Recipes",
                           "ingredients": [], "tags": []})[0] == "core_si"
    assert relevance_tier({"title": "Ven Pongal", "cuisine_raw": "", "ingredients": [],
                           "tags": []})[0] == "core_si"
    t, regions = relevance_tier({"title": "Fish curry", "cuisine_raw": "Kerala",
                                 "ingredients": ["kudampuli", "curry leaves", "coconut",
                                                 "mustard seeds"], "tags": []})
    assert t == "core_si" and "kerala" in regions


def test_tier_general_and_pan_indian():
    assert relevance_tier({"title": "Beef Bourguignon", "cuisine_raw": "French",
                           "ingredients": ["beef", "wine"], "tags": []})[0] == "general"
    assert relevance_tier({"title": "Paneer Butter Masala", "cuisine_raw": "North Indian",
                           "ingredients": ["paneer", "cream"], "tags": []})[0] == "pan_indian"


# -- parse_ingredient -------------------------------------------------

def test_parse_ingredient_quantities_and_units():
    p = parse_ingredient("2 cups idli rice")
    assert p["quantity"]["value"] == 2 and p["quantity"]["unit"] == "cup"
    assert p["quantity"]["value_canonical"] == 480.0 and p["name"] == "idli rice"

    p = parse_ingredient("1½ tsp mustard seeds")
    assert p["quantity"]["value"] == 1.5 and p["quantity"]["unit"] == "tsp"

    p = parse_ingredient("Salt, to taste")
    assert p["quantity"] is None

    p = parse_ingredient("2 onions, finely chopped")
    assert p["quantity"]["dimension"] == "count"
    assert "finely chopped" in p["prep_note"] and p["name"] == "onions"


# -- classify_step --------------------------------------------------

def test_classify_step_maps_to_enum_verbs_and_phases():
    assert classify_step("Soak the rice for 4 hours.") == {
        "action_verb": "soak", "phase": "soak", "until_condition": None, "duration_min": 240.0}
    assert classify_step("Grind to a smooth paste.")["action_verb"] == "grind"
    s = classify_step("Temper mustard seeds until they splutter.")
    assert s["action_verb"] == "temper" and s["phase"] == "temper"
    assert s["until_condition"] == "they splutter"
    assert classify_step("Pressure cook for 3 whistles.")["action_verb"] == "pressure_cook"
    assert classify_step("Serve hot with chutney.")["phase"] == "serve"


# -- to_prose -----------------------------------------------------

def test_to_prose_is_clean_readable_text():
    row = {"title": "Kerala Fish Curry", "cuisine_raw": "Kerala Recipes",
           "region_tags": ["kerala"], "servings": None, "total_min": 35,
           "ingredients": ["500 g sardines", "3 pieces kudampuli", "1 cup coconut, grated"],
           "instructions": ["Grind coconut with spices", "Simmer fish in the gravy"]}
    text = to_prose(row)
    assert text.startswith("Kerala Fish Curry")
    assert "A Kerala recipe from Kerala." in text
    assert "Ingredients:" in text and "Method:" in text
    assert "- 500 g sardines" in text
    assert "1. Grind coconut with spices." in text
