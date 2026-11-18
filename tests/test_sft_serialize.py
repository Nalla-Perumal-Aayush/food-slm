"""sft_serialize.py: canonical records -> {"messages": [...], "provenance": {...}}
instruction examples, one function per SS4.2 task type."""
from __future__ import annotations

import random

from food_slm.data.sft_serialize import (
    _clean_summary,
    abstention_examples,
    serialize_ingredient_definition,
    serialize_recipe_request,
    serialize_regional_comparison,
    serialize_scaling,
    serialize_substitution,
    serialize_technique,
    serialize_term_definition,
    serialize_troubleshooting,
)


def _rng() -> random.Random:
    return random.Random(0)


def _assert_example_shape(ex: dict, task_type: str, min_backing: int = 0) -> None:
    assert set(ex.keys()) == {"messages", "provenance"}
    assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
    assert ex["messages"][0]["content"] and ex["messages"][1]["content"]
    prov = ex["provenance"]
    assert prov["task_type"] == task_type
    assert prov["synthetic"] is True
    assert len(prov["backing_record_ids"]) >= min_backing


_RECIPE = {
    "id": "recipe:idli-x", "labels": ["Idli"],
    "content": {"tips": ["Use old rice."], "common_mistakes": ["Not fermenting long enough."],
               "serving_suggestions": "Serve hot with sambar."},
    "structure": {
        "yield": {"servings": 4},
        "ingredients": [
            {"ingredient_id": "ing:idli-rice", "quantity": {"original_text": "2 cups", "value": 2.0,
                                                            "unit_canonical": "cup"}, "form": "soaked"},
            {"ingredient_id": "ing:salt", "quantity": {"original_text": "to taste",
                                                       "qualitative": "to_taste"}, "optional": True},
        ],
        "steps": [
            {"order": 1, "action": "soak the rice and dal"},
            {"order": 2, "action": "grind into a batter"},
        ],
    },
}


def test_recipe_request():
    ex = serialize_recipe_request(_RECIPE, _rng())
    _assert_example_shape(ex, "recipe_request", 1)
    a = ex["messages"][1]["content"]
    assert "idli rice" in a and "2 cups" in a
    assert "1. Soak the rice and dal." in a
    assert "Serve hot with sambar." in a
    assert "Idli" in ex["messages"][0]["content"]


def test_scaling_computes_ratio_and_includes_nonlinear_note():
    ex = serialize_scaling(_RECIPE, 8, _rng())
    _assert_example_shape(ex, "scaling", 1)
    a = ex["messages"][1]["content"]
    assert "idli rice: 4" in a  # 2 cups * (8/4) = 4
    assert "don't scale linearly" in a


def test_scaling_returns_none_without_a_base_servings_count():
    recipe = {**_RECIPE, "structure": {**_RECIPE["structure"], "yield": {}}}
    assert serialize_scaling(recipe, 8, _rng()) is None


_SUB_REL = {
    "id": "rel:x", "content": {"note": "Changes texture slightly."},
    "structure": {"predicate": "substitutes_for", "subject_id": "ing:toor-dal",
                 "object_id": "ing:moong-dal",
                 "attributes": {"ratio": "1:1", "context": "kootu", "caveat": "slightly stickier"}},
}


def test_substitution():
    ex = serialize_substitution(_SUB_REL, _rng())
    _assert_example_shape(ex, "ingredient_substitution", 1)
    a = ex["messages"][1]["content"]
    assert "toor dal instead of moong dal" in a
    assert "1:1" in a and "kootu" in a and "stickier" in a
    assert "moong dal" in ex["messages"][0]["content"]


_PROBLEM = {
    "id": "problem:x",
    "content": {"symptoms": "The chutney separates.", "diagnosis": "Too much water while grinding.",
               "fixes": ["Grind with less water"], "prevention": ["Add water gradually"]},
}


def test_troubleshooting():
    ex = serialize_troubleshooting(_PROBLEM, _rng())
    _assert_example_shape(ex, "troubleshooting", 1)
    a = ex["messages"][1]["content"]
    assert "Too much water while grinding." in a
    assert "Grind with less water." in a
    assert "Add water gradually." in a
    assert "chutney separates" in ex["messages"][0]["content"]


_TECH = {
    "id": "tech:tempering", "labels": ["Tempering"],
    "content": {"summary": "Tempering is frying whole spices in hot oil to release flavour.",
               "how_to": "Heat oil, add mustard seeds, let them pop.",
               "sensory_cues": "the seeds crackle and the oil smells fragrant",
               "failure_modes": "burning the spices makes everything bitter"},
}


def test_technique_explanation():
    ex = serialize_technique(_TECH, _rng())
    _assert_example_shape(ex, "technique_explanation", 1)
    a = ex["messages"][1]["content"]
    assert "Tempering is frying" in a
    assert "crackle" in a
    assert "bitter" in a


_DISH_A = {
    "id": "dish:a", "labels": ["Tamil Rasam"],
    "content": {"summary": "A tangy Tamil Nadu rasam."},
    "structure": {"category": "rasam", "cuisine_state": "tamil_nadu",
                 "primary_ingredients": ["ing:tamarind", "ing:tomato", "ing:pepper"]},
}
_DISH_B = {
    "id": "dish:b", "labels": ["Andhra Rasam"],
    "content": {"summary": "A spicier Andhra-style rasam."},
    "structure": {"category": "rasam", "cuisine_state": "andhra_pradesh",
                 "primary_ingredients": ["ing:tamarind", "ing:tomato", "ing:red-chili"]},
}


def test_clean_summary_strips_unresolved_ref_audit_note():
    """pipeline.resolve_batch_refs() appends this exact shape to content.summary
    when a structural ref doesn't resolve -- confirmed live in 155/2,778 SFT
    training examples, and a real SFT'd model reproduced the leaked text verbatim
    ("[unresolved primary_ingredients: sambar-dal]") in a live generation."""
    polluted = ("A South Indian snack made from a mixture of bajra flour, whole "
               "wheat flour, and other ingredients, fried until crispy and golden "
               "brown.  [unresolved primary_ingredients: green-chillies; unresolved "
               "primary_ingredients: garlic; unresolved primary_ingredients: kalonji]")
    cleaned = _clean_summary(polluted)
    assert cleaned == ("A South Indian snack made from a mixture of bajra flour, whole "
                      "wheat flour, and other ingredients, fried until crispy and golden "
                      "brown.")
    assert "unresolved" not in cleaned
    assert "[" not in cleaned


def test_clean_summary_leaves_normal_text_untouched():
    assert _clean_summary("A tangy chutney made from tamarind.") == \
        "A tangy chutney made from tamarind."


def test_clean_summary_handles_none_and_empty():
    assert _clean_summary(None) is None
    assert _clean_summary("") == ""


def test_clean_summary_does_not_strip_an_unrelated_bracket():
    # only the specific "[unresolved ...]" audit-note shape is stripped -- a
    # legitimate bracketed aside in real content must survive untouched.
    text = "Payasam (also called kheer in the north) is a sweet pudding."
    assert _clean_summary(text) == text


def test_regional_comparison_strips_leaked_audit_notes_from_both_dishes():
    dish_a = {**_DISH_A, "content": {"summary": _DISH_A["content"]["summary"] +
                                    "  [unresolved primary_ingredients: fenugreek]"}}
    ex = serialize_regional_comparison(dish_a, _DISH_B, _rng())
    answer = ex["messages"][1]["content"]
    assert "unresolved" not in answer
    assert "[" not in answer


def test_regional_comparison():
    ex = serialize_regional_comparison(_DISH_A, _DISH_B, _rng())
    _assert_example_shape(ex, "regional_comparison", 2)
    a = ex["messages"][1]["content"]
    assert "pepper" in a and "red chili" in a and "tamarind" in a and "tomato" in a
    assert "Tamil Nadu" in a and "Andhra Pradesh" in a


def test_regional_comparison_caps_long_ingredient_lists():
    """Some dish records' primary_ingredients isn't curated down to a short
    headline set -- an 8-item difference shouldn't dump all 8 into the prose."""
    dish_long = {
        "id": "dish:long", "labels": ["Long Dish"],
        "content": {"summary": "x"},
        "structure": {"category": "kootu", "cuisine_state": "kerala",
                     "primary_ingredients": [f"ing:item-{i}" for i in range(8)]},
    }
    dish_empty = {
        "id": "dish:empty", "labels": ["Empty Dish"],
        "content": {"summary": "y"},
        "structure": {"category": "kootu", "cuisine_state": "tamil_nadu",
                     "primary_ingredients": []},
    }
    ex = serialize_regional_comparison(dish_long, dish_empty, _rng())
    a = ex["messages"][1]["content"]
    assert "and 3 more" in a
    assert "item-7" not in a


_TERM = {
    "id": "term:kootu",
    "structure": {"transliteration": "kootu"},
    "content": {"gloss": "A lentil-and-vegetable stew.", "usage_notes": "Thicker than sambar.",
               "example_sentence": "Amma made kootu for lunch."},
}


def test_term_definition():
    ex = serialize_term_definition(_TERM, _rng())
    _assert_example_shape(ex, "definition", 1)
    a = ex["messages"][1]["content"]
    assert "lentil-and-vegetable stew" in a
    assert "Amma made kootu" in a
    assert "kootu" in ex["messages"][0]["content"]


_ING = {
    "id": "ing:tamarind", "labels": ["Tamarind"],
    "content": {"summary": "A sour pulp used to flavour South Indian gravies.",
               "description": "Sold as a compressed block or paste."},
}


def test_ingredient_definition():
    ex = serialize_ingredient_definition(_ING, _rng())
    _assert_example_shape(ex, "definition", 1)
    a = ex["messages"][1]["content"]
    assert "sour pulp" in a
    assert "compressed block" in a


def test_abstention_examples_have_no_backing_records():
    examples = abstention_examples()
    assert len(examples) >= 10
    for ex in examples:
        _assert_example_shape(ex, "abstention", 0)
        assert ex["provenance"]["backing_record_ids"] == []


def test_abstention_examples_repeat_oversamples():
    base = abstention_examples()
    repeated = abstention_examples(repeat=8)
    assert len(repeated) == len(base) * 8
    # every underlying pair appears exactly `repeat` times, not just more rows
    # of a few pairs
    from collections import Counter
    counts = Counter(ex["messages"][0]["content"] for ex in repeated)
    assert set(counts.values()) == {8}
