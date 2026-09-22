"""Page text -> draft `recipe` records (schema-v2 RecipeStructure).

Separate from extract.py (dish/ingredient/technique/term) on purpose: recipe is a
much richer nested schema (step DAG, typed Measurement per ingredient/duration,
declared vs. used equipment) and this file is new/unproven, so it doesn't touch
the already-working flat-entity extraction at all.

Same philosophy as extract.py though: ask the model for a *simple, forgiving*
JSON shape (ingredient quantities as plain text, steps as an ordered list with a
rough tag each), then let deterministic Python build the rigid schema around it --
step_id/order/depends_on as a linear chain, quantity-string parsing into a typed
Measurement, and free-text enum values coerced via synonym maps. A 7-9B local
model reliably gets the loose shape right; it does not reliably construct a
self-consistent step-dependency graph or a schema-exact Measurement object, so
that work happens here instead of in the prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from food_slm.data.extract import slugify
from food_slm.data.llm import Provider, extract_json
from food_slm.schema.models import parse_record

_SYSTEM = """You extract a STRUCTURED RECIPE for ONE South Indian dish from a web page.
Return ONLY JSON. Do not invent steps/ingredients not supported by the text. If the
page has no actual recipe (steps + ingredients) for the named dish, return
{"recipes": [], "notes": "no recipe found"}.

Rules:
- Extract exactly one recipe object for the dish named in the prompt (its main/first
  version if the page shows variants).
- `ingredients`: one entry per ingredient AS WRITTEN in the recipe's ingredient list,
  with its quantity as plain text exactly as the page states it (e.g. "2 cups",
  "1 tsp", "a pinch", "to taste", "2-3", "a handful").
- `steps`: the COOKING METHOD only (not the ingredient list), in the exact order
  given, one step per distinct action. Keep each `action` to one sentence.
- `phase` per step MUST be EXACTLY one of: prep, soak, grind, ferment, temper, cook,
  assemble, finish, rest, serve.
- `action_verb` per step MUST be EXACTLY one of: soak, drain, rinse, grind, blend,
  mix, knead, ferment, rest, heat, temper, saute, fry, deep_fry, roast, dry_roast,
  boil, simmer, pressure_cook, steam, add, combine, mash, season, garnish, cool,
  serve, reduce, strain, cover, flip, marinate, sear, braise, grill.
- `heat_level` (only if the step involves heat) MUST be EXACTLY one of: low,
  medium-low, medium, medium-high, high.
- `role` per ingredient (its role IN THIS RECIPE) MUST be EXACTLY one of: base,
  thickener, souring, tempering, aromatic, sweetening, binding, garnish, fat,
  leavening, colouring, bulking, protein, marinade.
- `form` per ingredient (its physical form as used) MUST be EXACTLY one of: whole,
  split, skinned, ground, paste, roasted, soaked, sprouted, fresh, dried, powder,
  flattened, minced, cubed, sliced, cleaned. Omit if not stated/obvious.
- `difficulty` MUST be EXACTLY one of: easy, moderate, advanced.
- Never tag by community, caste, or religion. Use region / cooking style only.

JSON shape:
{
 "recipes": [{
   "variant_label": "e.g. 'Hotel-style' or null",
   "style": "e.g. 'chettinad' or null",
   "servings": 4,
   "total_time_minutes": 45,
   "active_time_minutes": 20,
   "difficulty": "easy|moderate|advanced",
   "ingredients": [{"name","quantity_text","form"?,"role"?,"optional"?:false,"group"?}],
   "equipment": ["idli stand", ...],
   "steps": [{"action","phase","action_verb","ingredients_used":["name",...],
              "equipment"?,"technique"?,"heat_level"?,"duration_minutes"?,
              "temperature_c"?,"doneness_cue"?}],
   "tips": ["..."],
   "common_mistakes": ["..."],
   "serving_suggestions": "..." or null
 }],
 "notes": "anything uncertain or needing human review"
}"""


def build_recipe_extraction_prompt(page_text: str, dish_name: str, source_id: str,
                                   *, max_chars: int = 14000) -> tuple[str, str]:
    user = (f"DISH: {dish_name}\nSOURCE: {source_id}\n\nPAGE TEXT (truncated):\n"
            f"{page_text[:max_chars]}\n\nExtract the recipe as JSON.")
    return _SYSTEM, user


@dataclass
class RecipeExtractResult:
    drafts: list[dict] = field(default_factory=list)
    invalid: list[tuple[dict, str]] = field(default_factory=list)
    notes: str = ""


# --- enum coercion (extract.py's proven pattern: map free-text model output to
# the closed schema-v2 enum, safe fallback rather than reject the whole record) ---

def _canon(value, synonyms: dict[str, str], fallback: str | None) -> str | None:
    if not value:
        return fallback
    v = str(value).strip().lower().replace(" ", "_").replace("-", "-")
    v_us = v.replace("-", "_")
    if v in synonyms:
        return synonyms[v]
    if v_us in synonyms:
        return synonyms[v_us]
    allowed = set(synonyms.values())
    return v if v in allowed else fallback


_PHASE_SYN = {p: p for p in (
    "prep", "soak", "grind", "ferment", "temper", "cook", "assemble", "finish",
    "rest", "serve")} | {
    "preparation": "prep", "grinding": "grind", "fermentation": "ferment",
    "tempering": "temper", "cooking": "cook", "assembly": "assemble",
    "finishing": "finish", "resting": "rest", "serving": "serve",
}
_ACTION_VERB_SYN = {a: a for a in (
    "soak", "drain", "rinse", "grind", "blend", "mix", "knead", "ferment", "rest",
    "heat", "temper", "saute", "fry", "deep_fry", "roast", "dry_roast", "boil",
    "simmer", "pressure_cook", "steam", "add", "combine", "mash", "season",
    "garnish", "cool", "serve", "reduce", "strain", "cover", "flip", "marinate",
    "sear", "braise", "grill")} | {
    "sauté": "saute", "sauteing": "saute", "sautee": "saute", "frying": "fry",
    "deep_frying": "deep_fry", "shallow_fry": "saute", "roasting": "roast",
    "dry_roasting": "dry_roast", "toast": "dry_roast", "toasting": "dry_roast",
    "boiling": "boil", "simmering": "simmer", "steaming": "steam",
    "pressure_cooking": "pressure_cook", "cook": "add", "mixing": "mix",
    "grinding": "grind", "combining": "combine", "mashing": "mash",
    "seasoning": "season", "garnishing": "garnish", "cooling": "cool",
    "reducing": "reduce", "straining": "strain", "covering": "cover",
    "flipping": "flip", "marinating": "marinate", "searing": "sear",
    "braising": "braise", "grilling": "grill", "chop": "add", "chopping": "add",
    "tempering": "temper", "cut": "add",
}
_HEAT_SYN = {h: h for h in ("low", "medium-low", "medium", "medium-high", "high")} | {
    "medium_low": "medium-low", "medium_high": "medium-high", "med": "medium",
    "low_heat": "low", "high_heat": "high", "medium_heat": "medium",
    "simmer": "low", "gentle": "low", "moderate": "medium",
}
_ROLE_SYN = {r: r for r in (
    "base", "thickener", "souring", "tempering", "aromatic", "sweetening",
    "binding", "garnish", "fat", "leavening", "colouring", "bulking", "protein",
    "marinade")} | {
    "main": "base", "primary": "base", "thickening": "thickener", "sour": "souring",
    "souring_agent": "souring", "temper": "tempering", "flavouring": "aromatic",
    "aroma": "aromatic", "sweetener": "sweetening", "sweet": "sweetening",
    "binder": "binding", "garnishing": "garnish", "fat_oil": "fat", "oil": "fat",
    "leavener": "leavening", "colour": "colouring", "color": "colouring",
    "coloring": "colouring", "bulk": "bulking", "meat": "protein",
    "marinating": "marinade",
}
_FORM_SYN = {f: f for f in (
    "whole", "split", "skinned", "ground", "paste", "roasted", "soaked", "sprouted",
    "fresh", "dried", "powder", "flattened", "minced", "cubed", "sliced",
    "cleaned")} | {
    "powdered": "powder", "ground_up": "ground", "chopped": "cubed", "diced": "cubed",
    "grated": "ground", "washed": "cleaned",
}
_DIFFICULTY_SYN = {d: d for d in ("easy", "moderate", "advanced")} | {
    "simple": "easy", "beginner": "easy", "medium": "moderate", "intermediate": "moderate",
    "hard": "advanced", "difficult": "advanced", "expert": "advanced",
}

_QUALITATIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bto\s*taste\b", re.I), "to_taste"),
    (re.compile(r"\ba\s*pinch\b|\bpinch\s*of\b", re.I), "a_pinch"),
    (re.compile(r"\ba\s*few\b|\bfew\b", re.I), "a_few"),
    (re.compile(r"\bhandful\b", re.I), "handful"),
    (re.compile(r"\bas\s*needed\b|\bas\s*required\b", re.I), "as_needed"),
    (re.compile(r"\bfor\s*garnish\b|\bgarnish\b", re.I), "garnish"),
]
_UNIT_WORDS = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "g": "g", "gram": "g", "grams": "g", "gm": "g", "gms": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml",
    "l": "l", "liter": "l", "litre": "l", "litres": "l",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "cup": "cup", "cups": "cup",
    "clove": "clove", "cloves": "clove",
    "sprig": "sprig", "sprigs": "sprig",
    "piece": "piece", "pieces": "piece", "pcs": "piece",
    "sec": "sec", "second": "sec", "seconds": "sec",
    "min": "min", "minute": "min", "minutes": "min", "mins": "min",
    "hr": "hr", "hour": "hr", "hours": "hr", "hrs": "hr",
}
_MASS = {"mg", "g", "kg"}
_VOLUME = {"ml", "l", "tsp", "tbsp", "cup"}
_COUNT = {"count", "clove", "sprig", "piece"}
_TIME = {"sec", "min", "hr"}


def _dimension_for_unit(unit: str) -> str:
    if unit in _MASS:
        return "mass"
    if unit in _VOLUME:
        return "volume"
    if unit in _TIME:
        return "time"
    return "count"


# Fractional/mixed quantities ("1/2 cup", "1 1/2 kg") are the normal way recipe
# ingredient lists write amounts -- more common than plain decimals -- so this has
# to be tried before the bare-integer case, not as an afterthought.
_NUM = r"\d+\s+\d/\d|\d/\d|\d+(?:\.\d+)?"


def _to_float(s: str) -> float:
    s = s.strip()
    if " " in s:
        whole, frac = s.split(None, 1)
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


def parse_measurement(text: str | float | int | None) -> dict | None:
    """Plain-text quantity ("2 cups", "a pinch", "1-2 tsp", "500") -> a schema-v2
    Measurement dict. None if there's nothing usable to parse."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return {"original_text": str(text), "value": float(text), "unit": "count",
                "value_canonical": float(text), "unit_canonical": "count",
                "dimension": "count", "approximate": False}
    s = str(text).strip()
    if not s:
        return None
    for pattern, qual in _QUALITATIVE_PATTERNS:
        if pattern.search(s):
            return {"original_text": s, "dimension": "qualitative", "qualitative": qual,
                    "approximate": True}
    # range: "2-3", "2 to 3"; numbers may be plain, fractional ("1/2"), or mixed ("1 1/2")
    m_range = re.search(rf"({_NUM})\s*(?:-|to)\s*({_NUM})\s*([a-zA-Z]+)?", s)
    m_single = re.search(rf"({_NUM})\s*([a-zA-Z]+)?", s)
    if m_range:
        lo, hi, unit_word = m_range.groups()
        unit = _UNIT_WORDS.get((unit_word or "").lower().strip("."), "count")
        lo_v, hi_v = _to_float(lo), _to_float(hi)
        mid = (lo_v + hi_v) / 2
        return {"original_text": s, "value": mid, "unit": unit,
                "value_canonical": mid, "unit_canonical": unit,
                "dimension": _dimension_for_unit(unit), "approximate": True,
                "range": [lo_v, hi_v]}
    if m_single:
        val, unit_word = m_single.groups()
        unit = _UNIT_WORDS.get((unit_word or "").lower().strip("."), "count")
        v = _to_float(val)
        return {"original_text": s, "value": v, "unit": unit,
                "value_canonical": v, "unit_canonical": unit,
                "dimension": _dimension_for_unit(unit), "approximate": False}
    # no number at all -- treat as an "as needed"-style qualitative catch-all
    return {"original_text": s, "dimension": "qualitative", "qualitative": "as_needed",
            "approximate": True}


def _envelope(rid: str, labels: list[str], content: dict, structure: dict,
             source_id: str, acquired_on: str) -> dict:
    today = datetime.now(UTC).date().isoformat()
    return {
        "id": rid, "type": "recipe", "schema_version": "2",
        "labels": labels or [rid.split(":", 1)[1].replace("-", " ").title()],
        "content": content, "structure": structure,
        "provenance": {"source_ids": [source_id], "transformations": ["extracted"],
                       "acquired_on": acquired_on, "synthetic": False},
        "quality": {"confidence": 0.55, "validation_status": "unverified",
                    "corroboration_count": 0},
        "version": {"revision": 1, "created_on": today, "updated_on": today, "status": "active"},
    }


def _dish_diet_to_profile(dish_diet: dict) -> dict:
    """`dish_list_v1.yaml`'s `diet: {vegetarian, vegan, non_veg, contains: [...]}`
    -> schema-v2 DietProfile dict. The dish list is the authority on this (it's
    hand-authored, one entry per dish), not the LLM -- diet is a property of the
    dish itself, not something worth re-deriving per recipe from free-text page
    content."""
    contains = set(dish_diet.get("contains") or [])
    return {
        "vegetarian": bool(dish_diet.get("vegetarian", True)),
        "vegan": bool(dish_diet.get("vegan", False)),
        "contains_dairy": "dairy" in contains,
        "contains_egg": "egg" in contains,
        "contains_seafood": "seafood" in contains,
        "contains_poultry": "poultry" in contains,
        "contains_red_meat": "red_meat" in contains or "beef" in contains,
    }


def _build_recipe(raw: dict, dish_id: str, cuisine_state: str, source_id: str,
                  acquired_on: str, dish_diet: dict) -> dict:
    ings_raw = raw.get("ingredients") or []
    if not ings_raw:
        raise ValueError("no ingredients")
    ingredients = []
    name_to_ing_id: dict[str, str] = {}
    for i in ings_raw:
        name = (i.get("name") or "").strip()
        if not name:
            continue
        iid = f"ing:{slugify(name)}"
        name_to_ing_id[name.lower()] = iid
        entry = {"ingredient_id": iid, "optional": bool(i.get("optional", False)),
                 "substitutable": True}
        q = parse_measurement(i.get("quantity_text"))
        if q:
            entry["quantity"] = q
        form = _canon(i.get("form"), _FORM_SYN, None)
        if form:
            entry["form"] = form
        role = _canon(i.get("role"), _ROLE_SYN, None)
        if role:
            entry["role_in_recipe"] = role
        if i.get("group"):
            entry["group"] = str(i["group"])
        ingredients.append(entry)
    if not ingredients:
        raise ValueError("no usable ingredients after parsing")

    steps_raw = raw.get("steps") or []
    if not steps_raw:
        raise ValueError("no steps")
    equipment_ids: set[str] = set()
    if raw.get("equipment"):
        equipment_ids.update(f"equip:{slugify(e)}" for e in raw["equipment"] if e)
    steps = []
    prev_id = None
    for idx, s in enumerate(steps_raw, start=1):
        action = (s.get("action") or "").strip()
        if not action:
            continue
        step_id = f"step-{idx}"
        input_ids = []
        for used in (s.get("ingredients_used") or []):
            iid = name_to_ing_id.get(str(used).strip().lower())
            if iid is None:
                iid = f"ing:{slugify(str(used))}"
            input_ids.append(iid)
        step = {
            "step_id": step_id, "order": idx,
            "phase": _canon(s.get("phase"), _PHASE_SYN, "cook"),
            "action": action,
            "action_verb": _canon(s.get("action_verb"), _ACTION_VERB_SYN, "add"),
            "input_ingredient_ids": input_ids,
            "depends_on": [prev_id] if prev_id else [],
        }
        eq = s.get("equipment")
        if eq:
            eid = f"equip:{slugify(eq)}"
            step["equipment_id"] = eid
            equipment_ids.add(eid)
        tech = s.get("technique")
        if tech:
            step["technique_id"] = f"tech:{slugify(tech)}"
        params = {}
        hl = _canon(s.get("heat_level"), _HEAT_SYN, None)
        if hl:
            params["heat_level"] = hl
        if s.get("duration_minutes") is not None:
            params["duration"] = parse_measurement(f"{s['duration_minutes']} min")
        if s.get("temperature_c") is not None:
            params["temperature"] = parse_measurement(f"{s['temperature_c']} C")
        if s.get("doneness_cue"):
            params["doneness_cue"] = str(s["doneness_cue"])
        if params:
            step["params"] = params
        steps.append(step)
        prev_id = step_id
    if not steps:
        raise ValueError("no usable steps after parsing")

    structure = {
        "dish_id": dish_id, "cuisine_state": cuisine_state,
        "diet": _dish_diet_to_profile(dish_diet),
        "ingredients": ingredients, "equipment_ids": sorted(equipment_ids), "steps": steps,
        "scalable": True, "make_ahead": False,
    }
    if raw.get("variant_label"):
        structure["variant_label"] = str(raw["variant_label"])
    if raw.get("style"):
        structure["style"] = str(raw["style"])
    if raw.get("servings") is not None:
        structure["yield"] = {"servings": int(raw["servings"])}
    if raw.get("total_time_minutes") is not None:
        structure["total_time"] = parse_measurement(f"{raw['total_time_minutes']} min")
    if raw.get("active_time_minutes") is not None:
        structure["active_time"] = parse_measurement(f"{raw['active_time_minutes']} min")
    diff = _canon(raw.get("difficulty"), _DIFFICULTY_SYN, None)
    if diff:
        structure["difficulty"] = diff

    content = {
        "summary": f"Recipe for {dish_id.split(':', 1)[1].replace('-', ' ')}"
                  f"{' (' + structure['variant_label'] + ')' if structure.get('variant_label') else ''}.",
        "tips": [str(t) for t in (raw.get("tips") or [])],
        "common_mistakes": [str(t) for t in (raw.get("common_mistakes") or [])],
    }
    if raw.get("serving_suggestions"):
        content["serving_suggestions"] = str(raw["serving_suggestions"])

    source_slug = source_id.split(":", 1)[1] if ":" in source_id else slugify(source_id)
    rid_parts = [dish_id.split(":", 1)[1], source_slug]
    if raw.get("variant_label"):
        rid_parts.insert(1, slugify(raw["variant_label"]))
    rid = f"recipe:{'-'.join(rid_parts)}"[:200]
    labels = [dish_id.split(":", 1)[1].replace("-", " ").title()]
    return _envelope(rid, labels, content, structure, source_id, acquired_on)


def parse_recipe_extraction(obj: dict, dish_id: str, cuisine_state: str, source_id: str,
                            dish_diet: dict, *, acquired_on: str | None = None) -> RecipeExtractResult:
    acquired_on = acquired_on or datetime.now(UTC).date().isoformat()
    res = RecipeExtractResult(notes=str(obj.get("notes", "")))
    for raw in obj.get("recipes", []) or []:
        try:
            rec = _build_recipe(raw, dish_id, cuisine_state, source_id, acquired_on, dish_diet)
        except Exception as exc:  # noqa: BLE001
            res.invalid.append((raw, f"build: {exc}"))
            continue
        try:
            parse_record(rec)
            res.drafts.append(rec)
        except Exception as exc:  # noqa: BLE001
            res.invalid.append((rec, f"schema: {type(exc).__name__}: {exc}"))
    return res


def extract_recipe(page_text: str, dish_id: str, dish_name: str, cuisine_state: str,
                   source_id: str, dish_diet: dict, provider: Provider, *,
                   acquired_on: str | None = None, temperature: float = 0.0) -> RecipeExtractResult:
    system, user = build_recipe_extraction_prompt(page_text, dish_name, source_id)
    raw = provider.complete(system, user, temperature=temperature, json_out=True)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return RecipeExtractResult(notes=f"unparseable model output: {raw[:200]!r}")
    return parse_recipe_extraction(obj, dish_id, cuisine_state, source_id, dish_diet,
                                   acquired_on=acquired_on)
