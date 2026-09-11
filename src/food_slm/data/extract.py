"""Page text -> draft canonical records (mcp_extraction_pipeline.md §2).

Two entry points:
  * :func:`build_extraction_prompt` — for **Claude Code** to run the extraction
    directly (highest quality, no separate model).
  * :func:`extract_records` — run it through any :class:`~food_slm.data.llm.Provider`
    (e.g. a local Ollama model) for scale.

Both produce the same shape: a list of **draft** schema-v2 record dicts with
provenance stamped and ``validation_status: "unverified"``. Nothing is committed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

from food_slm.data.llm import Provider, extract_json
from food_slm.schema.models import parse_record


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "unnamed"


_SYSTEM = """You extract STRUCTURED CULINARY FACTS from a web page about South Indian food.
Return ONLY JSON. Do not invent facts not supported by the text. If the page is not about
South Indian cooking, return {"dishes": [], "ingredients": [], "techniques": [], "terms": [],
"notes": "off-topic"}.

Rules:
- English names are the primary name. Put any native-script name in `names` as
  {lang: ta|ml|kn|te, native: "...", transliteration: "popular romanization"}.
- Never tag by community, caste or religion. Use region / cooking style only.
- `cuisine_state` is EXACTLY one of: tamil_nadu, kerala, karnataka, andhra_pradesh,
  telangana, puducherry, pan_south.
- For a dish, `diet` = {vegetarian, contains_dairy, contains_egg, contains_seafood,
  contains_poultry, contains_red_meat} booleans (vegetarian=false if any meat/egg).
- Dish `category` MUST be EXACTLY one of: tiffin, rice_dish, kuzhambu, sambar, rasam,
  kootu, poriyal, masiyal, thogayal, chutney, podi, pachadi, appalam_vadagam, sweet,
  snack, beverage, pickle, bread, festival_special, biryani, pulao, kurma,
  non_veg_curry, seafood_dish, egg_dish, thoran, avial, olan, mezhukkupuratti, palya,
  gojju, saaru, pulusu, pappu, vepudu, dry_curry, gravy_curry, stew, porridge, other.
  (breakfast/tiffin item -> "tiffin"; any curry with meat/fish/egg -> the matching
  non_veg_curry/seafood_dish/egg_dish; when unsure -> "other".)
- Ingredient `category` MUST be EXACTLY one of: dal_legume, rice_grain, flour,
  vegetable, leafy_green, fruit, spice_whole, spice_ground, herb, tempering_agent,
  souring_agent, sweetener, fat_oil, dairy, nut_seed, flavouring, leavening, salt,
  water, seafood, poultry, red_meat, egg, other.
- Ingredient `default_unit` MUST be EXACTLY one of: mg, g, kg, ml, l, tsp, tbsp, cup,
  count, clove, sprig, piece, sec, min, hr, C (singular, abbreviated — "cups"->"cup",
  "tablespoons"->"tbsp", "teaspoons"->"tsp", "grams"->"g").
- Technique `family` MUST be EXACTLY one of: heat_dry, heat_fat, heat_moist,
  mechanical, fermentation, soaking_hydration, tempering, resting, assembly,
  marination. (tempering/tadka -> "tempering"; frying -> "heat_fat"; boiling/steaming
  -> "heat_moist"; roasting/toasting dry -> "heat_dry"; grinding/mixing -> "mechanical".)
- Term `category` MUST be EXACTLY one of: ingredient, dish, technique, equipment,
  measure, general.

JSON shape:
{
 "dishes": [{"name_en","names":[],"category","cuisine_state","style"?,"diet":{...},
             "summary","primary_ingredients":["english name",...]}],
 "ingredients": [{"name_en","names":[],"category","default_unit","summary"}],
 "techniques": [{"name_en","names":[],"family","summary"}],
 "terms": [{"term_en"?,"lang","native","transliteration","gloss","category"}],
 "notes": "anything uncertain or needing human review"
}"""


def build_extraction_prompt(page_text: str, source_id: str, *, max_chars: int = 12000) -> tuple:
    user = (f"SOURCE: {source_id}\n\nPAGE TEXT (truncated):\n"
            f"{page_text[:max_chars]}\n\nExtract the structured facts as JSON.")
    return _SYSTEM, user


@dataclass
class ExtractResult:
    drafts: list[dict] = field(default_factory=list)
    invalid: list[tuple[dict, str]] = field(default_factory=list)
    notes: str = ""

    @property
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.drafts:
            out[d["type"]] = out.get(d["type"], 0) + 1
        return out


def _envelope(rid: str, rtype: str, labels: list[str], content: dict, structure: dict,
              source_id: str, acquired_on: str) -> dict:
    today = datetime.now(UTC).date().isoformat()
    return {
        "id": rid, "type": rtype, "schema_version": "2",
        "labels": labels or [rid.split(":", 1)[1].replace("-", " ").title()],
        "content": content, "structure": structure,
        "provenance": {"source_ids": [source_id], "transformations": ["extracted"],
                       "acquired_on": acquired_on, "synthetic": False},
        "quality": {"confidence": 0.55, "validation_status": "unverified",
                    "corroboration_count": 0},
        "version": {"revision": 1, "created_on": today, "updated_on": today, "status": "active"},
    }


_CUISINE_STATES = {"tamil_nadu", "kerala", "karnataka", "andhra_pradesh", "telangana",
                   "puducherry", "pan_south"}


def _canon_diet(raw: dict | None) -> dict:
    d = {"vegetarian": True, "vegan": False, "contains_dairy": False, "contains_egg": False,
         "contains_seafood": False, "contains_poultry": False, "contains_red_meat": False,
         **(raw or {})}
    has_meat_egg = (d["contains_egg"] or d["contains_seafood"] or d["contains_poultry"]
                    or d["contains_red_meat"])
    if has_meat_egg:
        d["vegetarian"] = False
        d["vegan"] = False
    if d["vegan"]:
        d["vegetarian"] = True
        d["contains_dairy"] = False
    return d


def _dish(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    cs = str(d.get("cuisine_state", "pan_south")).strip().lower().replace(" ", "_")
    st = {"category": _canon(d.get("category"), _DISH_CAT_SYN, "other"),
          "cuisine_state": cs if cs in _CUISINE_STATES else "pan_south",
          "diet": _canon_diet(d.get("diet")),
          "names": d.get("names", []),
          "primary_ingredients": [f"ing:{slugify(x)}" for x in d.get("primary_ingredients", [])]}
    if d.get("style"):
        st["style"] = d["style"]
    return _envelope(f"dish:{slugify(name)}", "dish", [name] + d.get("aliases", []),
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _ingredient(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    st = {"category": _canon(d.get("category"), _ING_CAT_SYN, "other"),
          "default_unit": _canon(d.get("default_unit"), _UNIT_SYN, "g"),
          "names": d.get("names", [])}
    return _envelope(f"ing:{slugify(name)}", "ingredient", [name],
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _technique(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    st = {"family": _canon(d.get("family"), _TECH_FAM_SYN, "assembly"),
          "names": d.get("names", [])}
    return _envelope(f"tech:{slugify(name)}", "technique", [name],
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _term(d: dict, source_id: str, acquired_on: str) -> dict:
    translit = d.get("transliteration", "")
    lang = str(d.get("lang", "ta")).strip().lower()
    st = {"lang": lang if lang in ("ta", "ml", "kn", "te") else "ta",
          "native": d.get("native", translit), "transliteration": translit,
          "category": _canon(d.get("category"), _TERM_CAT_SYN, "general")}
    return _envelope(f"term:{slugify(translit or d.get('term_en', ''))}", "term",
                     [d.get("term_en") or translit], {"gloss": d.get("gloss", translit)},
                     st, source_id, acquired_on)


_BUILDERS = {"dishes": _dish, "ingredients": _ingredient, "techniques": _technique,
             "terms": _term}


# --- enum canonicalization: local 7-9B models paraphrase enum values instead of
# copying them verbatim even when the prompt lists the closed set. Map common
# synonyms to the real schema-v2 enum before validation, rather than reject the
# whole record over one free-text field. Unmapped -> the type's safe fallback.

def _canon(value: str | None, synonyms: dict[str, str], fallback: str) -> str:
    if not value:
        return fallback
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in synonyms:
        return synonyms[v]
    if v.endswith("s") and v[:-1] in synonyms:      # crude depluralisation
        return synonyms[v[:-1]]
    return v if v in set(synonyms.values()) else fallback


_DISH_CAT_SYN = {c: c for c in (
    "tiffin", "rice_dish", "kuzhambu", "sambar", "rasam", "kootu", "poriyal",
    "masiyal", "thogayal", "chutney", "podi", "pachadi", "appalam_vadagam", "sweet",
    "snack", "beverage", "pickle", "bread", "festival_special", "biryani", "pulao",
    "kurma", "non_veg_curry", "seafood_dish", "egg_dish", "thoran", "avial", "olan",
    "mezhukkupuratti", "palya", "gojju", "saaru", "pulusu", "pappu", "vepudu",
    "dry_curry", "gravy_curry", "stew", "porridge", "other")} | {
    "breakfast": "tiffin", "tiffin_item": "tiffin", "steamed_cake": "tiffin",
    "crepe": "tiffin", "pancake": "tiffin", "dosa": "tiffin", "idli": "tiffin",
    "rice": "rice_dish", "curry": "gravy_curry", "gravy": "gravy_curry",
    "vegetable_curry": "gravy_curry", "dessert": "sweet", "sweets": "sweet",
    "candy": "sweet", "drink": "beverage", "beverages": "beverage", "soup": "rasam",
    "lentil_soup": "sambar", "dal": "sambar", "stir_fry": "poriyal",
    "side_dish": "poriyal", "salad": "pachadi", "spice_mix": "podi",
    "spice_powder": "podi", "condiment": "chutney", "relish": "chutney",
    "flatbread": "bread", "roti": "bread", "meat_curry": "non_veg_curry",
    "chicken_curry": "non_veg_curry", "mutton_curry": "non_veg_curry",
    "fish_curry": "seafood_dish", "fish_dish": "seafood_dish", "egg_curry": "egg_dish",
    "festival": "festival_special", "porridge_dish": "porridge",
}
_ING_CAT_SYN = {c: c for c in (
    "dal_legume", "rice_grain", "flour", "vegetable", "leafy_green", "fruit",
    "spice_whole", "spice_ground", "herb", "tempering_agent", "souring_agent",
    "sweetener", "fat_oil", "dairy", "nut_seed", "flavouring", "leavening", "salt",
    "water", "seafood", "poultry", "red_meat", "egg", "other")} | {
    "legume": "dal_legume", "legumes": "dal_legume", "lentil": "dal_legume",
    "lentils": "dal_legume", "pulse": "dal_legume", "bean": "dal_legume",
    "grain": "rice_grain", "grains": "rice_grain", "rice": "rice_grain",
    "cereal": "rice_grain", "root": "vegetable", "root_vegetable": "vegetable",
    "tuber": "vegetable", "veggie": "vegetable", "greens": "leafy_green",
    "green": "leafy_green", "leaf": "herb", "leaves": "herb", "fruits": "fruit",
    "spice": "spice_whole", "spices": "spice_whole", "seed": "spice_whole",
    "seeds": "spice_whole", "ground_spice": "spice_ground", "powder": "spice_ground",
    "herbs": "herb", "fat": "fat_oil", "oil": "fat_oil", "oils": "fat_oil",
    "ghee": "fat_oil", "dairy_product": "dairy", "milk": "dairy",
    "nut": "nut_seed", "nuts": "nut_seed", "sugar": "sweetener", "jaggery": "sweetener",
    "meat": "red_meat", "chicken": "poultry", "fish": "seafood", "eggs": "egg",
    "tamarind": "souring_agent", "sour_agent": "souring_agent",
}
_UNIT_SYN = {u: u for u in (
    "mg", "g", "kg", "ml", "l", "tsp", "tbsp", "cup", "count", "clove", "sprig",
    "piece", "sec", "min", "hr", "C")} | {
    "gram": "g", "grams": "g", "gm": "g", "gms": "g", "kilogram": "kg",
    "kilograms": "kg", "milligram": "mg", "milliliter": "ml", "millilitre": "ml",
    "liter": "l", "litre": "l", "litres": "l", "teaspoon": "tsp", "teaspoons": "tsp",
    "tablespoon": "tbsp", "tablespoons": "tbsp", "cups": "cup", "cloves": "clove",
    "sprigs": "sprig", "pieces": "piece", "seconds": "sec", "second": "sec",
    "minute": "min", "minutes": "min", "hour": "hr", "hours": "hr",
    "number": "count", "counts": "count", "nos": "count", "pinch": "count",
}
_TECH_FAM_SYN = {f: f for f in (
    "heat_dry", "heat_fat", "heat_moist", "mechanical", "fermentation",
    "soaking_hydration", "tempering", "resting", "assembly", "marination")} | {
    "cooking": "assembly", "frying": "heat_fat", "fry": "heat_fat",
    "deep_frying": "heat_fat", "shallow_frying": "heat_fat", "sauteing": "heat_fat",
    "roasting": "heat_dry", "toasting": "heat_dry", "grilling": "heat_dry",
    "baking": "heat_dry", "boiling": "heat_moist", "steaming": "heat_moist",
    "simmering": "heat_moist", "pressure_cooking": "heat_moist",
    "grinding": "mechanical", "mixing": "mechanical", "chopping": "mechanical",
    "blending": "mechanical", "tadka": "tempering", "seasoning": "tempering",
    "fermenting": "fermentation", "soaking": "soaking_hydration",
    "marinating": "marination", "resting_time": "resting",
}
_TERM_CAT_SYN = {c: c for c in
                ("ingredient", "dish", "technique", "equipment", "measure", "general")} | {
    "name": "general", "food": "dish", "cooking": "technique", "utensil": "equipment",
    "unit": "measure", "quantity": "measure",
}


def parse_extraction(obj: dict, source_id: str, *, acquired_on: str | None = None) -> ExtractResult:
    acquired_on = acquired_on or datetime.now(UTC).date().isoformat()
    res = ExtractResult(notes=str(obj.get("notes", "")))
    seen: set[str] = set()
    for key, build in _BUILDERS.items():
        for raw in obj.get(key, []) or []:
            try:
                rec = build(raw, source_id, acquired_on)
            except Exception as exc:  # noqa: BLE001
                res.invalid.append((raw, f"build: {exc}"))
                continue
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            try:
                parse_record(rec)              # schema-v2 validity gate
                res.drafts.append(rec)
            except Exception as exc:  # noqa: BLE001
                res.invalid.append((rec, f"schema: {type(exc).__name__}: {exc}"))
    return res


def extract_records(page_text: str, source_id: str, provider: Provider, *,
                    acquired_on: str | None = None, temperature: float = 0.0) -> ExtractResult:
    system, user = build_extraction_prompt(page_text, source_id)
    raw = provider.complete(system, user, temperature=temperature, json_out=True)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return ExtractResult(notes=f"unparseable model output: {raw[:200]!r}")
    return parse_extraction(obj, source_id, acquired_on=acquired_on)
