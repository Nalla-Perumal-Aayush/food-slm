"""Normalise heterogeneous recipe datasets into one common row shape + a South-Indian
relevance tier (plan: 150M dataset, Level 1/2).

Each source dataset has its own columns; :func:`normalise` maps them to::

    {source, sid, title, cuisine_raw, ingredients: [str], instructions: [str],
     servings, total_min, url, tags: [str]}

:func:`relevance_tier` labels a normalised row ``core_si`` / ``pan_indian`` / ``general``
from the cuisine field, dish-name signal, and an ingredient signature. The scraped
sources are copyright-encumbered — used for weights, not redistribution (see the 150M
data plan); provenance is still recorded per row.
"""
from __future__ import annotations

import ast
import re

# --- source adapters -------------------------------------------------------

SOURCES: dict[str, dict] = {
    "food_com": {
        "hf": "AkashPS11/recipes_data_food.com", "streaming": True,
        "license": "research-only (food.com scrape)",
    },
    "archanas": {
        "hf": "Anupam007/indian-recipe-dataset", "streaming": False,
        "license": "research-only (archanaskitchen.com scrape)",
    },
    "epicurious": {
        "hf": "Hieu-Pham/kaggle_food_recipes", "streaming": False,
        "license": "research-only (epicurious scrape)",
    },
}


def _r_vector(v) -> list[str]:
    """Parse an R-style ``c("a", "b")`` string (food.com columns) into a list."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if not isinstance(v, str):
        return []
    s = v.strip()
    if s.startswith("c(") and s.endswith(")"):
        try:
            return [str(x).strip() for x in ast.literal_eval("[" + s[2:-1] + "]") if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip(' "') for p in s.split(",") if p.strip(' "')]


def _iso_minutes(s) -> int | None:
    """PT1H30M -> 90."""
    if not isinstance(s, str) or not s.startswith("PT"):
        return None
    h = re.search(r"(\d+)H", s)
    m = re.search(r"(\d+)M", s)
    total = (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)
    return total or None


def _split_steps(text) -> list[str]:
    if isinstance(text, list):
        return [str(x).strip() for x in text if str(x).strip()]
    if not isinstance(text, str):
        return []
    s = text.strip()
    if s.startswith("c(") and s.endswith(")"):
        return [p for p in _r_vector(s) if len(p) > 3]
    if s[:1] in "[(" and s[-1:] in "])":                 # python list-literal string
        try:
            return [str(x).strip() for x in ast.literal_eval(s) if len(str(x).strip()) > 3]
        except (ValueError, SyntaxError):
            pass
    parts = re.split(r"(?<=[.!?])\s+|\n+", s)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def normalise(source: str, row: dict) -> dict | None:
    """Map a raw dataset row to the common shape. Returns None for unusable rows."""
    if source == "food_com":
        title = (row.get("Name") or "").strip()
        ings = _r_vector(row.get("RecipeIngredientParts"))
        qtys = _r_vector(row.get("RecipeIngredientQuantities"))
        ings = [f"{q} {i}".strip() for q, i in zip(qtys, ings, strict=False)] or ings
        out = dict(
            source=source, sid=str(row.get("RecipeId")), title=title,
            cuisine_raw=str(row.get("RecipeCategory") or ""),
            ingredients=ings, instructions=_split_steps(row.get("RecipeInstructions")),
            servings=row.get("RecipeServings"), total_min=_iso_minutes(row.get("TotalTime")),
            url=f"https://www.food.com/recipe/{row.get('RecipeId')}",
            tags=_r_vector(row.get("Keywords")),
        )
    elif source == "archanas":
        out = dict(
            source=source, sid=(row.get("URL") or "").rsplit("/", 1)[-1],
            title=(row.get("TranslatedRecipeName") or "").strip(),
            cuisine_raw=str(row.get("Cuisine") or ""),
            ingredients=[p.strip() for p in re.split(r",(?![^(]*\))",
                         row.get("TranslatedIngredients") or "") if p.strip()],
            instructions=_split_steps(row.get("TranslatedInstructions")),
            servings=None, total_min=row.get("TotalTimeInMins"),
            url=row.get("URL") or "", tags=[],
        )
    elif source == "epicurious":
        out = dict(
            source=source, sid=str(row.get("Unnamed: 0") or row.get("Title", ""))[:64],
            title=(row.get("Title") or "").strip(), cuisine_raw="",
            ingredients=_split_steps(row.get("Cleaned_Ingredients") or row.get("Ingredients")),
            instructions=_split_steps(row.get("Instructions")),
            servings=None, total_min=None, url="", tags=[],
        )
    else:
        raise ValueError(f"unknown source {source!r}")

    if not out["title"] or len(out["ingredients"]) < 2 or len(out["instructions"]) < 1:
        return None
    return out


# --- South-Indian relevance ---------------------------------------------------

_SI_CUISINE = re.compile(
    r"south\s*indian|tamil|kerala|karnataka|andhra|telangana|chettinad|udupi|"
    r"mangalore|hyderabad|malabar|coorg|kongunad|rayalaseema|pondicherr|puducherr|konkan",
    re.I,
)
_PAN_INDIAN = re.compile(r"\bindian\b|north indian|mughlai|awadhi|maharashtrian|bengali|"
                         r"gujarati|rajasthani|goan|punjabi|kashmiri|bihari|oriya|assamese", re.I)

# dishes / preparations that are unambiguously South Indian
_SI_DISHES = re.compile(
    r"\b(idli|idly|dosa|dosai|thosai|uttapam|utthapam|vada|vadai|medu vada|pongal|"
    r"upma|uppuma|pesarattu|adai|appam|idiyappam|puttu|paniyaram|kuzhi paniyaram|"
    r"sambar|sambhar|rasam|saaru|charu|pulusu|kootu|poriyal|thoran|mezhukkupuratti|"
    r"avial|aviyal|olan|erissery|kaalan|pulissery|kuzhambu|kulambu|vatha kuzhambu|"
    r"mor kuzhambu|kara kuzhambu|puliyogare|puliyodarai|bisi bele|vangi bath|"
    r"curd rice|thayir sadam|lemon rice|coconut rice|ven pongal|sakkarai pongal|"
    r"payasam|pradhaman|mysore pak|obbattu|holige|bobbatlu|ariselu|adhirasam|athirasam|"
    r"kozhukattai|modak|unni appam|neyyappam|pootharekulu|dharwad|chettinad|"
    r"gongura|pachadi|thogayal|thuvaiyal|molagapodi|milagai podi|paruppu podi|"
    r"kothu parotta|kerala parotta|malabar parotta|meen curry|meen moilee|fish molee|"
    r"kori gassi|neer dosa|akki rotti|ragi mudde|jolada rotti|kadala curry)\b",
    re.I,
)
_SI_INGREDIENTS = (
    "curry leaves", "curry leaf", "urad dal", "urad daal", "black gram", "toor dal",
    "tuvar dal", "chana dal", "tamarind", "asafoetida", "asafetida", "hing", "perungayam",
    "sambar powder", "rasam powder", "coconut", "mustard seed", "kadugu", "jaggery",
    "kokum", "kudampuli", "drumstick", "curd", "buttermilk", "idli rice", "sona masuri",
    "ponni rice", "semolina", "rava", "sooji", "rice flour", "ragi", "gram flour",
    "green chilli", "red chilli", "fenugreek", "methi seed", "turmeric",
)


def relevance_tier(row: dict) -> tuple[str, list[str]]:
    """Return ``(tier, region_tags)``. tier in {core_si, pan_indian, general}."""
    hay = " ".join([row["title"], row["cuisine_raw"], " ".join(row.get("tags", [])),
                    " ".join(row["ingredients"])]).lower()
    regions: list[str] = []
    for pat, tag in [(r"tamil|chettinad|kongunad|madras", "tamil_nadu"),
                     (r"kerala|malabar|travancore|nasrani", "kerala"),
                     (r"karnataka|udupi|mangalore|coorg|dharwad|kannada", "karnataka"),
                     (r"andhra|rayalaseema|coastal andhra", "andhra_pradesh"),
                     (r"telangana|hyderabad", "telangana"),
                     (r"pondicherr|puducherr", "puducherry")]:
        if re.search(pat, hay):
            regions.append(tag)

    si_cuisine = bool(_SI_CUISINE.search(row["cuisine_raw"]))
    si_dish = bool(_SI_DISHES.search(row["title"])) or bool(_SI_DISHES.search(hay[:400]))
    si_ing_hits = sum(1 for s in _SI_INGREDIENTS if s in hay)

    if si_cuisine or si_dish or si_ing_hits >= 4:
        return "core_si", regions or (["pan_south"] if (si_dish or si_ing_hits >= 4) else [])
    if _PAN_INDIAN.search(row["cuisine_raw"]) or si_ing_hits >= 2:
        return "pan_indian", regions
    return "general", []
