"""Transform a normalised recipe row (recipe_sources.normalise) into:

  - :func:`to_prose`        — clean readable recipe text for the pretraining corpus (Level 1)
  - :func:`parse_ingredient`/:func:`classify_step` — structured pieces for later schema-v2
    ``recipe`` draft assembly (Level 2; DAG + dish_id resolution + panel validation are
    done separately, off the GPU-contended path)

Rules-first: no LLM call here, so this runs fine alongside a training job.
"""
from __future__ import annotations

import re
from fractions import Fraction

# --- quantities -------------------------------------------------------------

_UNIT_CANON = {
    "g": ("g", "mass", 1.0), "gram": ("g", "mass", 1.0), "grams": ("g", "mass", 1.0),
    "kg": ("kg", "mass", 1000.0), "mg": ("mg", "mass", 0.001),
    "ml": ("ml", "volume", 1.0), "l": ("l", "volume", 1000.0), "litre": ("l", "volume", 1000.0),
    "tsp": ("tsp", "volume", 5.0), "teaspoon": ("tsp", "volume", 5.0),
    "tbsp": ("tbsp", "volume", 15.0), "tablespoon": ("tbsp", "volume", 15.0),
    "cup": ("cup", "volume", 240.0), "cups": ("cup", "volume", 240.0),
    "clove": ("clove", "count", 1.0), "cloves": ("clove", "count", 1.0),
    "sprig": ("sprig", "count", 1.0), "sprigs": ("sprig", "count", 1.0),
    "piece": ("piece", "count", 1.0), "pieces": ("piece", "count", 1.0),
    "pinch": ("qualitative", "qualitative", None),
}
_UNICODE_FRAC = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
                 "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
_QTY_RE = re.compile(
    r"^\s*(?P<qty>\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)?\s*"
    r"(?P<unit>[a-zA-Z]+)?\.?\s*(?P<rest>.*)$")


def _to_float(qty: str | None) -> float | None:
    if not qty:
        return None
    qty = qty.strip()
    try:
        if " " in qty:
            whole, frac = qty.split()
            return float(whole) + float(Fraction(frac))
        return float(Fraction(qty)) if "/" in qty else float(qty)
    except (ValueError, ZeroDivisionError):
        return None


def parse_ingredient(line: str) -> dict:
    """{raw, quantity: {value, unit, value_canonical, unit_canonical, dimension} | None,
        name, prep_note}."""
    raw = line.strip()
    s = raw
    for u, a in _UNICODE_FRAC.items():
        s = s.replace(u, " " + a)
    s = re.sub(r"\s+", " ", s).strip()

    m = _QTY_RE.match(s)
    qty_obj = None
    name = s
    if m and (m.group("qty") or (m.group("unit") and m.group("unit").lower() in _UNIT_CANON)):
        val = _to_float(m.group("qty"))
        unit_raw = (m.group("unit") or "").lower()
        rest = m.group("rest").strip()
        if unit_raw in _UNIT_CANON:
            cu, dim, factor = _UNIT_CANON[unit_raw]
            qty_obj = {
                "original_text": raw, "value": val,
                "unit": cu if cu != "qualitative" else None,
                "value_canonical": (val * factor) if (val is not None and factor) else None,
                "unit_canonical": {"mass": "g", "volume": "ml", "count": "count"}.get(dim),
                "dimension": dim, "approximate": bool(re.search(r"about|approx|~|-|to ", raw, re.I)),
            }
            name = rest
        elif val is not None:
            qty_obj = {"original_text": raw, "value": val, "unit": "count",
                       "value_canonical": val, "unit_canonical": "count",
                       "dimension": "count", "approximate": False}
            name = (unit_raw + " " + rest).strip() if unit_raw else rest

    prep = ""
    pm = re.search(r",\s*(chopped|sliced|diced|minced|grated|crushed|ground|soaked|"
                   r"roasted|toasted|peeled|deseeded|cubed|julienned|halved|torn|"
                   r"finely chopped|thinly sliced|to taste)\b.*$", name, re.I)
    if pm:
        prep = pm.group(0).lstrip(", ").strip()
        name = name[:pm.start()].strip()
    name = re.sub(r"^\(.*?\)\s*", "", name).strip(" ,.-")
    return {"raw": raw, "quantity": qty_obj, "name": name.lower(), "prep_note": prep}


# --- steps -----------------------------------------------------------------

_VERB_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bsoak\b", re.I), "soak", "soak"),
    (re.compile(r"\b(drain|strain)\b", re.I), "drain", "prep"),
    (re.compile(r"\b(rinse|wash)\b", re.I), "rinse", "prep"),
    (re.compile(r"\b(grind|blend|make .*paste|pulse)\b", re.I), "grind", "grind"),
    (re.compile(r"\bknead\b", re.I), "knead", "prep"),
    (re.compile(r"\bmash\b", re.I), "mash", "cook"),
    (re.compile(r"\b(ferment|leave overnight|rest .*8 hours|allow .*rise)\b", re.I),
     "ferment", "ferment"),
    (re.compile(r"\b(temper|tadka|splutter|crackle|add .*mustard seeds)\b", re.I),
     "temper", "temper"),
    (re.compile(r"\b(deep[- ]fry)\b", re.I), "deep_fry", "cook"),
    (re.compile(r"\b(saute|sauté|stir[- ]fry|fry)\b", re.I), "saute", "cook"),
    (re.compile(r"\b(dry[- ]roast|roast .*until fragrant)\b", re.I), "dry_roast", "prep"),
    (re.compile(r"\broast\b", re.I), "roast", "cook"),
    (re.compile(r"\b(pressure[- ]cook|cooker|whistle)\b", re.I), "pressure_cook", "cook"),
    (re.compile(r"\bsteam\b", re.I), "steam", "cook"),
    (re.compile(r"\bboil\b", re.I), "boil", "cook"),
    (re.compile(r"\b(simmer|reduce|cook .*low heat|cook .*covered)\b", re.I), "simmer", "cook"),
    (re.compile(r"\b(sear|brown)\b", re.I), "sear", "cook"),
    (re.compile(r"\bmarinate\b", re.I), "marinate", "prep"),
    (re.compile(r"\b(mash)\b", re.I), "mash", "cook"),
    (re.compile(r"\b(garnish|sprinkle .*coriander|top with)\b", re.I), "garnish", "finish"),
    (re.compile(r"\b(cool|let it cool|set aside to cool)\b", re.I), "cool", "finish"),
    (re.compile(r"\bserve\b", re.I), "serve", "serve"),
    (re.compile(r"\b(season|add salt|salt to taste)\b", re.I), "season", "cook"),
    (re.compile(r"\b(mix|combine|stir|whisk|fold)\b", re.I), "mix", "assemble"),
    (re.compile(r"\b(add|pour|put in|transfer)\b", re.I), "add", "cook"),
    (re.compile(r"\bheat\b", re.I), "heat", "cook"),
]


def classify_step(text: str) -> dict:
    """{action_verb, phase, until_condition, duration_min}."""
    verb, phase = "combine", "assemble"
    for pat, v, p in _VERB_RULES:
        if pat.search(text):
            verb, phase = v, p
            break
    dm = re.search(r"(\d+)\s*(?:-|to|–)?\s*(\d+)?\s*(minute|min|hour|hr|second|sec)", text, re.I)
    dur = None
    if dm:
        val = int(dm.group(2) or dm.group(1))
        unit = dm.group(3).lower()
        dur = val * (60 if unit.startswith(("hour", "hr")) else 1 if unit.startswith("min")
                     else 1 / 60)
    um = re.search(r"until ([^.,;]{3,60})", text, re.I)
    return {"action_verb": verb, "phase": phase,
            "until_condition": um.group(1).strip() if um else None,
            "duration_min": round(dur, 1) if dur else None}


# --- prose (Level 1 corpus) ----------------------------------------------

def to_prose(row: dict) -> str:
    """Clean, readable recipe text for the pretraining corpus."""
    lines: list[str] = [row["title"].strip()]
    cui = re.sub(r"\s*recipes?\s*$", "", row.get("cuisine_raw", ""), flags=re.I).strip()
    art = "An" if cui[:1].lower() in "aeiou" else "A"
    reg = row.get("region_tags") or []
    if reg and reg != ["pan_south"]:
        lines.append(f"{art} {cui or 'South Indian'} recipe from "
                     f"{', '.join(r.replace('_', ' ').title() for r in reg)}.")
    elif cui:
        lines.append(f"{art} {cui} recipe.")
    if row.get("servings"):
        lines.append(f"Serves {row['servings']}.")
    if row.get("total_min"):
        lines.append(f"Total time: about {row['total_min']} minutes.")

    lines.append("\nIngredients:")
    for ing in row["ingredients"]:
        lines.append(f"- {ing.strip()}")

    lines.append("\nMethod:")
    for i, step in enumerate(row["instructions"], 1):
        lines.append(f"{i}. {step.strip().rstrip('.')}."
                     if not step.strip().endswith((".", "!", "?")) else f"{i}. {step.strip()}")
    return "\n".join(lines).strip()
