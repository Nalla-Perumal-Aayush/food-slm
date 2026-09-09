"""Synthetic domain text from the canonical store (150M dataset plan, D-series).

Template-based, deterministic slot-filling — no LLM, so it runs alongside training.
LLM refinement / expansion is a later, GPU-window step.

  - :func:`phrasings` (D1)  — several prose renderings of a record, for the pretraining corpus
  - :func:`qa_pairs` (D3)   — {question, answer} pairs, for instruction SFT
  - :func:`abstention_items` (D6) — "is X real?" -> a hedged answer, from the forbidden list

Every item carries ``provenance`` (the source record id(s)) so synthetic data stays
auditable and can be held out per-record if needed.
"""
from __future__ import annotations

import random
from collections.abc import Iterator

from food_slm.schema.store import CanonicalStore


def _name(rec) -> str:
    return rec.labels[0]


_CAT_WORD = {  # ingredient/technique category -> a readable noun phrase
    "spice_whole": "whole spice", "spice_ground": "ground spice", "dal_legume": "lentil",
    "rice_grain": "rice", "leafy_green": "leafy green", "fat_oil": "cooking fat",
    "nut_seed": "nut or seed", "souring_agent": "souring agent", "tempering_agent": "tempering spice",
}


def _cat_phrase(cat: str) -> str:
    return _CAT_WORD.get(cat, cat.replace("_", " "))


def _state(rec) -> str:
    cs = getattr(rec.structure, "cuisine_state", None)
    return cs.replace("_", " ").title() if cs and cs != "pan_south" else "South India"


def _ing_names(store: CanonicalStore, ids) -> list[str]:
    return [_name(store.get(i)) for i in (ids or []) if store.get(i) is not None]


def _diet_phrase(diet) -> str:
    if diet.vegan:
        return "It is vegan."
    if diet.vegetarian:
        return "It is vegetarian." + (" It contains dairy." if diet.contains_dairy else "")
    tags = [k.replace("contains_", "").replace("_", " ")
            for k in ("contains_seafood", "contains_poultry", "contains_red_meat",
                      "contains_egg") if getattr(diet, k)]
    return "It is non-vegetarian" + (f" ({', '.join(tags)})." if tags else ".")


# --- D1: multi-phrasing (corpus prose) --------------------------------------

def _dish_phrasings(store: CanonicalStore, rec) -> Iterator[str]:
    s = rec.structure
    nm, st = _name(rec), _state(rec)
    cat = s.category.replace("_", " ")
    summ = (rec.content.summary or "").strip()
    ings = _ing_names(store, s.primary_ingredients)
    techs = [_name(store.get(t)) for t in (s.typical_techniques or []) if store.get(t)]
    ing_s = ", ".join(ings)
    diet = _diet_phrase(s.diet)

    if summ:
        yield f"{nm} is a {cat} from {st}. {summ} {diet}"
        yield f"{summ}\n{nm}, a {st} {cat}, {diet[3:].lower() if diet.startswith('It ') else diet}"
    if ings:
        yield (f"{nm} is made primarily with {ing_s}. It is a {cat} associated with {st}. "
               f"{diet}")
        yield f"To prepare {nm}, the key ingredients are {ing_s}."
    if techs:
        yield (f"{nm} ({st}) typically involves {', '.join(t.lower() for t in techs)}. "
               f"{summ}".strip())
    if s.diet.vegetarian and not s.diet.vegan:
        yield f"{nm} is suitable for vegetarians but not vegans, as it contains dairy."
    ctx = getattr(rec.content, "serving_context", None)
    if ctx:
        yield f"{nm} is usually served {ctx[0].lower()}{ctx[1:]}" if ctx else ""


def _technique_phrasings(rec) -> Iterator[str]:
    nm = _name(rec)
    c = rec.content
    fam = _cat_phrase(rec.structure.family)
    if c.summary:
        yield f"{nm} is a {fam} technique. {c.summary}"
    if getattr(c, "how_to", None):
        yield f"How to {nm.lower()}: {c.how_to}"
    if getattr(c, "sensory_cues", None):
        yield f"When you {nm.lower()}, {c.sensory_cues[0].lower()}{c.sensory_cues[1:]}"
    if getattr(c, "failure_modes", None):
        yield f"Common mistakes when doing {nm.lower()}: {c.failure_modes}"


def _ingredient_phrasings(rec) -> Iterator[str]:
    nm = _name(rec)
    c = rec.content
    cat = _cat_phrase(rec.structure.category)
    art = "an" if cat[:1] in "aeiou" else "a"
    if c.summary:
        yield f"{nm} is {art} {cat} used in South Indian cooking. {c.summary}"
    if getattr(c, "preparation_notes", None):
        yield f"To prepare {nm.lower()}: {c.preparation_notes}"
    subs = getattr(rec.structure, "common_substitutes", None)
    if subs:
        yield f"{nm} can sometimes be substituted, though the result differs."


def _term_phrasings(rec) -> Iterator[str]:
    s = rec.structure
    yield f"{s.transliteration} ({s.native}) means: {rec.content.gloss}"
    ex = getattr(rec.content, "example_sentence", None)
    if ex:
        yield f"{rec.content.gloss} For example: \"{ex}\""


def _problem_phrasings(rec) -> Iterator[str]:
    c = rec.content
    yield f"Problem: {c.description}" + (f" Cause: {c.diagnosis}" if c.diagnosis else "")
    for fix in (c.fixes or []):
        yield f"If {c.description[0].lower()}{c.description[1:]} — {fix}"


def phrasings(store: CanonicalStore, *, seed: int = 0) -> Iterator[dict]:
    """Yield {text, provenance, kind} for every renderable record."""
    rng = random.Random(seed)
    for rid, rec in store.records.items():
        if rec.version.status != "active":
            continue
        gen = {
            "dish": lambda r: _dish_phrasings(store, r),
            "technique": _technique_phrasings,
            "ingredient": _ingredient_phrasings,
            "term": _term_phrasings,
            "problem": _problem_phrasings,
        }.get(rec.type)
        if gen is None:
            continue
        variants = [t.strip() for t in gen(rec) if t and t.strip()]
        rng.shuffle(variants)
        for text in variants:
            yield {"text": text, "provenance": [rid], "kind": f"phrasing:{rec.type}"}


# --- D3: Q&A pairs (instruction SFT) --------------------------------------

def qa_pairs(store: CanonicalStore) -> Iterator[dict]:
    for rid, rec in store.records.items():
        if rec.version.status != "active":
            continue
        nm = _name(rec)
        if rec.type == "dish":
            s = rec.structure
            if rec.content.summary:
                yield _qa(f"What is {nm}?", rec.content.summary, rid)
            ings = _ing_names(store, s.primary_ingredients)
            if ings:
                yield _qa(f"What are the main ingredients in {nm}?",
                          f"The main ingredients in {nm} are {', '.join(ings)}.", rid)
            yield _qa(f"Is {nm} vegetarian?", _diet_phrase(s.diet), rid)
            if s.cuisine_state and s.cuisine_state != "pan_south":
                yield _qa(f"Which region is {nm} from?",
                          f"{nm} is from {_state(rec)}.", rid)
        elif rec.type == "technique" and rec.content.summary:
            yield _qa(f"What does it mean to {nm.lower()}?", rec.content.summary, rid)
            if getattr(rec.content, "how_to", None):
                yield _qa(f"How do you {nm.lower()}?", rec.content.how_to, rid)
        elif rec.type == "term":
            yield _qa(f"What does {rec.structure.transliteration} mean?",
                      rec.content.gloss, rid)
        elif rec.type == "problem":
            c = rec.content
            fixes = " ".join(c.fixes or [])
            if fixes:
                yield _qa(f"{c.description} What should I do?", fixes, rid)


def _qa(q: str, a: str, rid: str) -> dict:
    return {"question": q, "answer": a.strip(), "provenance": [rid], "kind": "qa"}


# --- D6: abstention (instruction SFT) -----------------------------------

_ABSTAIN = [
    "I'm not familiar with a dish called \"{name}\", so I can't describe it reliably.",
    "I don't have reliable information about \"{name}\". It may be a regional or family "
    "preparation I haven't seen.",
    "I can't confidently say what \"{name}\" is — I'd rather not guess.",
]


def abstention_items(forbidden) -> Iterator[dict]:
    for e in getattr(forbidden, "entities", []):
        for sf in e.surface_forms[:1]:
            for tmpl in _ABSTAIN:
                yield {"question": f"What is {sf}?", "answer": tmpl.format(name=sf),
                       "provenance": [e.id], "kind": "abstention"}
