"""Canonical store -> training text (PoC-minimal).

Turns canonical records into plain-text / lightly-structured documents for the 15M
PoC: enough real culinary vocabulary + structural-token usage to train and exercise
the tokenizer and packer. The full Phase 8/9 serialization (curriculum-ordered,
multi-phrasing, instruction formats) replaces this.

Native scripts appear only as ``English Name (transliteration, «native»)`` on first
mention (baseline §6). Structural tokens wrap the English span.
"""
from __future__ import annotations

from collections.abc import Iterator

from food_slm.schema.store import CanonicalStore


def _primary_name(rec) -> str:
    return rec.labels[0]


def _native_gloss(rec) -> str:
    """`(transliteration, «native»)` for the primary local name, else ''."""
    names = getattr(rec.structure, "names", None) or []
    prim = next((n for n in names if n.primary), names[0] if names else None)
    if prim is None:
        return ""
    return f" ({prim.transliteration}, {prim.native})"


def _measure_text(m) -> str:
    if m is None:
        return ""
    if m.dimension == "qualitative":
        return m.qualitative.replace("_", " ")
    if m.value == int(m.value):
        return f"{int(m.value)} {m.unit or ''}".strip()
    return f"{m.value:g} {m.unit or ''}".strip()


def _dish_doc(store: CanonicalStore, rec) -> str:
    s = rec.structure
    lines = [f"<DISH>{_primary_name(rec)}{_native_gloss(rec)}</DISH> is a "
             f"{s.category.replace('_', ' ')} from {s.cuisine_state.replace('_', ' ').title()}."]
    if rec.content.summary:
        lines.append(rec.content.summary)
    if rec.content.description:
        lines.append(rec.content.description)
    ings = [store.get(i) for i in s.primary_ingredients]
    ing_names = [_primary_name(i) for i in ings if i is not None]
    if ing_names:
        lines.append("Main ingredients: " + ", ".join(ing_names) + ".")
    diet = s.diet
    if diet.vegetarian:
        lines.append("Vegan." if diet.vegan else "Vegetarian.")
    else:
        tags = [k.replace("contains_", "").replace("_", " ")
                for k, v in diet.model_dump().items() if v]
        lines.append("Non-vegetarian" + (f" ({', '.join(tags)})." if tags else "."))
    return "\n".join(lines)


def _recipe_doc(store: CanonicalStore, rec) -> str:
    s = rec.structure
    dish = store.get(s.dish_id)
    title = _primary_name(dish) if dish else s.dish_id
    out = [f"<DISH>{title}</DISH> — {s.variant_label or 'recipe'}"]
    if rec.content.summary:
        out.append(rec.content.summary)
    y = s.yield_
    if y and y.servings:
        out.append(f"Serves {y.servings}.")
    out.append("Ingredients:")
    for ri in s.ingredients:
        ing = store.get(ri.ingredient_id)
        name = _primary_name(ing) if ing else ri.ingredient_id
        qty = _measure_text(ri.quantity)
        prep = f", {ri.preparation}" if ri.preparation else ""
        out.append(f"<INGREDIENT><QTY>{qty}</QTY> {name}{prep}</INGREDIENT>")
    out.append("Method:")
    for step in sorted(s.steps, key=lambda x: x.order):
        dur = _measure_text(step.params.duration)
        dur_txt = f" <TIME>{dur}</TIME>" if dur else ""
        out.append(f"<STEP>{step.order}. {step.action}{dur_txt}</STEP>")
    for tip in rec.content.tips:
        out.append(f"Tip: {tip}")
    for m in rec.content.common_mistakes:
        out.append(f"Common mistake: {m}")
    return "\n".join(out)


def _technique_doc(rec) -> str:
    s = rec.structure
    out = [f"<TECHNIQUE>{_primary_name(rec)}{_native_gloss(rec)}</TECHNIQUE> "
           f"({s.family.replace('_', ' ')})."]
    for field in ("summary", "how_to", "sensory_cues", "failure_modes"):
        v = getattr(rec.content, field, None)
        if v:
            out.append(v)
    return "\n".join(out)


def _problem_doc(rec) -> str:
    c = rec.content
    out = [f"Problem: {c.description}"]
    if c.diagnosis:
        out.append(f"Cause: {c.diagnosis}")
    for fix in c.fixes:
        out.append(f"Fix: {fix}")
    for p in c.prevention:
        out.append(f"Prevention: {p}")
    return "\n".join(out)


def _term_doc(rec) -> str:
    s = rec.structure
    return (f"{s.transliteration} ({s.native}) — {rec.content.gloss}"
            + (f" e.g. {rec.content.example_sentence}" if rec.content.example_sentence else ""))


def iter_documents(store: CanonicalStore) -> Iterator[str]:
    """Yield one text document per relevant canonical record."""
    dispatch = {
        "dish": lambda r: _dish_doc(store, r),
        "recipe": lambda r: _recipe_doc(store, r),
        "technique": _technique_doc,
        "problem": _problem_doc,
        "term": _term_doc,
    }
    for rec in store.records.values():
        fn = dispatch.get(rec.type)
        if fn is None or rec.version.status != "active":
            continue
        text = fn(rec).strip()
        if text:
            yield text


def store_to_text(store: CanonicalStore) -> str:
    """The whole store as one blank-line-separated text blob."""
    return "\n\n".join(iter_documents(store)) + "\n"
