"""Canonical records -> SFT instruction examples (master_baseline.md SS4.1/SS4.2).

Template-based, not LLM-based: each function below takes an already-trusted
canonical record (or a small, explicit combination of them) and deterministically
renders it into one of the 8 documented task shapes. No model call, no
fact-invention -- the facts all come from the record; only the English phrasing
around them is synthesized, which is exactly what SS4.1 means by "~60% agent-built
by serializing #1 into Q&A."

Each example is ``{"messages": [...], "provenance": {...}}`` (SS4.1's instruction-
data format). ``provenance.backing_record_ids`` always traces back to the
canonical record(s) this was built from, so a bad answer is traceable to the
canonical fact that produced it.

Task types (SS4.2's table) and their backing record type:
  recipe_request        -> recipe
  scaling                -> recipe (+ yield.servings)
  ingredient_substitution -> rel (substitutes_for)
  troubleshooting         -> problem
  technique_explanation   -> technique
  regional_comparison     -> 2x dish
  definition              -> term / ingredient
  abstention              -> none (hand-authored out-of-scope pairs)
"""
from __future__ import annotations

import random
import re

# food_slm.data.pipeline.resolve_batch_refs() appends a "  [unresolved field: value;
# ...]" audit-trail suffix to a record's content.summary whenever a structural
# reference doesn't resolve against the canonical store -- meant for staging-time
# human review (so a reviewer can see what's still missing), never meant to end up
# verbatim in SFT training data. Confirmed live: 155/2,778 regional_comparison
# training examples carried this text as "ground truth," and a real SFT'd model
# faithfully reproduced it ("[unresolved primary_ingredients: sambar-dal]") in a
# live generation -- the model was correctly learning to imitate what it was shown,
# the training data itself was the bug. Stripped at serialization time, below,
# rather than at the source (the audit note is still useful for staging review;
# it's only wrong once it crosses into training content).
_AUDIT_NOTE_SUFFIX = re.compile(r"\s*\[(?:unresolved [^\[\]]*)(?:;\s*unresolved [^\[\]]*)*\]\s*$")


def _clean_summary(text: str | None) -> str | None:
    if not text:
        return text
    return _AUDIT_NOTE_SUFFIX.sub("", text).strip() or None


def _label(rec: dict) -> str:
    labels = rec.get("labels") or []
    if labels:
        return labels[0]
    return rec["id"].split(":", 1)[1].replace("-", " ").title()


def _id_name(ref_id: str) -> str:
    return ref_id.split(":", 1)[1].replace("-", " ")


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _example(user: str, assistant: str, task_type: str, backing_ids: list[str]) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "provenance": {
            "task_type": task_type,
            "backing_record_ids": backing_ids,
            "generator": "food_slm.data.sft_serialize (mechanical, template-based)",
            "synthetic": True,
        },
    }


# --------------------------------------------------------------------------
# 1. Recipe request
# --------------------------------------------------------------------------

_RECIPE_PROMPTS = [
    "How do I make {dish}?",
    "Can you give me a recipe for {dish}?",
    "What's the recipe for {dish}?",
    "Tell me how to cook {dish}.",
    "I want to make {dish}. How do I do it?",
]


def _format_ingredient_line(ing: dict) -> str:
    name = _id_name(ing["ingredient_id"])
    q = ing.get("quantity")
    qty = q["original_text"] if q else None
    form = ing.get("form")
    bits = []
    if qty:
        bits.append(qty)
    bits.append(name)
    if form:
        bits.append(f"({form})")
    line = " ".join(bits)
    if ing.get("optional"):
        line += " (optional)"
    return line


def serialize_recipe_request(recipe: dict, rng: random.Random) -> dict:
    struct = recipe["structure"]
    name = _label(recipe)
    prompt = rng.choice(_RECIPE_PROMPTS).format(dish=name)

    lines = [f"Here's how to make {name}."]
    lines.append("")
    lines.append("Ingredients:")
    for ing in struct.get("ingredients", []):
        lines.append(f"- {_format_ingredient_line(ing)}")
    lines.append("")
    lines.append("Steps:")
    steps = sorted(struct.get("steps", []), key=lambda s: s["order"])
    for i, s in enumerate(steps, start=1):
        lines.append(f"{i}. {_sentence(s['action'])}")

    content = recipe.get("content", {})
    tips = content.get("tips") or []
    mistakes = content.get("common_mistakes") or []
    if tips:
        lines.append("")
        lines.append("Tips: " + " ".join(_sentence(t) for t in tips))
    if mistakes:
        lines.append("")
        lines.append("Common mistakes: " + " ".join(_sentence(m) for m in mistakes))
    serving = content.get("serving_suggestions")
    if serving:
        lines.append("")
        lines.append(_sentence(serving))

    return _example(prompt, "\n".join(lines), "recipe_request", [recipe["id"]])


# --------------------------------------------------------------------------
# 2. Scaling
# --------------------------------------------------------------------------

_SCALE_PROMPTS = [
    "How do I make {dish} for {n} people?",
    "Can you scale this {dish} recipe to serve {n}?",
    "I need to make {dish} for {n} people instead of the usual amount.",
]

_SCALE_NOTE = (
    "Cooking time, salt, and spice usually don't scale linearly -- taste and "
    "adjust seasoning near the end rather than scaling it by the same factor, "
    "and a bigger batch may need a slightly larger pan or a few extra minutes "
    "rather than exactly proportional time."
)


def serialize_scaling(recipe: dict, target_servings: int, rng: random.Random) -> dict | None:
    struct = recipe["structure"]
    base_servings = (struct.get("yield") or {}).get("servings")
    if not base_servings or base_servings <= 0:
        return None
    name = _label(recipe)
    factor = target_servings / base_servings
    prompt = rng.choice(_SCALE_PROMPTS).format(dish=name, n=target_servings)

    lines = [f"This recipe is written for {base_servings}, so for {target_servings} "
            f"scale everything by about {factor:.2g}x:"]
    for ing in struct.get("ingredients", []):
        name_ing = _id_name(ing["ingredient_id"])
        q = ing.get("quantity")
        if q and q.get("value") is not None:
            new_val = round(q["value"] * factor, 2)
            unit = q.get("unit_canonical") or ""
            lines.append(f"- {name_ing}: {new_val:g} {unit}".rstrip())
        elif q and q.get("qualitative"):
            lines.append(f"- {name_ing}: {q['original_text']} (adjust to taste, same as before)")
    lines.append("")
    lines.append(_SCALE_NOTE)

    return _example(prompt, "\n".join(lines), "scaling", [recipe["id"]])


# --------------------------------------------------------------------------
# 3. Ingredient substitution
# --------------------------------------------------------------------------

_SUB_PROMPTS = [
    "What can I use instead of {obj}?",
    "I don't have {obj}. What's a good substitute?",
    "Is there a substitute for {obj}?",
    "What's a good replacement for {obj} in South Indian cooking?",
]


def serialize_substitution(rel: dict, rng: random.Random) -> dict:
    struct = rel["structure"]
    subj_name = _id_name(struct["subject_id"])
    obj_name = _id_name(struct["object_id"])
    prompt = rng.choice(_SUB_PROMPTS).format(obj=obj_name)

    attrs = struct.get("attributes") or {}
    parts = [f"You can use {subj_name} instead of {obj_name}."]
    if attrs.get("ratio"):
        parts.append(_sentence(f"Use it at a ratio of {attrs['ratio']}"))
    if attrs.get("context"):
        parts.append(_sentence(f"This works well in {attrs['context']}"))
    if attrs.get("caveat"):
        parts.append(_sentence(attrs["caveat"]))
    note = (rel.get("content") or {}).get("note")
    if note:
        parts.append(_sentence(note))

    return _example(prompt, " ".join(parts), "ingredient_substitution", [rel["id"]])


# --------------------------------------------------------------------------
# 4. Troubleshooting
# --------------------------------------------------------------------------

_TROUBLE_PROMPTS = [
    "{symptom} What's wrong and how do I fix it?",
    "{symptom} Why does this happen?",
    "{symptom} How do I fix this?",
]


def serialize_troubleshooting(problem: dict, rng: random.Random) -> dict:
    c = problem.get("content", {})
    symptom = _sentence(c.get("symptoms") or c.get("description") or _label(problem))
    prompt = rng.choice(_TROUBLE_PROMPTS).format(symptom=symptom)

    lines = []
    if c.get("diagnosis"):
        lines.append(f"Likely cause: {_sentence(c['diagnosis'])}")
    fixes = c.get("fixes") or []
    if fixes:
        lines.append("Fix: " + " ".join(_sentence(f) for f in fixes))
    prevention = c.get("prevention") or []
    if prevention:
        lines.append("To prevent this next time: " + " ".join(_sentence(p) for p in prevention))

    return _example(prompt, "\n".join(lines), "troubleshooting", [problem["id"]])


# --------------------------------------------------------------------------
# 5. Technique explanation
# --------------------------------------------------------------------------

_TECH_PROMPTS = [
    "What is {name}?",
    "Can you explain the {name} technique?",
    "How do I {name}?",
    "What does {name} mean in cooking?",
]


def serialize_technique(tech: dict, rng: random.Random) -> dict:
    name = _label(tech)
    c = tech.get("content", {})
    prompt = rng.choice(_TECH_PROMPTS).format(name=name.lower())

    lines = [_sentence(_clean_summary(c["summary"]))]
    if c.get("how_to"):
        lines.append(_sentence(c["how_to"]))
    if c.get("sensory_cues"):
        lines.append(_sentence(f"You'll know it's done when {c['sensory_cues']}"))
    if c.get("failure_modes"):
        lines.append(_sentence(f"Common mistake: {c['failure_modes']}"))

    return _example(prompt, " ".join(lines), "technique_explanation", [tech["id"]])


# --------------------------------------------------------------------------
# 6. Regional comparison
# --------------------------------------------------------------------------

_REGIONAL_PROMPTS = [
    "How does {name_a} differ from {name_b}?",
    "What's the difference between {region_a} and {region_b} style {category}?",
    "Compare {name_a} and {name_b}.",
]


def serialize_regional_comparison(dish_a: dict, dish_b: dict, rng: random.Random) -> dict:
    a, b = dish_a["structure"], dish_b["structure"]
    name_a, name_b = _label(dish_a), _label(dish_b)
    region_a = a["cuisine_state"].replace("_", " ").title()
    region_b = b["cuisine_state"].replace("_", " ").title()
    category = a["category"].replace("_", " ")
    prompt = rng.choice(_REGIONAL_PROMPTS).format(
        name_a=name_a, name_b=name_b, region_a=region_a, region_b=region_b, category=category)

    lines = [f"{name_a} ({region_a}) and {name_b} ({region_b}) are both {category} dishes, "
            "but they differ in a few ways."]
    ing_a, ing_b = set(a.get("primary_ingredients", [])), set(b.get("primary_ingredients", []))
    only_a, only_b, common = ing_a - ing_b, ing_b - ing_a, ing_a & ing_b

    def fmt(ids: set[str], limit: int = 5) -> str:
        # Some dish records' primary_ingredients list isn't curated down to a
        # short headline set -- cap the prose so one over-long list doesn't
        # produce an unreadable wall of ingredient names.
        names = sorted(_id_name(i) for i in ids)
        shown = ", ".join(names[:limit])
        return shown + f", and {len(names) - limit} more" if len(names) > limit else shown

    if only_a:
        lines.append(f"{name_a} typically uses {fmt(only_a)}, which {name_b} doesn't.")
    if only_b:
        lines.append(f"{name_b} typically uses {fmt(only_b)}, which {name_a} doesn't.")
    if common:
        lines.append(f"Both share {fmt(common)}.")
    summary_a = _clean_summary(dish_a.get("content", {}).get("summary"))
    summary_b = _clean_summary(dish_b.get("content", {}).get("summary"))
    if summary_a:
        lines.append(f"{name_a}: {_sentence(summary_a)}")
    if summary_b:
        lines.append(f"{name_b}: {_sentence(summary_b)}")

    return _example(prompt, " ".join(lines), "regional_comparison", [dish_a["id"], dish_b["id"]])


# --------------------------------------------------------------------------
# 7. Definition / factual
# --------------------------------------------------------------------------

_DEF_PROMPTS = [
    "What is {name}?",
    "Can you define {name}?",
    "What does {name} mean?",
]


def serialize_term_definition(term: dict, rng: random.Random) -> dict:
    name = term["structure"]["transliteration"]
    prompt = rng.choice(_DEF_PROMPTS).format(name=name)
    c = term.get("content", {})
    lines = [_sentence(c["gloss"])]
    if c.get("usage_notes"):
        lines.append(_sentence(c["usage_notes"]))
    if c.get("example_sentence"):
        lines.append(f'For example: "{c["example_sentence"]}"')
    return _example(prompt, " ".join(lines), "definition", [term["id"]])


def serialize_ingredient_definition(ing: dict, rng: random.Random) -> dict:
    name = _label(ing)
    prompt = rng.choice(_DEF_PROMPTS).format(name=name)
    c = ing.get("content", {})
    lines = [_sentence(_clean_summary(c["summary"]))]
    if c.get("description"):
        lines.append(_sentence(c["description"]))
    return _example(prompt, " ".join(lines), "definition", [ing["id"]])


# --------------------------------------------------------------------------
# 8. Out-of-scope / abstention -- hand-authored, not derived from any canonical
# record (there's nothing to serialize; the point is declining gracefully).
# A small starter set -- like the golden sets, this is meant to be grown by a
# human over time, not template-generated at volume.
# --------------------------------------------------------------------------

_ABSTENTION_PAIRS = [
    ("What's the capital of France?",
     "I'm focused on South Indian cuisine, so I can't help with general geography -- "
     "but I'd be glad to tell you about a South Indian dish or technique instead."),
    ("Can you write me a Python script to sort a list?",
     "I'm specialized in South Indian food knowledge, not programming -- I can't help with that."),
    ("How do I make sushi?",
     "I only cover South Indian cuisine, so I don't have reliable knowledge about sushi -- "
     "happy to help with a South Indian dish instead."),
    ("Who won the last World Cup?",
     "That's outside what I know about -- I'm focused on South Indian food and cooking."),
    ("How do I fix my car's brakes?",
     "I can't help with that -- I'm a South Indian cuisine specialist, not a mechanic."),
    ("What's a good French dessert recipe?",
     "I don't have reliable knowledge of French cuisine -- I'm focused specifically on South "
     "Indian dishes. I could suggest a South Indian sweet instead, like payasam or Mysore pak."),
    ("Can you recommend a good thriller novel?",
     "That's outside what I can help with -- I only know about South Indian food and cooking."),
    ("What's the weather like today?",
     "I don't have access to weather information -- I'm focused on South Indian cuisine."),
    ("How do I make a Punjabi butter chicken?",
     "I'm specialized in South Indian cuisine specifically, so I don't have reliable knowledge "
     "of North Indian dishes like butter chicken."),
    ("What medications help with a stomach ache?",
     "I can't give medical advice -- I'm a cooking specialist, not a healthcare provider. "
     "Please check with a doctor or pharmacist."),
    ("Tell me about the history of Rome.",
     "That's outside my area -- I only know about South Indian food and cooking."),
    ("How much does a flight to Chennai cost?",
     "I don't have travel or pricing information -- I'm focused on South Indian cuisine."),
    ("Can you fix a leaking pipe for me?",
     "I can't help with that -- I'm a South Indian cuisine specialist, not a plumber."),
    ("My pipe is leaking, what do I do?",
     "I don't have plumbing expertise -- I'm focused on South Indian cooking. You'd want a "
     "plumber or a home-repair guide for that."),
    ("What stock should I invest in?",
     "I can't give financial advice -- I'm a cooking specialist, not a financial advisor."),
    ("Should I take out a loan to buy a car?",
     "That's outside what I can help with -- I only know about South Indian food and cooking, "
     "not financial decisions."),
    ("My partner and I keep arguing, what should I do?",
     "I'm not equipped to give relationship advice -- I'm focused on South Indian cuisine. "
     "A counselor would be a better resource for that."),
    ("Is it going to rain tomorrow in Chennai?",
     "I don't have access to weather forecasts -- I'm focused on South Indian cuisine, not "
     "meteorology."),
    ("Who's going to win the next election?",
     "I can't help with that -- I'm a cooking specialist, not a political analyst."),
    ("Can you translate 'thank you' into French?",
     "I'm specialized in South Indian cuisine, not translation -- I can't reliably help with "
     "that."),
    ("Write me a poem about the ocean.",
     "I'm focused specifically on South Indian cuisine, not creative writing -- I can't help "
     "with that."),
    ("How do I make authentic Italian carbonara?",
     "I don't have reliable knowledge of Italian cuisine -- I'm specialized in South Indian "
     "cooking specifically."),
    ("What's a good Chinese stir-fry recipe?",
     "I only cover South Indian cuisine, so I don't have reliable knowledge of Chinese cooking."),
    ("How do I make Mexican tacos?",
     "That's outside my area -- I'm focused specifically on South Indian cuisine, not Mexican "
     "food."),
    ("What's the best Bengali fish curry recipe?",
     "I'm specialized in South Indian cuisine specifically, so I don't have reliable knowledge "
     "of Bengali cooking, even though it's also Indian."),
    ("Give me a recipe for Punjabi sarson ka saag.",
     "I'm specialized in South Indian cuisine, so I don't have reliable knowledge of Punjabi "
     "dishes like sarson ka saag."),
    ("Can you diagnose why my computer is running slowly?",
     "I can't help with that -- I'm a cooking specialist, not a tech support resource."),
    ("What's my legal liability if my dog bites someone?",
     "I can't give legal advice -- I'm a cooking specialist, not a lawyer."),
    ("Can you recommend a good workout routine?",
     "That's outside what I can help with -- I'm focused on South Indian cuisine, not fitness."),
    ("What's the square root of 2048?",
     "I'm not built for general math problems -- I'm focused on South Indian cuisine."),
    ("Tell me a joke.",
     "I'm specialized in South Indian cuisine, not comedy -- I don't have a good joke for you."),
    ("What are the side effects of ibuprofen?",
     "I can't give medical advice -- please check with a doctor or pharmacist for that."),
]


def abstention_examples(*, repeat: int = 1) -> list[dict]:
    """`repeat` deliberately oversamples abstention examples in the final mix --
    at the default repeat=1, abstention was only 12/3,392 (0.35%) of training
    data, diluted to near-nothing across a multi-epoch run dominated by domain
    Q&A examples. Confirmed live: a real SFT'd model failed to abstain on a
    question that was in its training set VERBATIM ("How do I fix a leaking
    pipe?"), because the signal was too rare to generalize from reliably even
    on an exact match. Same fix pattern already used for the pretraining
    corpus's domain slice (--domain-repeat 6 in pack_40m.py) -- oversampling an
    underrepresented-but-important category, not a new technique for this
    project."""
    return [_example(q, a, "abstention", []) for q, a in _ABSTENTION_PAIRS for _ in range(repeat)]
