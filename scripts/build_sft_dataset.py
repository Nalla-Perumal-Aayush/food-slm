"""Canonical store -> data/instruction/sft_v1.jsonl (master_baseline.md SS4.1).

    python scripts/build_sft_dataset.py

Runs every food_slm.data.sft_serialize function over the current canonical
store. Deterministic (seeded RNG, keyed per-record so re-running after the
store grows doesn't reshuffle already-generated phrasing for unchanged
records) and purely mechanical -- no model call. Volume grows naturally as
canonical coverage grows; this is meant to be re-run, not a one-shot.

Regional-comparison pairs (dish x dish within the same category, different
region) are capped per category to avoid a combinatorial blow-up from a
large, generic category like "tiffin" (46 dishes) swamping everything else.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.sft_serialize import (  # noqa: E402
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
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
OUT = REPO / "data" / "instruction" / "sft_v1.jsonl"

MAX_REGIONAL_PAIRS_PER_CATEGORY = 20
SCALING_TARGETS = [2, 6]  # e.g. a 4-serving recipe gets a "for 2" and "for 6" variant
# Deliberate oversampling, same pattern as the pretraining corpus's domain upsample
# (pack_40m.py's --domain-repeat 6): at repeat=1 abstention was 12/3,392 (0.35%) of
# the mix, too rare a signal for a multi-epoch SFT run to generalize from -- a real
# SFT'd model failed to abstain on a question that was in its training data
# VERBATIM. ~32 distinct pairs x 8 = ~256, roughly 7% of the final mix.
ABSTENTION_REPEAT = 8


def _rng_for(record_id: str) -> random.Random:
    return random.Random(record_id)


def main() -> int:
    store = load_store(CANON, strict=False)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for rec in store.records.values():
        by_type[rec.type].append(rec.model_dump(mode="json", by_alias=True))

    examples: list[dict] = []

    for recipe in by_type.get("recipe", []):
        examples.append(serialize_recipe_request(recipe, _rng_for(recipe["id"])))
        for n in SCALING_TARGETS:
            ex = serialize_scaling(recipe, n, _rng_for(f"{recipe['id']}-scale-{n}"))
            if ex:
                examples.append(ex)

    for rel in by_type.get("rel", []):
        if rel["structure"].get("predicate") == "substitutes_for":
            examples.append(serialize_substitution(rel, _rng_for(rel["id"])))

    for problem in by_type.get("problem", []):
        examples.append(serialize_troubleshooting(problem, _rng_for(problem["id"])))

    for tech in by_type.get("technique", []):
        examples.append(serialize_technique(tech, _rng_for(tech["id"])))

    for term in by_type.get("term", []):
        examples.append(serialize_term_definition(term, _rng_for(term["id"])))

    for ing in by_type.get("ingredient", []):
        examples.append(serialize_ingredient_definition(ing, _rng_for(ing["id"])))

    dishes_by_category: dict[str, list[dict]] = defaultdict(list)
    for dish in by_type.get("dish", []):
        dishes_by_category[dish["structure"]["category"]].append(dish)
    for category, dishes in dishes_by_category.items():
        pairs = []
        for i, a in enumerate(dishes):
            for b in dishes[i + 1:]:
                if a["structure"]["cuisine_state"] != b["structure"]["cuisine_state"]:
                    pairs.append((a, b))
        rng = _rng_for(f"regional-{category}")
        rng.shuffle(pairs)
        for a, b in pairs[:MAX_REGIONAL_PAIRS_PER_CATEGORY]:
            examples.append(serialize_regional_comparison(a, b, _rng_for(f"{a['id']}-{b['id']}")))

    examples.extend(abstention_examples(repeat=ABSTENTION_REPEAT))

    by_task: dict[str, int] = defaultdict(int)
    for ex in examples:
        by_task[ex["provenance"]["task_type"]] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} examples to {OUT}")
    for task, n in sorted(by_task.items()):
        print(f"  {task}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
