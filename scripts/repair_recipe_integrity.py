"""One-off (but safely rerunnable) repair for recipe records written before two
fixes in extract_recipe.py:

1. Step.equipment_id entries not reflected in the recipe's own equipment_ids list
   (root cause unclear -- likely a now-superseded code path; 1,670 such issues
   found live).
2. Step.input_ingredient_ids referencing a fabricated ingredient id for an
   intermediate-state description ("the combined mixture") that was never added
   to the recipe's `ingredients` list -- the exact bug fixed in _build_recipe()'s
   ingredients_used handling (1,735 such issues found live).

Both are self-consistency repairs: recompute equipment_ids from the union of what
steps actually reference, and drop (never fabricate) a dangling ingredient
reference. Re-validates every repaired record against the schema before writing
back, and only writes a file whose content actually changed.

    python scripts/repair_recipe_integrity.py --staging    # data/interim/canon_staging/recipe/
    python scripts/repair_recipe_integrity.py --canonical  # data/canonical/recipes.jsonl
    python scripts/repair_recipe_integrity.py --both       # default
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.schema.models import parse_record  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STAGING_RECIPE_DIR = REPO / "data" / "interim" / "canon_staging" / "recipe"
CANONICAL_RECIPES = REPO / "data" / "canonical" / "recipes.jsonl"


def repair_recipe(rec: dict, *, known_equipment_ids: set[str] | None = None) -> tuple[dict, bool]:
    """Returns (repaired_record, changed).

    `known_equipment_ids`: when given (canonical repair only), a step's
    equipment_id -- and the top-level equipment_ids list -- is also stripped of
    anything not in this set, matching schema.store.check_all()'s requirement
    that equipment_ids resolve globally (unlike input_ingredient_ids, which is
    self-consistency-only -- see canon_staging.select_promotable's comment).
    When None (staging repair), only self-consistency is enforced -- a staged,
    not-yet-promoted recipe is allowed to reference equipment that doesn't exist
    in canonical yet; that's select_promotable's job to resolve at promotion time.
    """
    struct = rec.get("structure", {})
    steps = struct.get("steps", [])
    ingredient_ids = {i["ingredient_id"] for i in struct.get("ingredients", [])}

    equipment_ids = set(struct.get("equipment_ids", []))
    changed = False

    for step in steps:
        eid = step.get("equipment_id")
        if eid:
            if known_equipment_ids is not None and eid not in known_equipment_ids:
                step.pop("equipment_id", None)
                changed = True
            elif eid not in equipment_ids:
                equipment_ids.add(eid)
                changed = True
        input_ids = step.get("input_ingredient_ids", [])
        kept = [iid for iid in input_ids if iid in ingredient_ids]
        if len(kept) != len(input_ids):
            step["input_ingredient_ids"] = kept
            changed = True

    if known_equipment_ids is not None:
        equipment_ids &= known_equipment_ids

    new_equipment_ids = sorted(equipment_ids)
    if new_equipment_ids != struct.get("equipment_ids", []):
        struct["equipment_ids"] = new_equipment_ids
        changed = True

    return rec, changed


def repair_staging() -> tuple[int, int]:
    checked = repaired_count = 0
    for p in sorted(STAGING_RECIPE_DIR.glob("*.json")) if STAGING_RECIPE_DIR.exists() else []:
        checked += 1
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec, changed = repair_recipe(rec)
        if not changed:
            continue
        parse_record(rec)  # raise loudly if the repair somehow broke schema validity
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        repaired_count += 1
    return checked, repaired_count


def repair_canonical() -> tuple[int, int]:
    if not CANONICAL_RECIPES.exists():
        return 0, 0
    from food_slm.schema import load_store

    store = load_store(REPO / "data" / "canonical", strict=False)
    known_equipment_ids = {rid for rid in store.records if rid.startswith("equip:")}

    lines = CANONICAL_RECIPES.read_text(encoding="utf-8").splitlines()
    out_lines = []
    checked = repaired_count = 0
    for ln in lines:
        if not ln.strip():
            continue
        checked += 1
        rec = json.loads(ln)
        rec, changed = repair_recipe(rec, known_equipment_ids=known_equipment_ids)
        if changed:
            parse_record(rec)
            repaired_count += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    CANONICAL_RECIPES.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return checked, repaired_count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", action="store_true")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()
    do_staging = args.staging or args.both or not (args.staging or args.canonical)
    do_canonical = args.canonical or args.both or not (args.staging or args.canonical)

    if do_staging:
        checked, repaired = repair_staging()
        print(f"staging: {repaired}/{checked} recipe records repaired")
    if do_canonical:
        checked, repaired = repair_canonical()
        print(f"canonical: {repaired}/{checked} recipe records repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
