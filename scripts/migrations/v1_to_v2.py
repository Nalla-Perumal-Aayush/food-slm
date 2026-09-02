"""Migrate ``data/canonical/*.jsonl`` from schema v1 to v2 (South Indian domain).

Idempotent — records already at ``schema_version: "2"`` are left untouched.
Transforms (baseline §3.1):

  * ``schema_version`` "1" -> "2"
  * structure ``name_ta`` + ``transliteration`` (+ ``ipa`` dropped)
        -> ``names: [{lang: "ta", native, transliteration, primary: true}]``
  * ``term`` records: ``term_ta`` + ``transliteration`` -> ``lang`` / ``native`` / ``transliteration``
  * ``dish`` / ``recipe``: add ``cuisine_state`` (default ``tamil_nadu``) and ``diet``
        (inferred from ingredient categories / allergen classes)
  * ``region``: add ``level`` (heuristic) + ``cuisine_state`` + ``parent_region``;
        append a state-level region for any referenced ``cuisine_state`` that lacks one
  * ``constraint.kind``: ``ritual_religious`` -> ``cooking_style``,
        ``seasonal_ayurvedic`` -> ``preference``; doctrinal ``rationale`` cleared
  * ``constraint:no-onion-no-garlic``: content rewritten to a neutral cooking-style description

Run:  python scripts/migrations/v1_to_v2.py [--dir data/canonical] [--dry-run]
Then: python scripts/validate_canonical.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from food_slm.schema.models import RECORD_FILES, find_disallowed_term  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

STATE_SLUGS = {
    "tamil-nadu": "tamil_nadu", "kerala": "kerala", "karnataka": "karnataka",
    "andhra-pradesh": "andhra_pradesh", "telangana": "telangana", "puducherry": "puducherry",
}
STATE_KEY_TO_SLUG = {v: k for k, v in STATE_SLUGS.items()}
STYLE_SLUGS = {"chettinad", "kongunadu", "malabar", "udupi", "rayalaseema", "coorg",
               "hyderabadi-deccan", "mangalorean"}

NONVEG_CATEGORY = {"seafood": "contains_seafood", "poultry": "contains_poultry",
                   "red_meat": "contains_red_meat", "egg": "contains_egg"}


def infer_diet(ingredient_ids: list[str], ing_index: dict[str, dict]) -> dict:
    diet = {"vegetarian": True, "vegan": False, "contains_dairy": False, "contains_egg": False,
            "contains_seafood": False, "contains_poultry": False, "contains_red_meat": False}
    for iid in ingredient_ids:
        ing = ing_index.get(iid)
        if not ing:
            continue
        st = ing.get("structure", {})
        cat = st.get("category")
        allergens = st.get("allergen_class", []) or []
        if cat in NONVEG_CATEGORY:
            diet[NONVEG_CATEGORY[cat]] = True
            if cat != "egg":
                diet["vegetarian"] = False
        if cat == "egg":
            diet["vegetarian"] = False
        if cat == "dairy" or "dairy" in allergens:
            diet["contains_dairy"] = True
    return diet


def migrate_record(rec: dict, ing_index: dict[str, dict], needed_states: set[str]) -> dict:
    if rec.get("schema_version") == "2":
        return rec
    rtype = rec["type"]
    st = rec.setdefault("structure", {})
    rec["schema_version"] = "2"
    rec.setdefault("provenance", {}).setdefault("transformations", []).append("migration:v1_to_v2")

    # names[] --------------------------------------------------------------
    if rtype == "term":
        native = st.pop("term_ta", None)
        translit = st.pop("transliteration", None)
        if native:
            st["lang"] = "ta"
            st["native"] = native
            st["transliteration"] = translit or native
    elif rtype in {"ingredient", "dish", "recipe", "technique", "equipment", "region"}:
        native = st.pop("name_ta", None)
        translit = st.pop("transliteration", None)
        st.pop("ipa", None)
        names = st.get("names", [])
        if native and not names:
            names = [{"lang": "ta", "native": native,
                      "transliteration": translit or native, "primary": True}]
        st["names"] = names

    # cuisine_state + diet ----------------------------------------------
    if rtype in {"dish", "recipe"}:
        cs = st.setdefault("cuisine_state", "tamil_nadu")
        if cs != "pan_south":
            needed_states.add(cs)
        if "diet" not in st:
            if rtype == "recipe":
                ids = [ri["ingredient_id"] for ri in st.get("ingredients", [])]
            else:
                ids = st.get("primary_ingredients", [])
            st["diet"] = infer_diet(ids, ing_index)

    # region ----------------------------------------------------------
    if rtype == "region":
        slug = rec["id"].split(":", 1)[1]
        if "level" not in st:
            if slug in STATE_SLUGS:
                st["level"] = "state"
            elif slug in STYLE_SLUGS:
                st["level"] = "style"
            else:
                st["level"] = "sub_region"
        if st["level"] == "state":
            st["cuisine_state"] = STATE_SLUGS.get(slug)
            st["parent_region"] = None
        else:
            cs = st.get("cuisine_state") or "tamil_nadu"
            st["cuisine_state"] = cs
            needed_states.add(cs)
            if not st.get("parent_region"):
                st["parent_region"] = f"region:{STATE_KEY_TO_SLUG[cs]}"

    # constraint ----------------------------------------------------
    if rtype == "constraint":
        kind = st.get("kind")
        if kind == "ritual_religious":
            st["kind"] = "cooking_style"
        elif kind == "seasonal_ayurvedic":
            st["kind"] = "preference"
        if rec["id"] == "constraint:no-onion-no-garlic":
            rec["labels"] = ["No onion, no garlic", "Allium-free", "Saatvik style"]
            rec["content"] = {
                "description": ("Excludes onion, garlic and other alliums. A common cooking "
                                "style across South India for everyday meals, festival food and "
                                "temple kitchens."),
                "rationale": ("Cooks who avoid alliums lean on asafoetida for savoury depth and "
                              "on ginger, curry leaves and pepper for aroma."),
            }
        else:
            con = rec.get("content", {})
            for key in ("description", "rationale"):
                val = con.get(key)
                if val and find_disallowed_term(val):
                    con[key] = None  # force manual rewrite; validator will flag until fixed

    return rec


def state_region_record(cs_key: str) -> dict:
    slug = STATE_KEY_TO_SLUG[cs_key]
    name = slug.replace("-", " ").title()
    return {
        "id": f"region:{slug}", "type": "region", "schema_version": "2",
        "labels": [name],
        "content": {"summary": f"{name}, one of the South Indian states in project scope."},
        "structure": {"level": "state", "cuisine_state": cs_key, "parent_region": None,
                      "names": [], "districts": [], "signature_dishes": [],
                      "signature_ingredients": []},
        "provenance": {"source_ids": ["src:hand-authored-seed"], "acquired_on": "2026-09-02",
                       "transformations": ["migration:v1_to_v2"], "synthetic": False},
        "quality": {"confidence": 0.9, "validation_status": "human_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02",
                    "status": "active"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/canonical", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = REPO / args.dir if not args.dir.is_absolute() else args.dir

    # index ingredients first (needed for diet inference)
    ing_path = root / RECORD_FILES["ingredient"]
    ing_index: dict[str, dict] = {}
    if ing_path.exists():
        for line in ing_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                ing_index[o["id"]] = o

    needed_states: set[str] = set()
    changed = 0
    per_file: dict[Path, list[dict]] = {}

    for fname in RECORD_FILES.values():
        fpath = root / fname
        if not fpath.exists():
            continue
        recs = []
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            before = json.dumps(obj, sort_keys=True)
            obj = migrate_record(obj, ing_index, needed_states)
            if json.dumps(obj, sort_keys=True) != before:
                changed += 1
            recs.append(obj)
        per_file[fpath] = recs

    # ensure a state-level region exists for every referenced cuisine_state
    regions_path = root / RECORD_FILES["region"]
    existing_region_ids = {r["id"] for r in per_file.get(regions_path, [])}
    added_states = []
    for cs in sorted(needed_states):
        rid = f"region:{STATE_KEY_TO_SLUG[cs]}"
        if rid not in existing_region_ids:
            per_file.setdefault(regions_path, []).append(state_region_record(cs))
            added_states.append(rid)

    for fpath, recs in per_file.items():
        recs.sort(key=lambda r: r["id"])
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        if args.dry_run:
            print(f"[dry-run] would write {fpath.name}: {len(recs)} records")
        else:
            fpath.write_text(text, encoding="utf-8")

    print(f"{'[dry-run] ' if args.dry_run else ''}migrated {changed} record(s); "
          f"added state regions: {added_states or 'none'}")
    if not args.dry_run:
        print("next: python scripts/validate_canonical.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
