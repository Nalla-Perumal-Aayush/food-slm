"""Mechanically derive `rel` records from cross-reference fields already present on
promoted canonical records (mcp_extraction_pipeline.md / schema.store._ID_REF_FIELDS).

No LLM call, no new facts: this just re-expresses a link the store already asserts
(e.g. dish:achappam.primary_ingredients contains ing:coconut-milk) as its own typed
`rel` record, because that's the schema's native way to represent it for downstream
SFT serialization (regional-comparison / substitution / etc. task shapes read `rel`
records, not raw structure-field lists).

    python scripts/derive_rel.py            # stage only
    python scripts/derive_rel.py --promote --reviewer NAME   # stage + promote-ready

Idempotent: rel ids are deterministic (predicate + subject + object), so re-running
after new dishes are promoted only adds the new links.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.canon_staging import stage_records  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
STAGING = REPO / "data" / "interim" / "canon_staging"

# (source record type, structure field, predicate, subject-is-the-field-id) --
# subject_is_field_id=False means the CONTAINING record is the subject and the
# field's id(s) are the object(s); True means it's reversed (the field's id(s)
# are the subject, the containing record is the object) -- matches `used_in`'s
# direction (ingredient used_in dish, not dish used_in ingredient).
DERIVATIONS = [
    ("dish", "primary_ingredients", "used_in", True),
    ("dish", "typical_techniques", "requires_technique", False),
    ("dish", "regions", "belongs_to_region", False),
    ("dish", "constraints_typically_met", "satisfies_constraint", False),
    ("technique", "typical_equipment_ids", "requires_equipment", False),
    ("ingredient", "common_substitutes", "substitutes_for", False),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _rel_id(predicate: str, subject_id: str, object_id: str) -> str:
    slug = lambda rid: rid.split(":", 1)[1]  # noqa: E731
    return f"rel:{predicate}-{slug(subject_id)}-{slug(object_id)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--reviewer", default=None)
    args = ap.parse_args()
    if args.promote and not args.reviewer:
        ap.error("--promote requires --reviewer")

    today = datetime.now(UTC).date().isoformat()
    by_type = {
        "dish": _load_jsonl(CANON / "dishes.jsonl"),
        "technique": _load_jsonl(CANON / "techniques.jsonl"),
        "ingredient": _load_jsonl(CANON / "ingredients.jsonl"),
    }

    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for rtype, field, predicate, reversed_dir in DERIVATIONS:
        for rec in by_type.get(rtype, []):
            values = rec.get("structure", {}).get(field)
            if not values:
                continue
            if not isinstance(values, list):
                values = [values]
            for other_id in values:
                if not isinstance(other_id, str) or ":" not in other_id:
                    continue
                subj, obj = (other_id, rec["id"]) if reversed_dir else (rec["id"], other_id)
                rid = _rel_id(predicate, subj, obj)
                if rid in seen:
                    continue
                seen[rid] = {
                    "id": rid, "type": "rel", "schema_version": "2",
                    "labels": ["relationship"],
                    "content": {"note": f"Derived mechanically from {rec['id']}'s "
                               f"{field} (no independent extraction)."},
                    "structure": {"predicate": predicate, "subject_id": subj, "object_id": obj},
                    "provenance": {"source_ids": [], "transformations": ["derived"],
                                   "acquired_on": today, "synthetic": True,
                                   "generator": "scripts/derive_rel.py (mechanical, no LLM)",
                                   "seed_record_ids": [subj, obj]},
                    "quality": {"confidence": 0.95, "validation_status": "auto_validated",
                               "corroboration_count": 1},
                    "version": {"revision": 1, "created_on": today, "updated_on": today,
                               "status": "active"},
                }
                counts[predicate] = counts.get(predicate, 0) + 1

    print(f"derived {len(seen)} rel records:")
    for pred, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {pred:<22} {n}")

    res = stage_records(list(seen.values()), STAGING)
    print(f"\nstaged: {len(res.staged)}  rejected: {len(res.rejected)}")
    for rid, reason in res.rejected[:10]:
        print(f"  REJECTED {rid}: {reason}")

    if args.promote:
        subprocess.run(  # noqa: S603
            [sys.executable, str(REPO / "scripts" / "canon_run.py"), "promote-ready",
             "--reviewer", args.reviewer], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
