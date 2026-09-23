"""Load and validate the canonical knowledge store (``data/canonical/``).

Two layers:

1. **Per-record** validation — Pydantic models in :mod:`food_slm.schema.models`
   (envelope, enums, measurement normalization, recipe DAG rules).
2. **Cross-record** validation — the referential-integrity contract in §8 of
   ``Documents/phase5_canonical_schema.md`` (every ``*_id`` resolves, predicate
   subject/object types legal, synthetic records grounded, etc.).

``load_store(path, strict=True)`` runs both and raises :class:`ValidationError`
with every problem collected. This is the schema half of the Phase 11 hard gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from food_slm.schema.models import (
    RECORD_FILES,
    Record,
    Relationship,
    find_disallowed_term,
    iter_strings,
    parse_record,
)


class ValidationError(Exception):
    """Raised by :func:`load_store` when ``strict`` and any issue was found."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__(f"{len(issues)} canonical-store issue(s):\n  - " + "\n  - ".join(issues))


# fields whose value(s) are canonical entity ids that must resolve
_ID_REF_FIELDS: dict[str, list[str]] = {
    "dish": ["primary_ingredients", "typical_techniques", "regions", "expected_outcome_ids",
             "constraints_typically_met", "variant_of"],
    # dish_id excluded on purpose -- see pipeline._REF_FIELDS's comment: it's a
    # required schema field, not something dangling-ref cleanup should ever null out.
    "recipe": ["equipment_ids", "states", "outcome_ids", "constraints_met"],
    "ingredient": ["common_substitutes"],
    "technique": ["typical_equipment_ids", "precedes", "follows"],
    "equipment": ["substitutes", "required_for_techniques"],
    "region": ["parent_region", "signature_dishes", "signature_ingredients"],
    "state": ["reachable_from", "via_technique_ids"],
    "constraint": ["excludes_ingredients"],
    "problem": ["related_dish_ids", "related_technique_ids", "related_ingredient_ids",
                "fix_relationship_ids"],
    "term": ["maps_to_id"],
}


@dataclass
class CanonicalStore:
    """In-memory view of every canonical record, indexed by id."""

    records: dict[str, Record] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def by_type(self, rtype: str) -> list[Record]:
        return [r for r in self.records.values() if r.type == rtype]

    def get(self, rid: str) -> Record | None:
        return self.records.get(rid)

    def resolve(self, rid: str) -> str:
        """Follow a ``merged`` record to its survivor; return rid unchanged otherwise."""
        seen = set()
        cur = rid
        while cur in self.records and self.records[cur].version.status == "merged":
            if cur in seen:
                break
            seen.add(cur)
            cur = self.records[cur].merged_into or cur
        return cur

    # -- cross-record checks (§8) ------------------------------------------

    def check_references(self) -> list[str]:
        issues: list[str] = []

        def _refs(rec: Record) -> list[str]:
            out: list[str] = []
            struct = rec.structure.model_dump()
            for f in _ID_REF_FIELDS.get(rec.type, []):
                val = struct.get(f)
                if val is None:
                    continue
                out.extend(val if isinstance(val, list) else [val])
            return out

        for rid, rec in self.records.items():
            for ref in _refs(rec):
                if not isinstance(ref, str) or ":" not in ref:
                    issues.append(f"{rid}: malformed reference {ref!r}")
                    continue
                target = self.records.get(ref)
                if target is None:
                    issues.append(f"{rid}: dangling reference {ref!r}")
                elif target.version.status == "merged":
                    issues.append(
                        f"{rid}: reference {ref!r} is merged -> use {self.resolve(ref)!r}"
                    )

            if isinstance(rec, Relationship):
                for side in ("subject_id", "object_id"):
                    ref = getattr(rec.structure, side)
                    if self.records.get(ref) is None:
                        issues.append(f"{rid}: relationship {side} {ref!r} does not resolve")

            # recipe: step equipment / technique must be declared on the recipe
            if rec.type == "recipe":
                declared_equip = set(rec.structure.equipment_ids)
                for step in rec.structure.steps:
                    if step.equipment_id and step.equipment_id not in declared_equip:
                        issues.append(
                            f"{rid}: step {step.step_id} uses {step.equipment_id!r} "
                            "not in recipe.equipment_ids"
                        )
                    for iid in step.input_ingredient_ids:
                        if self.records.get(iid) is None:
                            issues.append(f"{rid}: step {step.step_id} ingredient {iid!r} dangling")

        return issues

    def check_community_tags(self, extra_terms: tuple[str, ...] = ()) -> list[str]:
        """Rev 5: no community / religion attribution anywhere in a record, including prose.

        The per-record Pydantic validator already rejects such terms in ``labels`` and
        ``structure``; this extends the scan to ``content`` (history / cultural_notes / …),
        which the validator deliberately leaves to human review but the Phase 11 gate checks.
        """
        issues: list[str] = []
        for rid, rec in self.records.items():
            for text in iter_strings(rec.content.model_dump()):
                term = find_disallowed_term(text, extra_terms)
                if term:
                    issues.append(
                        f"{rid}: community/religion term {term!r} in content prose — "
                        "reframe by region / cooking style"
                    )
        return issues

    def check_cuisine_states(self) -> list[str]:
        """Every dish/recipe cuisine_state must have a matching state-level region record
        (once any region records exist)."""
        issues: list[str] = []
        state_regions = {
            r.structure.cuisine_state for r in self.records.values()
            if r.type == "region" and r.structure.level == "state"
        }
        if not state_regions:
            return issues
        for rid, rec in self.records.items():
            if rec.type in ("dish", "recipe"):
                cs = rec.structure.cuisine_state
                if cs != "pan_south" and cs not in state_regions:
                    issues.append(f"{rid}: cuisine_state {cs!r} has no state-level region record")
        return issues

    def check_provenance(self) -> list[str]:
        issues: list[str] = []
        for rid, rec in self.records.items():
            p = rec.provenance
            if rec.type != "source" and not p.synthetic and not p.source_ids \
                    and "hand_authored" not in p.transformations:
                issues.append(f"{rid}: non-synthetic record has no source_ids")
            for sid in p.source_ids:
                if self.records.get(sid) is None:
                    issues.append(f"{rid}: provenance source {sid!r} not in sources")
            for seed in p.seed_record_ids:
                if self.records.get(seed) is None:
                    issues.append(f"{rid}: seed_record_id {seed!r} not found")
            q = rec.quality
            if (q.confidence < 0.5 or q.corroboration_count == 0) and rec.type not in {"source"} \
                    and q.validation_status == "human_validated":
                issues.append(
                    f"{rid}: human_validated but confidence<0.5 / no corroboration (review)"
                )
        return issues

    def check_all(self) -> list[str]:
        return (
            self.check_references()
            + self.check_provenance()
            + self.check_community_tags()
            + self.check_cuisine_states()
        )


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError([f"{path.name}:{lineno}: bad JSON — {exc}"]) from exc


def load_store(canonical_dir: str | Path, *, strict: bool = True) -> CanonicalStore:
    """Parse every ``*.jsonl`` under *canonical_dir*, validate, return a :class:`CanonicalStore`.

    With ``strict=True`` (default) raise :class:`ValidationError` on the first layer
    of per-record failures and on any cross-record contract violation.
    """
    root = Path(canonical_dir)
    store = CanonicalStore()
    issues: list[str] = []

    for rtype, fname in RECORD_FILES.items():
        fpath = root / fname
        if not fpath.exists():
            continue
        for lineno, obj in _iter_jsonl(fpath):
            try:
                rec = parse_record(obj)
            except Exception as exc:  # noqa: BLE001 - collect and report
                issues.append(f"{fname}:{lineno}: {type(exc).__name__}: {exc}")
                continue
            if rec.type != rtype:
                issues.append(f"{fname}:{lineno}: record type {rec.type!r} in wrong file")
            if rec.id in store.records:
                issues.append(f"{fname}:{lineno}: duplicate id {rec.id!r}")
            store.records[rec.id] = rec

    if not issues:
        issues.extend(store.check_all())

    if issues and strict:
        raise ValidationError(issues)
    return store
