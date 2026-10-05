"""Staging + the single gate into the canonical store (mcp_extraction_pipeline.md §2).

Extracted / validated drafts land in ``data/interim/canon_staging/<type>/<id>.json``.
``promote_staged(ids, reviewer)`` is the **only** path into ``data/canonical/`` — it
re-validates the whole store after the merge and refuses if integrity breaks.
A deterministic **2.5 % calibration sample** of every promoted batch is logged for
human review (Documents/evaluation_strategy.md §4).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from food_slm.schema import load_store, parse_record
from food_slm.schema.models import RECORD_FILES, find_disallowed_term, iter_strings
from food_slm.schema.store import _ID_REF_FIELDS

CALIBRATION_RATE = 0.025
CALIBRATION_LOG = "calibration_log.jsonl"


@dataclass
class StageResult:
    staged: list[str]
    rejected: list[tuple[str, str]]        # (id, reason)


def _merge_with_existing(obj: dict, existing_path: Path) -> dict:
    """A second (or third...) source independently produced a record with the same
    id — union their `provenance.source_ids` so `corroboration_count` (len of that
    list) reflects genuine cross-source agreement, per the AI-primary/human-anchored
    design (evaluation_strategy.md §3: "cross-source corroboration (not a model)").
    Keeps the newer record's content/verdict (usually the more complete one) but
    never loses a source_id or drops below a status the earlier version already
    reached (never downgrade auto_validated/human_validated back to unverified)."""
    try:
        prev = json.loads(existing_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return obj
    prev_ids = prev.get("provenance", {}).get("source_ids", [])
    new_ids = obj.setdefault("provenance", {}).setdefault("source_ids", [])
    merged_ids = list(dict.fromkeys([*prev_ids, *new_ids]))
    obj["provenance"]["source_ids"] = merged_ids
    q = obj.setdefault("quality", {})
    q["corroboration_count"] = max(len(merged_ids), q.get("corroboration_count", 0),
                                   prev.get("quality", {}).get("corroboration_count", 0))
    _rank = {"unverified": 0, "needs_human": 0, "auto_validated": 1, "human_validated": 2,
            "rejected": -1}
    prev_status = prev.get("quality", {}).get("validation_status", "unverified")
    if _rank.get(prev_status, 0) > _rank.get(q.get("validation_status", "unverified"), 0):
        q["validation_status"] = prev_status
    return obj


def stage_records(records: list[dict], staging_dir: str | Path) -> StageResult:
    """Validate each draft against the schema; write the valid ones to staging.
    A record whose id is already staged (from an earlier source) is merged, not
    clobbered — see `_merge_with_existing`."""
    out = Path(staging_dir)
    staged, rejected = [], []
    for obj in records:
        rid = obj.get("id", "<no-id>")
        try:
            rec = parse_record(obj)
        except Exception as exc:  # noqa: BLE001
            rejected.append((rid, f"{type(exc).__name__}: {exc}"))
            continue
        obj.setdefault("quality", {}).setdefault("validation_status", "auto_validated")
        d = out / rec.type
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{rec.id.split(':', 1)[1]}.json"
        if p.exists():
            obj = _merge_with_existing(obj, p)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.append(rec.id)
    return StageResult(staged, rejected)


def list_staged(staging_dir: str | Path) -> list[dict]:
    out = Path(staging_dir)
    recs = []
    for p in sorted(out.rglob("*.json")):
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    return recs


_PREFIX_TYPE = {
    "src": "source", "ing": "ingredient", "dish": "dish", "recipe": "recipe",
    "tech": "technique", "equip": "equipment", "region": "region", "state": "state",
    "outcome": "outcome", "constraint": "constraint", "problem": "problem",
    "term": "term", "rel": "rel",
}


def _type_for_prefix(prefix: str) -> str:
    return _PREFIX_TYPE[prefix]


def _staged_path(staging_dir: Path, rid: str) -> Path:
    rtype, slug = rid.split(":", 1)
    return staging_dir / _type_for_prefix(rtype) / f"{slug}.json"


def _calibration_ids(promoted: list[str], batch_id: str) -> list[str]:
    """Deterministic ~2.5% sample: hash(batch_id + id) < rate."""
    picked = []
    for rid in sorted(promoted):
        h = int(hashlib.sha256(f"{batch_id}:{rid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < CALIBRATION_RATE:
            picked.append(rid)
    return picked


@dataclass
class PromoteResult:
    promoted: list[str]
    calibration_sample: list[str]
    store_ok: bool
    issues: list[str]


def _content_flagged(obj: dict) -> bool:
    """True if any string in `content` trips the community/religion-term scan
    (Rev 5 scope). Per-record schema validation already checks labels/structure;
    this catches it in prose too, same as `CanonicalStore.check_community_tags`."""
    for text in iter_strings(obj.get("content", {})):
        if find_disallowed_term(text):
            return True
    return False


def _refs_of(obj: dict) -> list[str]:
    out = list(obj.get("provenance", {}).get("source_ids", []))
    struct = obj.get("structure", {}) or {}
    for field_name in _ID_REF_FIELDS.get(obj.get("type", ""), []):
        val = struct.get(field_name)
        if val is None:
            continue
        out.extend(val if isinstance(val, list) else [val])
    return out


# A `recipe` record can't accumulate corroboration the way a flat entity (ingredient,
# term, ...) does: every source produces its OWN `recipe:{dish}-{source}` id, so two
# sources never cite the same record -- corroboration_count structurally never
# reaches min_corroboration for recipes, even a well-extracted one. validate.py's
# RecordVerdict.apply_to() already bumps quality.confidence to >= 0.85 if and only if
# the data_validation panel unanimously accepted the record (confirmed empirically:
# every staged recipe today is at exactly 0.85 or 0.55, nothing in between) -- so for
# type=recipe specifically, that confidence floor substitutes for the corroboration
# bar below, rather than requiring independent-source agreement that structurally
# can't happen here.
_RECIPE_CONFIDENCE_ANCHOR = 0.8


def select_promotable(
    staged: list[dict],
    already_canonical: set[str],
    min_corroboration: int = 2,
) -> tuple[list[str], dict[str, dict]]:
    """Trust-anchored transitive closure — the "no dangling reference" way to pick
    a partial slice of staged records to promote (mcp_extraction_pipeline.md §2).

    An **anchor** record is trusted directly: every `source` record (a provenance
    envelope, not a factual claim), any record with `corroboration_count >=
    min_corroboration` (independent sources agreeing — the AI-primary validation
    signal, evaluation_strategy.md §3), any already `auto_validated`/`human_validated`
    record, or a `recipe` whose panel unanimously accepted it (see
    `_RECIPE_CONFIDENCE_ANCHOR` above — corroboration doesn't apply to recipes the
    same way).

    Promoting only anchors would leave every supporting ingredient/technique/term/
    source they reference (usually single-source, still `needs_human`) behind in
    staging — creating a dangling reference the moment the anchor lands in the
    canonical store. So this pulls in the full closure: everything a selected
    record structurally depends on (`schema.store._ID_REF_FIELDS`) or cites via
    `provenance.source_ids`, recursively, as long as it isn't rejected or itself
    community/religion-term-flagged.

    A `rel` record needs both endpoints to resolve or it's meaningless, and a
    non-synthetic record needs at least one real source_id — either one dropped
    by the closure disqualifies the record too, which can itself orphan whatever
    depended on *it*, so this iterates to a fixpoint. Whatever's still unresolved
    after that (source blocklisted, a dependency's source content was flagged, ...)
    is stripped from the surviving record's structure/provenance instead of
    blocking that record — same pattern as the manual contamination cleanup this
    session already used, just applied consistently at promotion time.

    Returns `(ids_to_promote, patches)` — `patches` maps id -> {"structure": {...},
    "provenance": {...}} field overrides to apply on top of the staged object.
    """
    by_id = {r["id"]: r for r in staged}
    unusable = {
        r["id"] for r in staged
        if r.get("quality", {}).get("validation_status") == "rejected" or _content_flagged(r)
    }

    def is_anchor(r: dict) -> bool:
        if r["id"] in unusable:
            return False
        if r["type"] == "source":
            return True
        q = r.get("quality", {})
        if (q.get("corroboration_count", 0) >= min_corroboration
                or q.get("validation_status") in ("auto_validated", "human_validated")):
            return True
        if r["type"] == "recipe" and not r.get("provenance", {}).get("synthetic"):
            # synthetic (LLM-generated, no source document) recipes are deliberately
            # excluded even at confidence 0.85 -- they should never auto-promote
            # without an actual human look, per extract_recipe.generate_recipe's design.
            return q.get("confidence", 0.0) >= _RECIPE_CONFIDENCE_ANCHOR
        return False

    selected = {rid for rid, r in by_id.items() if is_anchor(r)}

    changed = True
    while changed:
        changed = False
        for rid in list(selected):
            for ref in _refs_of(by_id[rid]):
                if ref in selected or ref in already_canonical or ref in unusable:
                    continue
                if ref in by_id:
                    selected.add(ref)
                    changed = True

    # fixpoint prune: drop anything that can no longer stand on its own once
    # unresolved refs are accounted for, then re-check what depended on it
    changed = True
    while changed:
        changed = False
        resolvable = already_canonical | selected
        for rid in list(selected):
            r = by_id[rid]
            if r["type"] == "rel":
                st = r.get("structure", {})
                if st.get("subject_id") not in resolvable or st.get("object_id") not in resolvable:
                    selected.discard(rid)
                    changed = True
                    continue
            prov = r.get("provenance", {})
            has_source = any(s in resolvable for s in prov.get("source_ids", []))
            if not has_source and not prov.get("synthetic") \
                    and "hand_authored" not in prov.get("transformations", []) \
                    and r["type"] != "source":
                selected.discard(rid)
                changed = True

    # final patch pass: strip whatever reference still doesn't resolve, now that
    # `selected` is stable
    resolvable = already_canonical | selected
    patches: dict[str, dict] = {}
    for rid in selected:
        r = by_id[rid]
        patch: dict = {}
        sids = r.get("provenance", {}).get("source_ids", [])
        kept_sids = [s for s in sids if s in resolvable]
        if kept_sids != sids:
            patch["provenance"] = {"source_ids": kept_sids}
        struct_patch = {}
        struct = r.get("structure", {}) or {}
        for field_name in _ID_REF_FIELDS.get(r["type"], []):
            val = struct.get(field_name)
            if val is None:
                continue
            if isinstance(val, list):
                kept = [v for v in val if v in resolvable]
                if kept != val:
                    struct_patch[field_name] = kept
            elif val not in resolvable:
                struct_patch[field_name] = None
        if struct_patch:
            patch["structure"] = struct_patch
        if patch:
            patches[rid] = patch

    return sorted(selected), patches


def promote_staged(
    ids: list[str],
    *,
    staging_dir: str | Path,
    canonical_dir: str | Path,
    reviewer: str,
    batch_id: str | None = None,
    eval_dir: str | Path | None = None,
    patches: dict[str, dict] | None = None,
) -> PromoteResult:
    if not reviewer:
        raise ValueError("promote_staged requires a reviewer name")
    staging = Path(staging_dir)
    canon = Path(canonical_dir)
    batch_id = batch_id or datetime.now(UTC).strftime("batch-%Y%m%dT%H%M%S")
    patches = patches or {}

    # load candidates
    to_add: dict[str, dict] = {}
    for rid in ids:
        p = _staged_path(staging, rid)
        if not p.exists():
            raise FileNotFoundError(f"staged record not found: {rid} ({p})")
        obj = json.loads(p.read_text(encoding="utf-8"))
        for key, overrides in patches.get(rid, {}).items():
            obj.setdefault(key, {}).update(overrides)
        obj.setdefault("provenance", {}).setdefault("transformations", []).append(
            f"promoted:{batch_id}")
        obj["quality"]["validation_status"] = "human_validated"
        obj["quality"].setdefault("notes", None)
        obj.setdefault("version", {})["updated_on"] = datetime.now(UTC).date().isoformat()
        to_add[rid] = obj

    # merge into the per-type JSONL files, keyed by id so re-promoting (or a
    # record already present from an earlier batch) overwrites in place instead
    # of duplicating the row.
    by_file: dict[Path, dict[str, dict]] = {}
    for rid, obj in to_add.items():
        fpath = canon / RECORD_FILES[_type_for_prefix(rid.split(":", 1)[0])]
        rows = by_file.get(fpath)
        if rows is None:
            rows = {}
            if fpath.exists():
                for ln in fpath.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        existing = json.loads(ln)
                        rows[existing["id"]] = existing
            by_file[fpath] = rows
        rows[rid] = obj

    for fpath, rows in by_file.items():
        ordered = sorted(rows.values(), key=lambda r: r["id"])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n",
                         encoding="utf-8")

    # verify store integrity post-merge (non-strict so we can report, not raise)
    store = load_store(canon, strict=False)
    issues = store.check_all()
    store_ok = not issues

    sample = _calibration_ids(list(to_add), batch_id)
    if eval_dir is not None:
        log = Path(eval_dir) / CALIBRATION_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "batch_id": batch_id, "reviewer": reviewer,
                "promoted": len(to_add), "calibration_sample": sample,
                "store_ok": store_ok, "issues": issues[:20],
            }) + "\n")

    if store_ok:
        for rid in to_add:
            _staged_path(staging, rid).unlink(missing_ok=True)

    return PromoteResult(list(to_add), sample, store_ok, issues)
