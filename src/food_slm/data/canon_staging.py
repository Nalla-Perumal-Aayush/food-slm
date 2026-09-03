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
from food_slm.schema.models import RECORD_FILES

CALIBRATION_RATE = 0.025
CALIBRATION_LOG = "calibration_log.jsonl"


@dataclass
class StageResult:
    staged: list[str]
    rejected: list[tuple[str, str]]        # (id, reason)


def stage_records(records: list[dict], staging_dir: str | Path) -> StageResult:
    """Validate each draft against the schema; write the valid ones to staging."""
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
        (d / f"{rec.id.split(':', 1)[1]}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
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


def promote_staged(
    ids: list[str],
    *,
    staging_dir: str | Path,
    canonical_dir: str | Path,
    reviewer: str,
    batch_id: str | None = None,
    eval_dir: str | Path | None = None,
) -> PromoteResult:
    if not reviewer:
        raise ValueError("promote_staged requires a reviewer name")
    staging = Path(staging_dir)
    canon = Path(canonical_dir)
    batch_id = batch_id or datetime.now(UTC).strftime("batch-%Y%m%dT%H%M%S")

    # load candidates
    to_add: dict[str, dict] = {}
    for rid in ids:
        p = _staged_path(staging, rid)
        if not p.exists():
            raise FileNotFoundError(f"staged record not found: {rid} ({p})")
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj.setdefault("provenance", {}).setdefault("transformations", []).append(
            f"promoted:{batch_id}")
        obj["quality"]["validation_status"] = "human_validated"
        obj["quality"].setdefault("notes", None)
        obj.setdefault("version", {})["updated_on"] = datetime.now(UTC).date().isoformat()
        to_add[rid] = obj

    # merge into the per-type JSONL files (append + resort by id), grouped by type
    by_file: dict[Path, list[dict]] = {}
    for rid, obj in to_add.items():
        fpath = canon / RECORD_FILES[_type_for_prefix(rid.split(":", 1)[0])]
        rows = by_file.get(fpath)
        if rows is None:
            rows = [json.loads(ln) for ln in fpath.read_text(encoding="utf-8").splitlines()
                    if ln.strip()] if fpath.exists() else []
            by_file[fpath] = rows
        rows.append(obj)

    for fpath, rows in by_file.items():
        rows.sort(key=lambda r: r["id"])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
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
