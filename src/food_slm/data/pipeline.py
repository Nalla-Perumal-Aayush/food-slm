"""One source URL -> validated, staged draft records (Documents/mcp_extraction_pipeline.md).

`process_source` chains: fetch -> license classify -> build a `source` record ->
extract draft records -> resolve dangling id-refs against (store + this batch) ->
validate (schema + forbidden + panel + corroboration) -> stage.
Nothing reaches `data/canonical/` here — that is `promote_staged` (canon_staging.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from food_slm.data.canon_staging import stage_records
from food_slm.data.extract import ExtractResult, extract_records, slugify
from food_slm.data.forbidden import ForbiddenList
from food_slm.data.ingest import Fetcher, FetchResult
from food_slm.data.licensing import LicenseVerdict, classify_license, verdict_to_source_structure
from food_slm.data.llm import Panel, Provider
from food_slm.data.validate import RecordVerdict, validate_records
from food_slm.schema.store import CanonicalStore

# structure fields holding id-refs that must resolve (subset of store._ID_REF_FIELDS)
_REF_FIELDS = {
    "dish": ["primary_ingredients", "typical_techniques", "regions", "variant_of"],
    "recipe": ["dish_id", "equipment_ids", "outcome_ids"],
    "ingredient": ["common_substitutes"],
    "technique": ["typical_equipment_ids", "precedes", "follows"],
    "term": ["maps_to_id"],
}


def source_id_for(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.replace("www.", "").replace(".", "-")
    tail = slugify(p.path.rsplit("/", 1)[-1] or "index")[:40]
    return f"src:{host}-{tail}".rstrip("-")


def build_source_record(url: str, verdict: LicenseVerdict, *, kind: str,
                        title: str = "") -> dict:
    today = datetime.now(UTC).date().isoformat()
    sid = source_id_for(url)
    return {
        "id": sid, "type": "source", "schema_version": "2",
        "labels": [title or urlparse(url).netloc],
        "content": {"title": title or url, "description": f"Fetched {today}"},
        "structure": verdict_to_source_structure(verdict, kind=kind, accessed_on=today),
        "provenance": {"source_urls": [url], "transformations": ["registered"],
                       "acquired_on": today},
        "quality": {"confidence": verdict.confidence,
                    "validation_status": "auto_validated" if verdict.confidence >= 0.8
                    else "unverified", "corroboration_count": 1},
        "version": {"revision": 1, "created_on": today, "updated_on": today, "status": "active"},
    }


def resolve_batch_refs(records: list[dict], store: CanonicalStore | None) -> tuple[list[dict], list[str]]:
    """Drop any id-ref that resolves neither in *store* nor within this batch.
    Dropped dish/technique refs are appended to a `content` note. Returns (records, dropped)."""
    known = set(store.records) if store else set()
    known |= {r["id"] for r in records}
    dropped: list[str] = []
    for rec in records:
        st = rec.get("structure", {})
        for field_name in _REF_FIELDS.get(rec["type"], []):
            val = st.get(field_name)
            if val is None:
                continue
            if isinstance(val, list):
                keep = [v for v in val if v in known]
                lost = [v for v in val if v not in known]
                st[field_name] = keep
                if lost:
                    dropped += [f"{rec['id']}.{field_name}:{v}" for v in lost]
                    rec.setdefault("content", {}).setdefault("_extraction_notes", [])
                    rec["content"]["_extraction_notes"] = list(
                        {*rec["content"].get("_extraction_notes", []),
                         *(f"unresolved {field_name}: {v.split(':', 1)[1]}" for v in lost)})
            elif val not in known:
                st[field_name] = None
                dropped.append(f"{rec['id']}.{field_name}:{val}")
    # `_extraction_notes` isn't a schema field -> fold into an existing text field or drop it
    for rec in records:
        notes = rec.get("content", {}).pop("_extraction_notes", None)
        if notes and "summary" in rec.get("content", {}):
            rec["content"]["summary"] += "  [" + "; ".join(notes) + "]"
    return records, dropped


def _stamp_envelope(rec: dict, source_id: str, today: str) -> dict:
    """Add the provenance/quality/version envelope to a Claude-Code-extracted record
    that carries only id/type/labels/content/structure."""
    rec.setdefault("schema_version", "2")
    q = rec.setdefault("quality", {})
    q.setdefault("confidence", 0.8)
    q.setdefault("validation_status", "unverified")
    q.setdefault("corroboration_count", 1)
    p = rec.setdefault("provenance", {})
    p.setdefault("source_ids", [source_id])
    p.setdefault("transformations", ["extracted:claude-code"])
    p.setdefault("acquired_on", today)
    p.setdefault("synthetic", False)
    v = rec.setdefault("version", {})
    v.setdefault("revision", 1)
    v.setdefault("created_on", today)
    v.setdefault("updated_on", today)
    v.setdefault("status", "active")
    return rec


def stage_prepared(
    *,
    source_id: str,
    source_url: str,
    source_kind: str,
    license_class: str,
    source_text: str,
    records: list[dict],
    staging_dir,
    store: CanonicalStore | None = None,
    panel: Panel | None = None,
    forbidden: ForbiddenList | None = None,
    heavy_provider: Provider | None = None,
    corroboration: dict[str, int] | None = None,
) -> SourceRun:
    """Stage a batch of *already-extracted* records (e.g. produced by Claude Code) +
    their source record, then run the panel over them. Same gates as `process_source`."""
    from datetime import date

    from food_slm.data.licensing import USABLE_FOR

    today = date.today().isoformat()
    src = {
        "id": source_id, "type": "source", "schema_version": "2",
        "labels": [source_url.split("/")[-1].replace("_", " ") or source_url],
        "content": {"title": source_url, "description": f"Extracted by Claude Code {today}"},
        "structure": {"kind": source_kind, "license": license_class,
                      "license_class": license_class,
                      "usable_for": USABLE_FOR[license_class], "authority": "tertiary",
                      "accessed_on": today},
        "provenance": {"source_urls": [source_url], "transformations": ["registered"],
                       "acquired_on": today},
        "quality": {"confidence": 0.85, "validation_status": "auto_validated",
                    "corroboration_count": 1},
        "version": {"revision": 1, "created_on": today, "updated_on": today, "status": "active"},
    }
    run = SourceRun(url=source_url, source_id=source_id,
                    fetch=FetchResult(url=source_url, status=200, text=source_text))
    run.source_record = src

    stamped = [_stamp_envelope(dict(r), source_id, today) for r in records]
    drafts, run.dropped_refs = resolve_batch_refs(stamped, store)

    # cache the source text so `validate-staged` can score records against it later
    st_dir = Path(staging_dir) / "_source_text"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / f"{source_id.split(':', 1)[1]}.txt").write_text(source_text, encoding="utf-8")
    if panel is not None:
        run.verdicts = validate_records(drafts, source_text, panel=panel, forbidden=forbidden,
                                        heavy_provider=heavy_provider, corroboration=corroboration)
        for v, rec in zip(run.verdicts, drafts, strict=True):
            v.apply_to(rec)
        keep = [rec for v, rec in zip(run.verdicts, drafts, strict=True)
                if v.decision != "rejected"]
    else:
        keep = drafts
    run.staged = stage_records([src, *keep], staging_dir).staged
    return run


@dataclass
class SourceRun:
    url: str
    source_id: str
    fetch: FetchResult
    license: LicenseVerdict | None = None
    source_record: dict | None = None
    extraction: ExtractResult | None = None
    dropped_refs: list[str] = field(default_factory=list)
    verdicts: list[RecordVerdict] = field(default_factory=list)
    staged: list[str] = field(default_factory=list)
    skipped: str = ""

    def summary(self) -> dict:
        by_dec: dict[str, int] = {}
        for v in self.verdicts:
            by_dec[v.decision] = by_dec.get(v.decision, 0) + 1
        return {"url": self.url, "source_id": self.source_id,
                "status": self.fetch.status, "skipped": self.skipped,
                "license": self.license.license_class if self.license else None,
                "extracted": len(self.extraction.drafts) if self.extraction else 0,
                "invalid": len(self.extraction.invalid) if self.extraction else 0,
                "dropped_refs": len(self.dropped_refs),
                "verdicts": by_dec, "staged": len(self.staged)}


def process_cached_text(
    source_id: str,
    source_url: str,
    kind: str,
    text: str,
    license_class: str,
    *,
    title: str,
    extract_provider: Provider,
    panel: Panel,
    staging_dir,
    store: CanonicalStore | None = None,
    forbidden: ForbiddenList | None = None,
    heavy_provider: Provider | None = None,
    corroboration: dict[str, int] | None = None,
) -> SourceRun:
    """Same chain as `process_source` (extract -> resolve refs -> validate -> stage) but
    for text already on disk (a blog post / wiki article / recipe entry we scraped
    earlier) instead of a live fetch. `license_class` is supplied directly — these
    sources were already classified once at collection time (see the per-source
    fetch scripts), so we don't re-derive it from HTML that no longer exists here."""
    from food_slm.data.licensing import USABLE_FOR, LicenseVerdict

    run = SourceRun(url=source_url, source_id=source_id,
                    fetch=FetchResult(url=source_url, status=200, text=text))
    run.license = LicenseVerdict(license_class, "pre-classified at collection",
                                 USABLE_FOR[license_class], [], 0.85)
    run.source_record = build_source_record(source_url, run.license, kind=kind, title=title)

    run.extraction = extract_records(text, source_id, extract_provider)
    drafts = list(run.extraction.drafts)
    drafts, run.dropped_refs = resolve_batch_refs(drafts, store)

    run.verdicts = validate_records(
        drafts, text, panel=panel, forbidden=forbidden,
        heavy_provider=heavy_provider, corroboration=corroboration)
    for v, rec in zip(run.verdicts, drafts, strict=True):
        v.apply_to(rec)

    to_stage = [run.source_record] + [
        rec for v, rec in zip(run.verdicts, drafts, strict=True) if v.decision != "rejected"]
    run.staged = stage_records(to_stage, staging_dir).staged
    return run


def process_source(
    url: str,
    kind: str,
    *,
    fetcher: Fetcher,
    extract_provider: Provider,
    panel: Panel,
    staging_dir,
    store: CanonicalStore | None = None,
    forbidden: ForbiddenList | None = None,
    heavy_provider: Provider | None = None,
    corroboration: dict[str, int] | None = None,
) -> SourceRun:
    run = SourceRun(url=url, source_id=source_id_for(url), fetch=fetcher.fetch(url))
    if not run.fetch.ok:
        run.skipped = run.fetch.reason or f"http {run.fetch.status}"
        return run

    run.license = classify_license(run.fetch.html, url)
    title = (re.search(r"<title[^>]*>([^<]+)</title>", run.fetch.html) or [None, ""])[1].strip()
    run.source_record = build_source_record(url, run.license, kind=kind, title=title)

    run.extraction = extract_records(run.fetch.text, run.source_id, extract_provider)
    drafts = list(run.extraction.drafts)
    drafts, run.dropped_refs = resolve_batch_refs(drafts, store)

    run.verdicts = validate_records(
        drafts, run.fetch.text, panel=panel, forbidden=forbidden,
        heavy_provider=heavy_provider, corroboration=corroboration)
    for v, rec in zip(run.verdicts, drafts, strict=True):
        v.apply_to(rec)

    to_stage = [run.source_record] + [
        rec for v, rec in zip(run.verdicts, drafts, strict=True) if v.decision != "rejected"]
    run.staged = stage_records(to_stage, staging_dir).staged
    return run
