"""The frozen forbidden-entity list (baseline §9.2).

A hand-curated set of culinary entities that must **never** appear in any training
data — canonical store, pretraining corpus, synthetic data, or SFT. They are held
back so hallucination / abstention probes are genuine:

- ``withheld_real``  — real Tamil Nadu dishes/ingredients deliberately excluded, to
  measure whether the model *admits it does not know* instead of confabulating.
- ``fictitious``     — plausible-sounding invented names, to catch confident fabrication.

The list is **frozen before data collection**. Its bytes are pinned by
``forbidden_entities.lock.json``; any change requires re-running
``scripts/freeze_forbidden.py`` with a logged rationale. ``verify_lock`` is a hard
gate — training/assembly tooling must call it and refuse to proceed on mismatch.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DIR = Path("data/eval")
LIST_NAME = "forbidden_entities.jsonl"
LOCK_NAME = "forbidden_entities.lock.json"

_WS = re.compile(r"\s+")
_STRIP = re.compile(r"[^\w\s]", re.UNICODE)  # \w keeps Latin + Tamil letters; drop punctuation


def norm_surface(s: str) -> str:
    """Normalize a surface form for matching: NFC, casefold, strip punctuation, collapse ws."""
    s = unicodedata.normalize("NFC", s)
    s = _STRIP.sub(" ", s)
    s = _WS.sub(" ", s).strip().casefold()
    return s


@dataclass(frozen=True)
class ForbiddenEntity:
    id: str
    kind: str                 # withheld_real | fictitious
    entity_type: str          # dish | ingredient | technique | pairing | claim
    surface_forms: tuple[str, ...]
    tamil_forms: tuple[str, ...]
    rationale: str
    probe_intent: str         # abstention | hallucination
    frozen_on: str

    @property
    def normalized(self) -> set[str]:
        return {norm_surface(x) for x in (*self.surface_forms, *self.tamil_forms) if x}


@dataclass
class ForbiddenList:
    entities: list[ForbiddenEntity] = field(default_factory=list)
    _norm_index: dict[str, str] = field(default_factory=dict)  # normalized form -> entity id

    def __post_init__(self) -> None:
        for e in self.entities:
            for n in e.normalized:
                self._norm_index[n] = e.id

    def __len__(self) -> int:
        return len(self.entities)

    def match(self, text: str) -> str | None:
        """Return the entity id whose surface form exactly equals *text* (normalized), else None."""
        return self._norm_index.get(norm_surface(text))

    def scan_text(self, text: str) -> set[str]:
        """Return every forbidden entity id whose normalized surface form occurs as a
        whitespace-delimited phrase inside *text*."""
        hay = f" {norm_surface(text)} "
        hits: set[str] = set()
        for n, eid in self._norm_index.items():
            if f" {n} " in hay:
                hits.add(eid)
        return hits


def load_forbidden(eval_dir: str | Path = DEFAULT_DIR) -> ForbiddenList:
    path = Path(eval_dir) / LIST_NAME
    entities: list[ForbiddenEntity] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            obj = json.loads(line)
            try:
                entities.append(ForbiddenEntity(
                    id=obj["id"],
                    kind=obj["kind"],
                    entity_type=obj["entity_type"],
                    surface_forms=tuple(obj.get("surface_forms", [])),
                    tamil_forms=tuple(obj.get("tamil_forms", [])),
                    rationale=obj.get("rationale", ""),
                    probe_intent=obj["probe_intent"],
                    frozen_on=obj["frozen_on"],
                ))
            except KeyError as exc:
                raise ValueError(f"{LIST_NAME}:{lineno}: missing field {exc}") from exc
    return ForbiddenList(entities)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_lock(eval_dir: str | Path = DEFAULT_DIR) -> dict:
    d = Path(eval_dir)
    fl = load_forbidden(d)
    return {
        "sha256": _sha256(d / LIST_NAME),
        "n_entries": len(fl),
        "kinds": {k: sum(1 for e in fl.entities if e.kind == k)
                  for k in ("withheld_real", "fictitious")},
    }


def verify_lock(eval_dir: str | Path = DEFAULT_DIR) -> None:
    """Raise RuntimeError if the list does not match its lock. Call before any training run."""
    d = Path(eval_dir)
    lock_path = d / LOCK_NAME
    if not lock_path.exists():
        raise RuntimeError(f"{LOCK_NAME} missing — the forbidden list is not frozen")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    actual = _sha256(d / LIST_NAME)
    if lock.get("sha256") != actual:
        raise RuntimeError(
            f"forbidden list changed since freeze (lock {lock.get('sha256', '')[:12]}, "
            f"file {actual[:12]}). Re-run scripts/freeze_forbidden.py --note '...' if intended."
        )


# --- integrity checks against project data --------------------------------

def check_canonical_store(store, forbidden: ForbiddenList) -> list[str]:
    """Return an issue per canonical record that collides with a forbidden entity."""
    issues: list[str] = []
    for rid, rec in store.records.items():
        candidates = list(rec.labels) + list(getattr(rec, "slug_aliases", []))
        candidates.append(rid.split(":", 1)[1].replace("-", " "))
        st = rec.structure.model_dump()
        for k in ("transliteration", "name_ta", "term_ta"):
            if st.get(k):
                candidates.append(st[k])
        for k in ("aliases",):
            candidates.extend(st.get(k, []) or [])
        for c in candidates:
            hit = forbidden.match(c)
            if hit:
                issues.append(f"{rid}: surface form {c!r} is on the forbidden list ({hit})")
    return issues


def scan_corpus_files(paths, forbidden: ForbiddenList, *, chunk_chars: int = 200_000) -> dict:
    """Stream one or more corpus text files and report every forbidden-entity hit.

    Returns ``{"files_scanned", "chars_scanned", "hits": {entity_id: [file, ...]}}``.
    A hit means a held-out probe entity leaked into training data — a hard gate.
    """
    from pathlib import Path as _P

    hits: dict[str, set[str]] = {}
    chars = 0
    files = list(paths)
    for p in files:
        p = _P(p)
        with p.open(encoding="utf-8") as fh:
            carry = ""
            while True:
                block = fh.read(chunk_chars)
                if not block:
                    break
                chars += len(block)
                for eid in forbidden.scan_text(carry + block):
                    hits.setdefault(eid, set()).add(p.name)
                carry = block[-500:]
    return {
        "files_scanned": len(files),
        "chars_scanned": chars,
        "hits": {k: sorted(v) for k, v in sorted(hits.items())},
        "clean": not hits,
    }
