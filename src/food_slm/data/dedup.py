"""Deduplication for extracted records + corpus text (mcp_extraction_pipeline.md §3).

Two layers:
  - **MinHash / LSH** (`datasketch`) — cheap near-exact catch for copied text.
  - **Semantic** (`fastembed` bge-small, dim 384) — "same dish, different words".

For records, the match key is the English name + transliteration + the ingredient
set; a hit means "this is an existing canonical entity, merge don't add".
Embeddings run locally (ONNX, CPU) — no network, no torch/transformers conflict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _shingles(text: str, k: int = 3) -> set[str]:
    toks = _norm(text).split()
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


class MinHashDeduper:
    """Near-exact text dedup via datasketch MinHash LSH (catches verbatim / lightly-edited
    copies of a passage; paraphrases are the SemanticDeduper's job)."""

    def __init__(self, threshold: float = 0.7, num_perm: int = 128) -> None:
        from datasketch import MinHashLSH

        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._n = 0

    def _mh(self, text: str):
        from datasketch import MinHash

        m = MinHash(num_perm=self.num_perm)
        for sh in _shingles(text):
            m.update(sh.encode())
        return m

    def query(self, text: str) -> list[str]:
        return list(self.lsh.query(self._mh(text)))

    def add(self, key: str, text: str) -> None:
        self.lsh.insert(key, self._mh(text))
        self._n += 1

    def is_duplicate(self, text: str) -> tuple[bool, list[str]]:
        hits = self.query(text)
        return bool(hits), hits


@dataclass
class SemanticDeduper:
    """Cosine-similarity dedup over local bge-small embeddings."""

    threshold: float = 0.90
    _keys: list[str] = field(default_factory=list)
    _vecs: list = field(default_factory=list)

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding("BAAI/bge-small-en-v1.5")

    def _embed(self, text: str):
        import numpy as np

        v = next(iter(self._model.embed([text])))
        return np.asarray(v, dtype="float32")

    def add(self, key: str, text: str) -> None:
        self._keys.append(key)
        self._vecs.append(self._embed(text))

    def nearest(self, text: str) -> tuple[str | None, float]:
        import numpy as np

        if not self._vecs:
            return None, 0.0
        q = self._embed(text)
        mat = np.vstack(self._vecs)
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
        i = int(np.argmax(sims))
        return self._keys[i], float(sims[i])

    def is_duplicate(self, text: str) -> tuple[bool, str | None, float]:
        key, score = self.nearest(text)
        return score >= self.threshold, key, score


def record_signature(rec: dict) -> str:
    """Stable text signature for a canonical record — name(s) + key structural fields."""
    labels = rec.get("labels", [])
    st = rec.get("structure", {})
    translits = [n.get("transliteration", "") for n in st.get("names", [])]
    parts = [*labels, *translits]
    if rec.get("type") in ("dish", "recipe"):
        parts += [i if isinstance(i, str) else i.get("ingredient_id", "")
                  for i in (st.get("primary_ingredients") or st.get("ingredients") or [])]
        if st.get("cuisine_state"):
            parts.append(st["cuisine_state"])
    return _norm(" ".join(p for p in parts if p))


@dataclass
class RecordDeduper:
    """Match new draft records against existing canonical records."""

    semantic: SemanticDeduper = field(default_factory=lambda: SemanticDeduper(threshold=0.93))
    _exact: dict[str, str] = field(default_factory=dict)   # normalized-name -> existing id

    def index(self, records: list[dict]) -> None:
        for rec in records:
            sig = record_signature(rec)
            self._exact.setdefault(sig, rec["id"])
            for lbl in rec.get("labels", []):
                self._exact.setdefault(_norm(lbl), rec["id"])
            self.semantic.add(rec["id"], sig)

    def match(self, draft: dict) -> tuple[str, str | None, float]:
        """Return (verdict, matched_id, score). verdict ∈ new | merge | ambiguous."""
        sig = record_signature(draft)
        if sig in self._exact:
            return "merge", self._exact[sig], 1.0
        for lbl in draft.get("labels", []):
            if _norm(lbl) in self._exact:
                return "merge", self._exact[_norm(lbl)], 1.0
        dup, key, score = self.semantic.is_duplicate(sig)
        if dup:
            return "merge", key, score
        if score >= 0.85:
            return "ambiguous", key, score
        return "new", None, score
