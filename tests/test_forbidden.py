"""Forbidden-entity list: lock integrity + leak detection (baseline §9.2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from food_slm.data.forbidden import (
    LIST_NAME,
    check_canonical_store,
    compute_lock,
    load_forbidden,
    norm_surface,
    scan_corpus_files,
    verify_lock,
)
from food_slm.schema import load_store

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
CANON = ROOT / "data" / "canonical"


def test_list_is_frozen_and_matches_lock() -> None:
    verify_lock(EVAL)  # raises on mismatch / missing lock
    lock = json.loads((EVAL / "forbidden_entities.lock.json").read_text(encoding="utf-8"))
    assert lock["sha256"] == compute_lock(EVAL)["sha256"]
    assert lock["n_entries"] == len(load_forbidden(EVAL))


def test_list_has_both_kinds_and_required_fields() -> None:
    fl = load_forbidden(EVAL)
    kinds = {e.kind for e in fl.entities}
    assert kinds == {"withheld_real", "fictitious"}
    for e in fl.entities:
        assert e.surface_forms, e.id
        assert e.probe_intent in {"abstention", "hallucination"}


def test_seed_canonical_store_has_no_forbidden_leak() -> None:
    store = load_store(CANON, strict=True)
    assert check_canonical_store(store, load_forbidden(EVAL)) == []


def test_leak_is_detected(tmp_path: Path) -> None:
    store = load_store(CANON, strict=True)
    fl = load_forbidden(EVAL)
    victim = next(iter(store.records.values()))
    object.__setattr__(victim, "labels", [*victim.labels, "Arachuvitta Sambar"])
    leaks = check_canonical_store(store, fl)
    assert any("fe:0001" in x for x in leaks)


def test_scan_text_matches_phrase_not_substring() -> None:
    fl = load_forbidden(EVAL)
    assert fl.scan_text("a bowl of kollu rasam for winter") == {"fe:0003"}
    assert fl.scan_text("this is unrelated prose about idli") == set()


def test_norm_surface_is_punctuation_and_case_insensitive() -> None:
    assert norm_surface("Ennai-Kathirikai  Kuzhambu!") == norm_surface("ennai kathirikai kuzhambu")


def test_scan_corpus_files_flags_leak_and_passes_clean(tmp_path: Path) -> None:
    fl = load_forbidden(EVAL)
    clean = tmp_path / "clean.txt"
    clean.write_text("Idli and dosa are steamed and griddled breakfast staples.\n" * 50,
                     encoding="utf-8")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("A traditional recipe for kollu rasam served in winter.\n" * 20,
                     encoding="utf-8")

    ok = scan_corpus_files([clean], fl)
    assert ok["clean"] and ok["hits"] == {}

    bad = scan_corpus_files([clean, dirty], fl, chunk_chars=128)  # tiny chunks -> exercise carry
    assert not bad["clean"]
    assert bad["hits"] == {"fe:0003": ["dirty.txt"]}


def test_lock_detects_tampering(tmp_path: Path) -> None:
    d = tmp_path / "eval"
    d.mkdir()
    (d / LIST_NAME).write_text(
        json.dumps({"id": "fe:x", "kind": "fictitious", "entity_type": "dish",
                    "surface_forms": ["X"], "probe_intent": "hallucination",
                    "frozen_on": "2026-09-02"}) + "\n",
        encoding="utf-8",
    )
    (d / "forbidden_entities.lock.json").write_text(
        json.dumps({"sha256": "0" * 64, "n_entries": 1}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since freeze"):
        verify_lock(d)
