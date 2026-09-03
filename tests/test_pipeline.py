"""Extraction-pipeline spine: licensing, coverage, staging (no network, no models)."""
from __future__ import annotations

from pathlib import Path

import pytest

from food_slm.data.canon_staging import (
    CALIBRATION_RATE,
    list_staged,
    promote_staged,
    stage_records,
)
from food_slm.data.coverage import compute_coverage, list_gaps, load_targets
from food_slm.data.licensing import classify_license
from food_slm.schema import load_store

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"
TARGETS = ROOT / "configs" / "coverage_targets.yaml"


# -- licensing ----------------------------------------------------------------

@pytest.mark.parametrize(("html", "cls", "corpus_ok"), [
    ('<a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/">x</a>',
     "share_alike", True),
    ('<link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/">',
     "public_domain", True),
    ('license: "https://creativecommons.org/licenses/by-nc/4.0/"', "noncommercial", False),
    ("<footer>© 2025 Tasty Recipes. All Rights Reserved.</footer>", "all_rights_reserved", False),
    ("<p>A nice recipe blog with no license info at all.</p>", "unknown", False),
])
def test_classify_license(html: str, cls: str, corpus_ok: bool) -> None:
    v = classify_license(html, "https://example.com/recipe")
    assert v.license_class == cls
    assert v.corpus_eligible is corpus_ok
    assert "reference_only" in v.usable_for or corpus_ok


def test_unknown_defaults_to_reference_only() -> None:
    v = classify_license("nothing here", "https://foo.example/x")
    assert v.usable_for == ["reference_only"]
    assert not v.corpus_eligible


# -- coverage / list_gaps -------------------------------------------------

def test_coverage_report_on_seed_store() -> None:
    store = load_store(CANON, strict=True)
    targets = load_targets(TARGETS)
    report = compute_coverage(store, targets)
    assert report.totals["dish"] >= 1
    assert report.dish_cells.get("tamil_nadu/tiffin", 0) >= 1   # ven pongal
    assert "heat_dry" in report.technique_families
    assert sum(report.totals.values()) == len(store)


def test_list_gaps_ranks_ingredient_and_technique_shortfalls_first() -> None:
    report, gaps = list_gaps(CANON, TARGETS, limit=50)
    assert gaps, "seed store is tiny -> everything is a gap"
    # the seed store has ~1 dish; ingredient-category / technique-family minimums dominate
    kinds = [g.kind for g in gaps[:10]]
    assert "ingredient_category" in kinds or "technique_family" in kinds
    for g in gaps:
        assert g.have < g.target and g.deficit > 0


# -- staging / promote ----------------------------------------------------

def _term(id_: str, translit: str) -> dict:
    return {
        "id": id_, "type": "term", "schema_version": "2", "labels": [translit],
        "content": {"gloss": f"gloss for {translit}"},
        "structure": {"lang": "ta", "native": "தமிழ்", "transliteration": translit,
                      "category": "general"},
        "provenance": {"source_ids": ["src:hand-authored-seed"],
                       "transformations": ["extracted"], "acquired_on": "2026-09-02"},
        "quality": {"confidence": 0.9, "validation_status": "unverified",
                    "corroboration_count": 2},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }


def test_stage_then_promote_into_a_copy_of_the_store(tmp_path: Path) -> None:
    # work on a copy so the real seed store is untouched
    canon = tmp_path / "canonical"
    canon.mkdir()
    for p in CANON.glob("*.jsonl"):
        (canon / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    staging = tmp_path / "staging"

    drafts = [_term("term:kootu", "kootu"), _term("term:poriyal", "poriyal"),
              {"id": "term:bad", "type": "term"}]  # invalid -> rejected
    sr = stage_records(drafts, staging)
    assert set(sr.staged) == {"term:kootu", "term:poriyal"}
    assert sr.rejected and sr.rejected[0][0] == "term:bad"
    assert len(list_staged(staging)) == 2

    res = promote_staged(["term:kootu", "term:poriyal"], staging_dir=staging,
                         canonical_dir=canon, reviewer="aayush",
                         batch_id="b1", eval_dir=tmp_path / "eval")
    assert res.store_ok, res.issues
    assert set(res.promoted) == {"term:kootu", "term:poriyal"}
    store = load_store(canon, strict=True)
    assert store.get("term:kootu").quality.validation_status == "human_validated"
    assert not list_staged(staging)                       # promoted -> removed from staging
    assert (tmp_path / "eval" / "calibration_log.jsonl").exists()


def test_promote_requires_reviewer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reviewer"):
        promote_staged([], staging_dir=tmp_path, canonical_dir=tmp_path, reviewer="")


def test_calibration_sample_is_deterministic_and_small() -> None:
    from food_slm.data.canon_staging import _calibration_ids

    ids = [f"dish:d{i}" for i in range(4000)]
    a = _calibration_ids(ids, "batch-x")
    b = _calibration_ids(ids, "batch-x")
    assert a == b
    assert 0.5 * CALIBRATION_RATE < len(a) / len(ids) < 2 * CALIBRATION_RATE
