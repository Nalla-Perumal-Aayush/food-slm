"""Validation: schema + forbidden gates + multi-family panel (FakeProvider, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from food_slm.data.forbidden import load_forbidden
from food_slm.data.llm import FakeProvider, Panel, ProviderSpec
from food_slm.data.validate import summarize, validate_records

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = load_forbidden(ROOT / "data" / "eval")


def _panel(*decisions: str) -> Panel:
    return Panel([
        FakeProvider(ProviderSpec(f"m{i}", "fake", "x", fam),
                     responder=(lambda d: (lambda s, p: f'{{"decision":"{d}","reason":"r"}}'))(dec))
        for i, (dec, fam) in enumerate(zip(decisions, ["qwen", "gemma", "deepseek"], strict=False))
    ])


def _draft(rid: str = "term:kootu", *, confidence: float = 0.85, corr: int = 2, **struct) -> dict:
    return {
        "id": rid, "type": "term", "schema_version": "2", "labels": [rid.split(":")[1]],
        "content": {"gloss": "a stew of lentils and vegetables"},
        "structure": {"lang": "ta", "native": "கூட்டு", "transliteration": "kootu",
                      "category": "dish", **struct},
        "provenance": {"source_ids": ["src:a", "src:b"][:corr], "transformations": ["extracted"],
                       "acquired_on": "2026-09-02"},
        "quality": {"confidence": confidence, "validation_status": "unverified",
                    "corroboration_count": 0},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }


def test_unanimous_accept_with_corroboration_auto_validates() -> None:
    v = validate_records([_draft(corr=2)], "kootu is a Tamil stew of dal and vegetables",
                         panel=_panel("accept", "accept", "accept"))[0]
    assert v.decision == "auto_validated"
    rec = v.apply_to(_draft(corr=2))
    assert rec["quality"]["validation_status"] == "auto_validated"


def test_unanimous_accept_without_corroboration_needs_human() -> None:
    v = validate_records([_draft(corr=1)], "kootu source text",
                         panel=_panel("accept", "accept", "accept"))[0]
    assert v.decision == "needs_human"
    assert any("corroboration" in r for r in v.reasons)


def test_low_confidence_always_needs_human() -> None:
    v = validate_records([_draft(confidence=0.5, corr=3)], "kootu source text",
                         panel=_panel("accept", "accept", "accept"))[0]
    assert v.decision == "needs_human" and "confidence<0.8" in v.reasons


def test_unanimous_reject() -> None:
    v = validate_records([_draft()], "an off-topic page about car engines",
                         panel=_panel("reject", "reject", "reject"))[0]
    assert v.decision == "rejected"


def test_split_without_tiebreak_needs_human() -> None:
    v = validate_records([_draft()], "ambiguous text",
                         panel=_panel("accept", "reject", "uncertain"))[0]
    assert v.decision == "needs_human"


def test_split_with_heavy_tiebreak() -> None:
    heavy = FakeProvider(ProviderSpec("deepseek", "fake", "x", "deepseek"),
                         responder=lambda s, p: '{"decision":"accept","reason":"ok"}')
    v = validate_records([_draft()], "text", panel=_panel("accept", "reject", "uncertain"),
                         heavy_provider=heavy)[0]
    assert "tiebreak:accept" in v.reasons
    assert v.decision == "needs_human"     # 2 accept / 1 reject / 1 uncertain -> not unanimous


def test_schema_invalid_draft_is_rejected_before_panel() -> None:
    bad = {"id": "term:x", "type": "term", "schema_version": "2", "labels": ["x"],
           "content": {}, "structure": {}, "provenance": {}, "quality": {},
           "version": {"created_on": "2026-09-02", "updated_on": "2026-09-02"}}
    v = validate_records([bad], "text", panel=_panel("accept", "accept", "accept"))[0]
    assert v.decision == "rejected" and not v.schema_ok


def test_forbidden_entity_is_rejected() -> None:
    d = _draft("term:kollu-rasam")
    d["labels"] = ["Kollu Rasam"]
    v = validate_records([d], "Kollu rasam is a horse gram rasam.",
                         panel=_panel("accept", "accept", "accept"), forbidden=FORBIDDEN)[0]
    assert v.decision == "rejected" and v.forbidden_hit == "fe:0003"


def test_summarize() -> None:
    vs = validate_records(
        [_draft("term:a", corr=2), _draft("term:b", corr=1)],
        "text", panel=_panel("accept", "accept", "accept"))
    assert summarize(vs) == {"auto_validated": 1, "needs_human": 1}


def _coherent_term() -> dict:
    return {
        "id": "term:kuzhambu", "type": "term", "schema_version": "2", "labels": ["kuzhambu"],
        "content": {"gloss": "A Tamil tamarind-based gravy, thinner than a curry, eaten with rice."},
        "structure": {"lang": "ta", "native": "குழம்பு", "transliteration": "kuzhambu",
                      "category": "dish"},
        "provenance": {"source_ids": ["src:a", "src:b"], "transformations": ["extracted"],
                       "acquired_on": "2026-09-02"},
        "quality": {"confidence": 0.85, "validation_status": "unverified",
                    "corroboration_count": 0},
        "version": {"revision": 1, "created_on": "2026-09-02", "updated_on": "2026-09-02"},
    }


@pytest.mark.slow
def test_live_ollama_panel_on_true_and_false_records() -> None:
    """Real local Qwen + Gemma panel: accepts a source-consistent record, rejects a
    contradicted one."""
    httpx = pytest.importorskip("httpx")
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        pytest.skip("ollama not running")
    from food_slm.data.llm import build_provider

    panel = Panel([
        build_provider(ProviderSpec("qwen", "ollama", "qwen2.5:7b-instruct", "qwen", local=True)),
        build_provider(ProviderSpec("gemma", "ollama", "gemma2:9b-instruct-q4_K_M", "gemma",
                                    local=True)),
    ])
    src = ("Kuzhambu is a tangy Tamil Nadu gravy soured with tamarind, thinner than a kootu, "
           "served over hot rice with a spoon of ghee.")
    good = validate_records([_coherent_term()], src, panel=panel)[0]
    assert good.panel is not None and good.panel.families == 2
    assert good.decision in {"auto_validated", "needs_human"}   # not a hard reject

    fake = _coherent_term()
    fake["content"]["gloss"] = "A sweet coconut-milk dessert served chilled at weddings."
    bad = validate_records([fake], src, panel=panel)[0]
    assert bad.decision == "rejected" or "reject" in " ".join(bad.reasons)
