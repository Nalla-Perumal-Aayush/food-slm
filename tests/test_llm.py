"""Multi-model panel / consensus logic (no network, FakeProvider only)."""
from __future__ import annotations

from pathlib import Path

from food_slm.data.llm import (
    FakeProvider,
    Panel,
    ProviderSpec,
    build_provider,
    load_agents_config,
    parse_json_verdict,
)

ROOT = Path(__file__).resolve().parents[1]


def _fake(name: str, family: str, decision: str) -> FakeProvider:
    spec = ProviderSpec(name=name, kind="fake", model="x", family=family)
    return build_provider(spec, responder=lambda s, p: f'{{"decision": "{decision}", "reason": "r"}}')


def test_unanimous_accept() -> None:
    panel = Panel([_fake("a", "openai", "accept"), _fake("b", "google", "accept"),
                   _fake("c", "open", "accept")])
    res = panel.vote("sys", "prompt", parse=parse_json_verdict)
    assert res.decision == "accept"
    assert res.unanimous and not res.split and not res.needs_human
    assert res.families == 3


def test_split_needs_human() -> None:
    panel = Panel([_fake("a", "openai", "accept"), _fake("b", "google", "reject"),
                   _fake("c", "open", "uncertain")])
    res = panel.vote("sys", "prompt", parse=parse_json_verdict)
    assert res.split and res.needs_human


def test_majority_not_unanimous_still_needs_human() -> None:
    panel = Panel([_fake("a", "openai", "accept"), _fake("b", "google", "accept"),
                   _fake("c", "open", "reject")])
    res = panel.vote("sys", "prompt", parse=parse_json_verdict)
    assert res.decision == "accept"
    assert not res.unanimous and res.needs_human and not res.split


def test_parse_json_verdict_tolerates_fences_and_garbage() -> None:
    assert parse_json_verdict('```json\n{"decision": "ACCEPT"}\n```').decision == "accept"
    assert parse_json_verdict("not json at all").decision == "uncertain"


def test_agents_config_loads_and_excludes_anthropic_from_judge() -> None:
    roles = load_agents_config(ROOT / "configs" / "agents.yaml")
    assert "data_validation" in roles and "model_judge" in roles
    judge_families = {s.family for s in roles["model_judge"]}
    assert "anthropic" not in judge_families           # anti-circularity
    assert any(s.local for s in roles["model_judge"])   # >=1 local model
    assert any(s.local for s in roles["data_validation"])


def test_every_panel_has_multiple_families() -> None:
    roles = load_agents_config(ROOT / "configs" / "agents.yaml")
    for role in ("data_validation", "model_judge"):
        assert len({s.family for s in roles[role]}) >= 2, role
