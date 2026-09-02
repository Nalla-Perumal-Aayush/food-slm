"""Provider-agnostic multi-model LLM client (baseline §9; Documents/mcp_extraction_pipeline.md §5).

One thin wrapper per provider (Anthropic / OpenAI / Google / local Ollama), a
:class:`FakeProvider` for tests, and a :class:`Panel` that runs several *different
model families* and reports consensus. Roles → providers are wired in
``configs/agents.yaml``.

SDKs are imported lazily and API keys read from the environment, so importing this
module never requires network access or credentials.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from food_slm.utils import load_yaml


@dataclass(frozen=True)
class ProviderSpec:
    name: str                      # free label, e.g. "claude-sonnet"
    kind: str                      # anthropic | openai | google | ollama | fake
    model: str
    family: str                   # anthropic | openai | google | open  -- for anti-circularity
    local: bool = False
    max_tokens: int = 2048


class Provider(Protocol):
    spec: ProviderSpec

    def complete(self, system: str, prompt: str, *, temperature: float = 0.0,
                 json_out: bool = False) -> str: ...


# --- concrete providers (lazy SDK import) ----------------------------------

class _Base:
    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec


class AnthropicProvider(_Base):
    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=self.spec.model, max_tokens=self.spec.max_tokens, system=system,
            temperature=temperature, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class OpenAIProvider(_Base):
    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        kw = {"response_format": {"type": "json_object"}} if json_out else {}
        r = client.chat.completions.create(
            model=self.spec.model, temperature=temperature, max_tokens=self.spec.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}], **kw,
        )
        return r.choices[0].message.content or ""


class GoogleProvider(_Base):
    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(self.spec.model, system_instruction=system)
        r = model.generate_content(
            prompt, generation_config={"temperature": temperature,
                                       "max_output_tokens": self.spec.max_tokens})
        return r.text


class OllamaProvider(_Base):
    """Local, fully offline (Ollama / llama.cpp OpenAI-compatible endpoint)."""

    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        import httpx

        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = httpx.post(f"{base}/api/chat", timeout=120, json={
            "model": self.spec.model, "stream": False,
            "options": {"temperature": temperature},
            "format": "json" if json_out else None,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"]


@dataclass
class FakeProvider(_Base):
    """Deterministic stand-in for tests. ``responder`` maps (system, prompt) -> str."""

    spec: ProviderSpec
    responder: Callable[[str, str], str] = field(default=lambda s, p: "{}")

    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        return self.responder(system, prompt)


_KINDS = {
    "anthropic": AnthropicProvider, "openai": OpenAIProvider,
    "google": GoogleProvider, "ollama": OllamaProvider, "fake": FakeProvider,
}


def build_provider(spec: ProviderSpec, **kw) -> Provider:
    cls = _KINDS[spec.kind]
    return cls(spec, **kw) if spec.kind == "fake" else cls(spec)


def load_agents_config(path: str | os.PathLike) -> dict[str, list[ProviderSpec]]:
    """``configs/agents.yaml`` -> {role: [ProviderSpec, ...]}."""
    raw = load_yaml(path)
    out: dict[str, list[ProviderSpec]] = {}
    for role, entries in raw.get("roles", {}).items():
        out[role] = [ProviderSpec(**e) for e in entries]
    return out


# --- panel / consensus ---------------------------------------------------

@dataclass
class Verdict:
    provider: str
    family: str
    decision: str                 # e.g. accept | reject | uncertain
    reason: str = ""
    raw: Any = None


@dataclass
class ConsensusResult:
    decision: str                 # majority decision, or "split"
    unanimous: bool
    split: bool
    verdicts: list[Verdict]
    families: int                 # number of distinct model families that voted

    @property
    def needs_human(self) -> bool:
        return self.split or not self.unanimous


class Panel:
    """Runs the same query across providers of *different families* and tallies verdicts."""

    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers

    def vote(self, system: str, prompt: str, *, parse: Callable[[str], Verdict],
             temperature: float = 0.0) -> ConsensusResult:
        verdicts: list[Verdict] = []
        for p in self.providers:
            text = p.complete(system, prompt, temperature=temperature, json_out=True)
            v = parse(text)
            v.provider, v.family = p.spec.name, p.spec.family
            verdicts.append(v)
        tally = Counter(v.decision for v in verdicts)
        top, n = tally.most_common(1)[0]
        unanimous = len(tally) == 1
        split = n <= len(verdicts) / 2 and len(verdicts) > 1
        return ConsensusResult(
            decision=top if not split else "split",
            unanimous=unanimous, split=split, verdicts=verdicts,
            families=len({v.family for v in verdicts}),
        )


def parse_json_verdict(text: str, *, decision_key: str = "decision",
                       reason_key: str = "reason") -> Verdict:
    """Tolerant JSON verdict parser: strips code fences, defaults to 'uncertain'."""
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return Verdict(provider="", family="", decision="uncertain", reason="unparseable", raw=text)
    return Verdict(provider="", family="", decision=str(obj.get(decision_key, "uncertain")).lower(),
                   reason=str(obj.get(reason_key, "")), raw=obj)
