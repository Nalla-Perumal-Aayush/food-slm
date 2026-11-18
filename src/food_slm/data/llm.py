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
import re
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
        # Heavy local models (deepseek-r1:32b) that spill past 16 GB VRAM onto the CPU
        # can take several minutes per call on the 70 W-capped card — allow a long read.
        timeout_s = float(os.environ.get("OLLAMA_TIMEOUT_S", "600"))
        options = {"temperature": temperature}
        # OLLAMA_NUM_GPU=0 forces CPU-only inference — for running extraction alongside
        # an active GPU training job without any VRAM/power contention (verified: 0
        # measurable VRAM impact). Much slower (~10-13 tok/s vs GPU) but fully isolated.
        num_gpu = os.environ.get("OLLAMA_NUM_GPU")
        if num_gpu is not None:
            options["num_gpu"] = int(num_gpu)
        r = httpx.post(f"{base}/api/chat", timeout=httpx.Timeout(timeout_s, connect=10.0), json={
            "model": self.spec.model, "stream": False,
            "options": options,
            "format": "json" if json_out else None,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"]


class GroqProvider(_Base):
    """Groq's free-tier hosted inference (OpenAI-compatible endpoint). Used only for
    openai/gpt-oss-120b recipe generation on source-gap dishes -- see agents.yaml's
    recipe_generation_remote role. Reasoning-model specific: gpt-oss spends part of
    its token budget on an internal reasoning pass before the visible answer, so
    reasoning_effort is forced to "low" (this task needs a formulaic recipe JSON,
    not deep reasoning) and include_reasoning=false keeps the reasoning trace out of
    the parsed response -- confirmed empirically: ~30-100 reasoning tokens on "low"
    vs. exhausting the whole max_tokens budget with reasoning-only output otherwise.
    """

    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        import httpx

        key = os.environ["GROQ_API_KEY"]
        body: dict[str, Any] = {
            "model": self.spec.model, "temperature": temperature,
            "max_tokens": self.spec.max_tokens,
            "reasoning_effort": "low", "include_reasoning": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }
        if json_out:
            body["response_format"] = {"type": "json_object"}
        r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                       timeout=httpx.Timeout(120.0, connect=10.0),
                       headers={"Authorization": f"Bearer {key}"}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


@dataclass
class FakeProvider(_Base):
    """Deterministic stand-in for tests. ``responder`` maps (system, prompt) -> str."""

    spec: ProviderSpec
    responder: Callable[[str, str], str] = field(default=lambda s, p: "{}")

    def complete(self, system, prompt, *, temperature=0.0, json_out=False) -> str:
        return self.responder(system, prompt)


_KINDS = {
    "anthropic": AnthropicProvider, "openai": OpenAIProvider,
    "google": GoogleProvider, "ollama": OllamaProvider, "groq": GroqProvider,
    "fake": FakeProvider,
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


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str):
    """Pull the first JSON object/array out of a model response.

    Tolerates code fences and a leading ``<think>…</think>`` block (deepseek-r1).
    Returns the parsed value, or ``None`` if nothing parses.
    """
    t = _THINK.sub("", text).strip()
    t = t.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_json_verdict(text: str, *, decision_key: str = "decision",
                       reason_key: str = "reason") -> Verdict:
    """Tolerant JSON verdict parser; defaults to 'uncertain'."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return Verdict(provider="", family="", decision="uncertain", reason="unparseable", raw=text)
    return Verdict(provider="", family="", decision=str(obj.get(decision_key, "uncertain")).lower(),
                   reason=str(obj.get(reason_key, "")), raw=obj)
