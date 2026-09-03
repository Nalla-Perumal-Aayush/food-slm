"""Page text -> draft canonical records (mcp_extraction_pipeline.md §2).

Two entry points:
  * :func:`build_extraction_prompt` — for **Claude Code** to run the extraction
    directly (highest quality, no separate model).
  * :func:`extract_records` — run it through any :class:`~food_slm.data.llm.Provider`
    (e.g. a local Ollama model) for scale.

Both produce the same shape: a list of **draft** schema-v2 record dicts with
provenance stamped and ``validation_status: "unverified"``. Nothing is committed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

from food_slm.data.llm import Provider, extract_json
from food_slm.schema.models import parse_record


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "unnamed"


_SYSTEM = """You extract STRUCTURED CULINARY FACTS from a web page about South Indian food.
Return ONLY JSON. Do not invent facts not supported by the text. If the page is not about
South Indian cooking, return {"dishes": [], "ingredients": [], "techniques": [], "terms": [],
"notes": "off-topic"}.

Rules:
- English names are the primary name. Put any native-script name in `names` as
  {lang: ta|ml|kn|te, native: "...", transliteration: "popular romanization"}.
- Never tag by community, caste or religion. Use region / cooking style only.
- `cuisine_state` is one of: tamil_nadu, kerala, karnataka, andhra_pradesh, telangana,
  puducherry, pan_south.
- For a dish, `diet` = {vegetarian, contains_dairy, contains_egg, contains_seafood,
  contains_poultry, contains_red_meat} booleans (vegetarian=false if any meat/egg).

JSON shape:
{
 "dishes": [{"name_en","names":[],"category","cuisine_state","style"?,"diet":{...},
             "summary","primary_ingredients":["english name",...]}],
 "ingredients": [{"name_en","names":[],"category","default_unit","summary"}],
 "techniques": [{"name_en","names":[],"family","summary"}],
 "terms": [{"term_en"?,"lang","native","transliteration","gloss","category"}],
 "notes": "anything uncertain or needing human review"
}"""


def build_extraction_prompt(page_text: str, source_id: str, *, max_chars: int = 12000) -> tuple:
    user = (f"SOURCE: {source_id}\n\nPAGE TEXT (truncated):\n"
            f"{page_text[:max_chars]}\n\nExtract the structured facts as JSON.")
    return _SYSTEM, user


@dataclass
class ExtractResult:
    drafts: list[dict] = field(default_factory=list)
    invalid: list[tuple[dict, str]] = field(default_factory=list)
    notes: str = ""

    @property
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.drafts:
            out[d["type"]] = out.get(d["type"], 0) + 1
        return out


def _envelope(rid: str, rtype: str, labels: list[str], content: dict, structure: dict,
              source_id: str, acquired_on: str) -> dict:
    today = datetime.now(UTC).date().isoformat()
    return {
        "id": rid, "type": rtype, "schema_version": "2",
        "labels": labels or [rid.split(":", 1)[1].replace("-", " ").title()],
        "content": content, "structure": structure,
        "provenance": {"source_ids": [source_id], "transformations": ["extracted"],
                       "acquired_on": acquired_on, "synthetic": False},
        "quality": {"confidence": 0.55, "validation_status": "unverified",
                    "corroboration_count": 0},
        "version": {"revision": 1, "created_on": today, "updated_on": today, "status": "active"},
    }


def _dish(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    st = {"category": d.get("category", "other"),
          "cuisine_state": d.get("cuisine_state", "pan_south"),
          "diet": {"vegetarian": True, "vegan": False, "contains_dairy": False,
                   "contains_egg": False, "contains_seafood": False,
                   "contains_poultry": False, "contains_red_meat": False,
                   **(d.get("diet") or {})},
          "names": d.get("names", []),
          "primary_ingredients": [f"ing:{slugify(x)}" for x in d.get("primary_ingredients", [])]}
    if d.get("style"):
        st["style"] = d["style"]
    return _envelope(f"dish:{slugify(name)}", "dish", [name] + d.get("aliases", []),
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _ingredient(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    st = {"category": d.get("category", "other"),
          "default_unit": d.get("default_unit", "g"),
          "names": d.get("names", [])}
    return _envelope(f"ing:{slugify(name)}", "ingredient", [name],
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _technique(d: dict, source_id: str, acquired_on: str) -> dict:
    name = d.get("name_en", "")
    st = {"family": d.get("family", "assembly"), "names": d.get("names", [])}
    return _envelope(f"tech:{slugify(name)}", "technique", [name],
                     {"summary": d.get("summary", name)}, st, source_id, acquired_on)


def _term(d: dict, source_id: str, acquired_on: str) -> dict:
    translit = d.get("transliteration", "")
    st = {"lang": d.get("lang", "ta"), "native": d.get("native", translit),
          "transliteration": translit, "category": d.get("category", "general")}
    return _envelope(f"term:{slugify(translit or d.get('term_en', ''))}", "term",
                     [d.get("term_en") or translit], {"gloss": d.get("gloss", translit)},
                     st, source_id, acquired_on)


_BUILDERS = {"dishes": _dish, "ingredients": _ingredient, "techniques": _technique,
             "terms": _term}


def parse_extraction(obj: dict, source_id: str, *, acquired_on: str | None = None) -> ExtractResult:
    acquired_on = acquired_on or datetime.now(UTC).date().isoformat()
    res = ExtractResult(notes=str(obj.get("notes", "")))
    seen: set[str] = set()
    for key, build in _BUILDERS.items():
        for raw in obj.get(key, []) or []:
            try:
                rec = build(raw, source_id, acquired_on)
            except Exception as exc:  # noqa: BLE001
                res.invalid.append((raw, f"build: {exc}"))
                continue
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            try:
                parse_record(rec)              # schema-v2 validity gate
                res.drafts.append(rec)
            except Exception as exc:  # noqa: BLE001
                res.invalid.append((rec, f"schema: {type(exc).__name__}: {exc}"))
    return res


def extract_records(page_text: str, source_id: str, provider: Provider, *,
                    acquired_on: str | None = None, temperature: float = 0.0) -> ExtractResult:
    system, user = build_extraction_prompt(page_text, source_id)
    raw = provider.complete(system, user, temperature=temperature, json_out=True)
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return ExtractResult(notes=f"unparseable model output: {raw[:200]!r}")
    return parse_extraction(obj, source_id, acquired_on=acquired_on)
