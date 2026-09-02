# MCP Extraction & Synthesis Pipeline

**Status:** design frozen 2026-09-02. Phase 4 deliverable. Tracks `master_baseline.md` §4 / §8.0 / §8.2 / Phase 6–7b.
**Owner:** Aayush.

A **domain-agnostic** engine with three verbs — **extract · synthesize · validate** — that
builds and grows the canonical knowledge store. The domain is defined entirely by
`configs/coverage_targets.yaml` + the schema (`Documents/phase5_canonical_schema.md`); swap the
config and the same MCP tools build any structured-knowledge dataset. This is deliberate: the
same system that seeds the South-Indian store is what scales it later and what expands future
datasets.

---

## 0. The one rule

**Agents produce structured facts in the canonical store. They never write to `data/corpus/`.**

A recipe's ingredient list and step sequence are largely uncopyrightable facts/procedures; the
surrounding prose is copyrighted. The pipeline extracts `dish` / `recipe` / `ingredient` / …
records, tags provenance, routes through validation, and later **re-expresses** them in our own
words for training (Phase 7b). Scraped prose is never a training token. This is what keeps the
open-source release license-clean and the evaluation uncontaminated.

---

## 1. Architecture — 3 layers

```
LAYER 1  MCP server "food-slm-canon"  — the tools Claude calls
  list_gaps · propose_sources · register_source(+license classifier)
  fetch · extract_records · resolve_entities · validate_records
  stage_records · promote_staged(reviewer) · coverage_report · synthesize

LAYER 2  Orchestration  — Claude (lead) + sub-agents
  gap → propose_sources → [HUMAN approves source list] → register_source
  per source: sub-agent  fetch → extract_records → resolve_entities
                         → validate_records → stage_records
  aggregate → dedup across sub-agents → review batch
  [HUMAN calibration: 2.5% random sample] → promote_staged → coverage_report → repeat

LAYER 3  Scraping backend  — self-hosted only
  httpx + trafilatura            (static pages)
  self-hosted headless Chromium  (JS pages)
  robots.txt enforced · per-domain rate limit · local response cache (fetch once)
```

Only path into `data/canonical/` is `promote_staged` with a named `reviewer`.

---

## 2. MCP tool contract

| Tool | Input | Output | Guardrail |
|---|---|---|---|
| `list_gaps()` | — | ranked gaps vs `configs/coverage_targets.yaml` | read-only |
| `propose_sources(gap, k)` | gap id, count | candidate URLs/domains + rationale | **does not fetch** |
| `register_source(url)` | url | `source` record id; `license_class` + `usable_for` set by the license classifier (§4) | ARR ⇒ `reference_only` unless a human overrides; unknown-rights ⇒ blocked pending sign-off |
| `fetch(url)` | url | cleaned page text + fetch metadata | robots.txt; per-domain QPS cap; cached by URL-hash; **one URL, never a crawl** |
| `extract_records(source_id, text)` | source id, text | draft schema-v2 records, `validation_status: unverified` | drafts only; no commit |
| `resolve_entities(drafts)` | drafts | per-draft: `matched <id>` / `new` / `ambiguous` | embedding + name/transliteration + ingredient-set similarity |
| `validate_records(drafts)` | drafts | per-record verdicts from the panel (§3) + issues | schema validator + forbidden-entity scan + measurement sanity + corroboration |
| `stage_records(records)` | validated drafts | written to `data/interim/canon_staging/` at `auto_validated` | **not** `data/canonical/` |
| `promote_staged(ids, reviewer)` | staged ids, reviewer name | moved into `data/canonical/`, `human_validated` or `auto_validated` per §3 | **only path into the store**; `reviewer` required |
| `coverage_report()` | — | store stats vs targets | read-only |
| `synthesize(record_ids, mode)` | canonical ids, mode | serialized/paraphrased training text | grounded only; every fact traces to a seed record; ratio-capped (baseline §8.2) |

---

## 3. Validation panel — AI-primary, human-anchored

See `Documents/evaluation_strategy.md` for the full design. In brief:

- **`validate_records`** fans out to a panel of **different model families** (Claude + GPT-4o +
  a local Qwen2.5-7B) plus **cross-source corroboration**.
- Unanimous panel **and** `corroboration_count ≥ 2` ⇒ `stage` as `auto_validated`, eligible for
  `promote_staged` without individual human review.
- Split verdict, or a `substitutes_for` / `fixes_problem` / `produces_outcome` relationship, or
  `confidence < 0.8` ⇒ routed to the human review queue.
- **2.5 % random sample** of every promoted batch is human-checked for calibration. If
  AI-vs-human agreement on that sample drops below **90 %**, the pipeline halts and the panel /
  prompts / golden sets are re-anchored before resuming.
- Records backing the eval set are **always** human-validated (small, bounded set).

---

## 4. License classifier (`src/food_slm/data/licensing.py`)

For each source: inspect the page + site for CC license markers, `LICENSE`/`terms` pages,
government / public-domain signals, `<link rel="license">`, JSON-LD `license` fields. Map to:

| `license_class` | `usable_for` |
|---|---|
| `public_domain`, `permissive` | `[training, grounding]` |
| `share_alike` | `[training, grounding]` (attribution obligations tracked) |
| `noncommercial` | `[grounding, reference_only]` |
| `unknown`, `all_rights_reserved` | `[reference_only]` — **facts only, re-expressed**; prose never trained on |

`reference_only` sources still contribute facts to the canonical store; they never contribute
tokens to `data/corpus/`.

---

## 5. Agent roster — ALL-FREE STACK (no paid API keys)

No external funding until the project ships, so: **Claude Code itself** (the CLI already in use)
is the orchestrator / fetch / extraction / synthesis agent — WebSearch + WebFetch + sub-agents,
no API key — and the validation / judge **panels are local Ollama models of different
pretraining lineages**, fully offline, no rate limits, no cost. Full roster: `configs/agents.yaml`.

| Role | Agent | Cost |
|---|---|---|
| Orchestrator · source discovery · one-off fetch · extraction · synthesis | **Claude Code** (this session) | already have it |
| Batch fetch (static pages) | `httpx` + `trafilatura` | free, local |
| Batch fetch (JS pages) | self-hosted Playwright/Chromium (optional) | free, local |
| Entity resolution / dedup | `bge-small-en-v1.5` embeddings | free, local |
| Language ID | fastText `lid.176` | free, local (in stack) |
| **Data-validation panel** | **Qwen2.5-7B (Alibaba) + Llama-3.1-8B (Meta) + Gemma-2-9B (Google)** via Ollama | free, local |
| Validation heavyweight cross-check | **deepseek-r1:32b** (partial CPU offload, ~6 tok/s) | free, local |
| Cross-source corroboration | N independent sources (not a model) | — |
| Optional extra votes | Google AI Studio / Gemini free tier · Groq · GitHub Models (free signup, no billing) | free |

**Anti-circularity:** Claude (via Claude Code) generates; the deciding validation votes are
**Qwen + Llama + Gemma** — three independent lineages, none related to Claude — plus deepseek-r1
and cross-source corroboration. This is a *stronger* separation than a GPT-4o + Gemini panel.

**Quality note:** local 7–9B models are weaker at open-ended *extraction* but competent at
*verification* ("is this record consistent with this source text?"). Extraction quality comes
from Claude Code doing it directly; the local panels only check. The 2.5% human calibration
catches residual drift.

**Security:** ≥1 fully-local model in every panel (here: all of them). Only public URLs + public
source text ever leave the machine — never canonical records, eval sets, or weights. Local
models run during Phase 6–7 while the training GPU is idle.

---

## 6. Synthesis mode (Phase 7b — the same engine backwards)

`synthesize(record_ids, mode)` — modes: `serialize` (record → prose), `paraphrase` (multi-phrasing),
`qa` (instruction pair), `persona` (difficulty / audience variation). Every generated fact is
checked against its seed canonical record by `validate_records` (fact-level); mismatches
rejected. Provenance: `synthetic=true`, `generator`, `seed_record_ids`. Ratio-capped per
capability (≤ 1–2× authentic). Never enters the eval set.

---

## 7. Build order

1. ✅ `configs/coverage_targets.yaml` — the domain target matrix.
2. ✅ `src/food_slm/data/llm.py` — provider-agnostic client + `Panel` (+ `FakeProvider`); Ollama path verified.
3. `src/food_slm/data/licensing.py` — the license classifier.
4. `src/food_slm/data/ingest.py` — self-hosted scrape (`httpx` + `trafilatura`; robots.txt; rate limit; URL-hash cache).
5. `src/food_slm/data/extract.py` — page text → draft records (Claude Code directly, or `extraction_local`).
6. `src/food_slm/data/dedup.py` — `fastembed` bge-small (dim 384) semantic dedup + `datasketch` MinHash.
7. `src/food_slm/data/canon_staging.py` — `stage_records` / `promote_staged` (the store gate).
8. `scripts/canon_run.py` — the orchestration driver (Claude Code + sub-agents).
9. `src/food_slm/data/mcp_server.py` — OPTIONAL: expose the tools to a separate session via MCP.
