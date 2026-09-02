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

## 5. Agent roster

| Role | Model | Access |
|---|---|---|
| Lead orchestrator | Claude Sonnet (Opus for hard cases) | Anthropic API / Claude Code |
| Extraction | Claude Sonnet | Anthropic API |
| Synthesis | Claude Sonnet | Anthropic API |
| Entity resolution / dedup | `bge-small-en-v1.5` embeddings | local, no network |
| Language ID | fastText `lid.176` | local (in stack) |
| Validation vote 1 | Claude | Anthropic API |
| Validation vote 2 | GPT-4o / o4-mini | OpenAI API |
| Validation vote 3 | Qwen2.5-7B-Instruct (4-bit) | local (Ollama / llama.cpp) |
| Corroboration | N independent sources (not a model) | — |

**Security:** first-party APIs only; zero-data-retention where offered; **≥1 fully-local model in
every panel**. Only public URLs + public source text leave the machine — never canonical
records, eval sets, or weights. Local 7B models run during Phase 6–7 while the training GPU is idle.

---

## 6. Synthesis mode (Phase 7b — the same engine backwards)

`synthesize(record_ids, mode)` — modes: `serialize` (record → prose), `paraphrase` (multi-phrasing),
`qa` (instruction pair), `persona` (difficulty / audience variation). Every generated fact is
checked against its seed canonical record by `validate_records` (fact-level); mismatches
rejected. Provenance: `synthetic=true`, `generator`, `seed_record_ids`. Ratio-capped per
capability (≤ 1–2× authentic). Never enters the eval set.

---

## 7. Build order

1. `configs/coverage_targets.yaml` — the domain target matrix.
2. `src/food_slm/data/llm.py` — provider-agnostic multi-model client (+ `FakeProvider` for tests).
3. `src/food_slm/data/licensing.py` — the license classifier.
4. `src/food_slm/data/{ingest,extract,dedup}.py` — scraping (self-hosted), extraction, dedup.
5. `src/food_slm/data/mcp_server.py` — the 11 tools.
6. `scripts/canon_run.py` — the orchestration driver.
