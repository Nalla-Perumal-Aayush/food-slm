# data/

The three data representations (baseline §4.1): **canonical store** (typed JSONL, source of truth) →
**corpus** (text → packed token shards, what the model trains on) + **instruction** (chat JSONL).

| Dir | Versioned? | Contents |
|---|---|---|
| `raw/` | no (gitignored) | Authentic source dumps exactly as acquired, one folder per source, each with a `source.json` provenance record. |
| `interim/` | no (gitignored) | Intermediate cleaned/normalized records between pipeline stages. |
| `canonical/` | **yes** | The canonical knowledge store — one JSONL file per record type (`sources.jsonl`, `ingredients.jsonl`, `dishes.jsonl`, `recipes.jsonl`, `techniques.jsonl`, `equipment.jsonl`, `regions.jsonl`, `states.jsonl`, `outcomes.jsonl`, `constraints.jsonl`, `problems.jsonl`, `terminology.jsonl`, `relationships.jsonl`). Common envelope; per-type `content`/`structure`. Grounding source for all synthesis. Schema: `Documents/phase5_canonical_schema.md` (v1; v2 deltas in baseline §3.1). |
| `corpus/` | no (gitignored) | Assembled training text → packed tokenized shards (`uint16`/`uint32` memmap) + JSON split/manifest. General-EN base + culinary authentic + synthetic. |
| `instruction/` | no (gitignored) | Instruction / SFT data — `{"messages": [...], "provenance": {...}}` chat JSONL, ~60% agent-built from the canonical store. |
| `eval/` | **yes** | Held-out evaluation sets, the frozen **forbidden-entity list** (`forbidden_entities.jsonl` + `.lock.json` + `.amendments.log`), per-region / veg-non-veg / community-leakage / hallucination / generalization-gap probes. Never contains synthetic data. |

Rules (baseline §4):
- Every record carries provenance (source, license, acquisition date, transformations) and quality (confidence, validation status).
- Splitting is entity-disjoint, not row-random.
- `train × eval` n-gram contamination scan **and** forbidden-entity scan are hard gates before any training run.
- Native Dravidian script ≤ 0.5% of pretraining tokens (hard gate at corpus assembly).
