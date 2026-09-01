# data/

| Dir | Versioned? | Contents |
|---|---|---|
| `raw/` | no (gitignored) | Authentic source dumps exactly as acquired, one folder per source, each with a `source.json` provenance record. |
| `interim/` | no (gitignored) | Intermediate cleaned/normalized records between pipeline stages. |
| `canonical/` | **yes** | The canonical knowledge store — linked JSONL record types (`dishes.jsonl`, `ingredients.jsonl`, `recipes.jsonl`, `sources.jsonl`, `relationships.jsonl`, `processes.jsonl`, `terminology.jsonl`, `troubleshooting.jsonl`, ...). Grounding source for all synthesis. |
| `corpus/` | no (gitignored) | Packed, tokenized training shards (general-EN base + culinary authentic + synthetic), plus the entity-disjoint split manifests. |
| `eval/` | **yes** | Held-out evaluation sets, the frozen **forbidden-entity list**, hallucination + generalization-gap probes. Never contains synthetic data. |

Rules (baseline §4):
- Every record carries provenance (source, license, acquisition date, transformations) and quality (confidence, validation status).
- Splitting is entity-disjoint, not row-random.
- An n-gram `train × eval` contamination scan is a hard gate before any training run.
