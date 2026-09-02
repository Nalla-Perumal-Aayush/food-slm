# Food-SLM — South Indian Cuisine SLM

A **true from-scratch** (random initialization, no pretrained weights) decoder-only
Transformer Small Language Model specialized in **South Indian cuisine** (Tamil Nadu,
Kerala, Karnataka, Andhra Pradesh, Telangana, Puducherry; vegetarian & non-vegetarian),
interacting in English. Food-domain specialist — not a general assistant.

> Full frozen specification: [`Documents/master_baseline.md`](Documents/master_baseline.md) (Revision 6 — the single final reference).
> Canonical-store schema: [`Documents/phase5_canonical_schema.md`](Documents/phase5_canonical_schema.md) (v2 — see §0.1).
> Dravidian-script policy: [`Documents/tamil_script_policy.md`](Documents/tamil_script_policy.md) (Tamil today; generalizes to 4 scripts — baseline §6).
> Phase roadmap: [`Documents/Steps.txt`](Documents/Steps.txt).

## Status

| | |
|---|---|
| Phase | 4 — Data Source ID + MCP contract. Fork A: model + tokenizer + sequence packing built & tested; next = training loop → checkpoint/resume → eval harness |
| Domain | South Indian cuisine, 6 regions, veg + non-veg; religion/community out of scope |
| Canonical store | **schema v2** (`src/food_slm/schema/`, `configs/canonical_enums.yaml`); seed store migrated — `python scripts/validate_canonical.py` |
| Native scripts | 4-script Dravidian policy "recognized, never required" — English I/O, transliteration bridge, ≤0.5% of tokens |
| Forbidden entities | mechanism live (SHA-locked); list to be rebuilt at South India scope (~80–150 entries) |
| Model size | **LOCKED ~150M (Config B: d=768, L=20, 12 heads, SwiGLU 1984) → 151.2M, ±5% tolerance.** Ladder: 15M PoC → 40M → 89M de-risk → 150M release. |
| Data budget | ~15–45B training tokens; general-English base from permissive/open corpora (downloaded, not scraped) |
| Context / precision | 4K (hard-locked, 1K→2K→4K curriculum) · bf16 (hard-locked) |
| Vocabulary | byte-level BPE; size 16k vs 24k under benchmark |
| Timeline | 6–12 months (150M run ≈ 1.5–3 weeks) |
| Hardware | RTX 2000 Ada 16 GB (70 W cap) · i9-14900K · 64 GB RAM |

## Layout

```
configs/            model + tokenizer + training YAML configs
data/
  raw/              (gitignored) authentic source dumps, provenance-tagged
  canonical/        VERSIONED canonical JSONL knowledge store (grounding source)
  corpus/           (gitignored) assembled text + packed training shards
  instruction/      (gitignored) chat-format SFT data
  eval/             VERSIONED held-out sets + frozen forbidden-entity list
src/food_slm/
  schema/           pydantic models + validators for the canonical store
  data/             ingestion, cleaning, normalization, dedup, synthetic pipeline
  tokenizer/        byte-level BPE trainer + analysis report — built
  model/            config-driven transformer (FoodSLM) — built + validated at 15M/40M/89M
  train/            sequence packing (done) + training loop, checkpoint/resume, curriculum
  eval/             perplexity, generation, AI-judge harness
scripts/            thin CLI entry points
artifacts/          (gitignored) checkpoints, trained tokenizers, reports
```

## Environment

- Python 3.11 (conda env `slm_building`)
- PyTorch 2.13 + CUDA 12.6 (installed from the PyTorch CUDA index, not `requirements.txt`)
- All caches/data/checkpoints live on **D:** — never C: (`PIP_CACHE_DIR`, `HF_HOME`,
  `HF_DATASETS_CACHE`, `TORCH_HOME`, `WANDB_DIR` are set to `D:\Food SLM\.cache\...`)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126   # already done
pip install -r requirements.txt
pip install -e .
pytest
```

## License

Apache-2.0 (code). Dataset and model artifacts carry their own provenance and
licensing metadata per the baseline data principles (§4).
