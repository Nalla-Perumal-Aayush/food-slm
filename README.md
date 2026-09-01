# Food-SLM — Tamil Nadu Vegetarian Culinary SLM

A **true from-scratch** (random initialization, no pretrained weights) decoder-only
Transformer Small Language Model specialized in **Tamil Nadu vegetarian cuisine**,
interacting in English.

> Full frozen specification: [`Documents/master_baseline.md`](Documents/master_baseline.md) (Revision 2).
> Phase roadmap: [`Documents/Steps.txt`](Documents/Steps.txt).

## Status

| | |
|---|---|
| Phase | 4 — Data Source Identification (reorganized data block, see baseline §13) |
| Model target | ~89M params, **Config A** (d=640, L=16, 10 MHA heads, SwiGLU, RoPE, RMSNorm+QK-norm) |
| Scale ladder | **15M PoC** → 40M ablations → 89M release |
| Context | 4K target / 2K fallback, trained 1K→2K→4K |
| Precision | bf16 |
| Hardware | RTX 2000 Ada 16 GB (70 W cap) · i9-14900K · 64 GB RAM |

## Layout

```
configs/            model + tokenizer + training YAML configs
data/
  raw/              (gitignored) authentic source dumps, provenance-tagged
  canonical/        VERSIONED canonical JSONL knowledge store (grounding source)
  corpus/           (gitignored) packed training shards
  eval/             VERSIONED held-out sets + forbidden-entity list
src/food_slm/
  schema/           pydantic models + validators for the canonical store
  data/             ingestion, cleaning, normalization, dedup, synthetic pipeline
  tokenizer/        tokenizer training + fertility/fragmentation analysis
  model/            Config A transformer (plain PyTorch)
  train/            training loop, sequence packing, checkpoint/resume, curriculum
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
