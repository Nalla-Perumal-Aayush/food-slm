# Changelog

## [0.0.1] — 2026-09-01
### Added
- Repository scaffold: `src/food_slm/` packages (schema, data, tokenizer, model, train, eval, utils),
  `configs/`, `scripts/`, `tests/`, `data/` tree, `artifacts/` tree.
- Model configs for the scale ladder: `model_15m.yaml` (PoC), `model_40m.yaml`, `model_89m.yaml` (Config A).
- `configs/tokenizer.yaml` (16k byte-level), `configs/train_poc.yaml`.
- `ModelConfig` with parameter-count accounting; smoke tests asserting the ladder sizes.
- `scripts/check_env.py`, `scripts/show_model_stats.py`.
- Canonical knowledge-store schema **v0 stub** (`schema/models.py`) — pending full Phase 5 design.

### Baseline
- Tracks `Documents/master_baseline.md` **Revision 2** (89M Config A, 4K context,
  general-English base corpus, formalized synthetic pipeline, re-scoped reasoning criterion,
  15M PoC as first milestone).

### Environment
- PyTorch 2.13.0 + CUDA 12.6, bf16 verified on RTX 2000 Ada.
- All caches/data/checkpoints pinned to `D:` (`PIP_CACHE_DIR`, `HF_HOME`,
  `HF_DATASETS_CACHE`, `TORCH_HOME`, `WANDB_DIR`).
