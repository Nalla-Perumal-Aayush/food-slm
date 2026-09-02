# Changelog

## [0.0.9] — 2026-09-02
### Added
- **Extraction / synthesis pipeline design** — `Documents/mcp_extraction_pipeline.md`:
  domain-agnostic **extract · synthesize · validate** engine; **self-hosted scraping only**
  (httpx+trafilatura; robots.txt + rate-limit + cache); license classifier gates every source;
  agents write only to `data/interim/canon_staging/`, `promote_staged(reviewer)` is the sole
  path into `data/canonical/`; nothing scraped enters `data/corpus/`.
- **Evaluation strategy** — `Documents/evaluation_strategy.md`: **AI-primary, human-anchored**.
  Human builds ~1,500 golden items once; **2.5% random calibration sample** per batch/round
  (halt if AI-vs-human agreement < 90%). Anti-circularity: Claude generates, so the deciding
  votes are non-Claude lineages.
- **ALL-FREE agent stack (no paid API keys — no funding until ship):** Claude Code itself is the
  orchestrator / fetch / extraction / synthesis agent (WebSearch + WebFetch + sub-agents); the
  validation / judge **panels are local Ollama models of 3 independent lineages — Qwen (Alibaba)
  + Llama (Meta) + Gemma (Google)** + deepseek-r1:32b cross-check + cross-source corroboration.
  Verified on the machine: Ollama v0.33.2 JSON inference, `httpx`+`trafilatura` scrape
  (Wikipedia → clean text), `fastembed` bge-small embeddings (dim 384).
- `configs/coverage_targets.yaml` — domain target matrix (~20k records).
- `configs/agents.yaml` — all-free roster; judge panel excludes Claude-lineage models.
- `src/food_slm/data/llm.py` — provider-agnostic client (Anthropic / OpenAI / Google / local
  Ollama, lazy SDK import), `FakeProvider`, `Panel` + consensus. `tests/test_llm.py` (6 tests).
### Changed
- `requirements.txt` — `httpx` + `trafilatura` + `fastembed` pinned. **`sentence-transformers`
  explicitly rejected** (force-upgrades `transformers` to 5.x and breaks the pinned stack;
  `fastembed` is isolated ONNX). `transformers` kept at 4.48.2.
- `.env.example` — `FASTEMBED_CACHE_DIR`, `OLLAMA_HOST`; optional free-tier keys noted.

## [0.0.8] — 2026-09-02
### Added
- **Training loop** (`src/food_slm/train/loop.py`) — causal LM, bf16 autocast on CUDA, AdamW
  (β 0.9/0.95, wd 0.1), linear warmup + cosine decay to `min_lr_ratio` (pure function of step),
  gradient accumulation, global grad-norm clip, z-loss (in the model). Periodic val eval +
  checkpoints. `TrainConfig` from YAML; `deterministic=True` / `device=` overrides;
  `stop_after` for resume testing.
- **Bit-exact checkpoint/resume** (`train/checkpoint.py`) — saves model + optimizer + step +
  python/numpy/torch RNG state. `BatchSampler` — deterministic, position-independent micro-batch
  index stream (batch *b* is a pure function of `(seed, b)`), so resume just sets `start_batch`.
  `prune_checkpoints` keeps the last N.
- **Eval** (`src/food_slm/eval/`) — `perplexity.py` (`evaluate_loss`, `perplexity_report`),
  `generate.py` (temperature / top-k / top-p sampling, `generate_text`).
- `scripts/train.py`, `scripts/run_poc_checks.py`; `tests/test_train.py` (7 tests — LR schedule,
  sampler determinism/resumability, checkpoint round-trip, exact checkpoint at stop, bit-exact
  resume in deterministic mode, no-divergence resume, short run learns).
### Verified
- 15M PoC trains end-to-end on the RTX 2000 Ada: loss 9.77 → 7.0 in 60 steps, ~38k tok/s;
  interrupt-at-20 → resume-to-40 continues the trajectory cleanly.
- `ModelConfig` now validates `hidden_dim % num_heads` and `num_heads % num_kv_heads`.

## [0.0.7] — 2026-09-02
### Added
- **Sequence packing** (`src/food_slm/train/packing.py`) — tokenize each document, append
  `<|eos|>`, concatenate, cut into fixed-length `seq_len` windows (trailing remainder dropped).
  Output: flat `uint16` `tokens.bin` (+ optional `val.bin`) + `manifest.json` with counts,
  dtype, `eos`/`bos` ids, tokenizer-meta sha256 and corpus sha256 for reproducibility.
  Optional per-window BOS. Deterministic — identical inputs produce byte-identical shards.
- **`PackedDataset`** — memmap view of a shard; `ds[i] -> (input_ids, labels)` int64,
  `labels == input_ids` (the model shifts internally).
- `scripts/pack_corpus.py`; `tests/test_packing.py` (6 tests — window shape/dtype, token-budget
  conservation, EOS presence, per-window BOS, val split + dataset, determinism).
- Packed the PoC corpus: 9.37M raw tokens → **8,969 train + 183 val windows** of 1024
  (`data/corpus/poc/`, gitignored).

## [0.0.6] — 2026-09-02
### Added
- **Tokenizer trainer** (`src/food_slm/tokenizer/train.py`) — byte-level BPE from
  `configs/tokenizer.yaml` via HF `tokenizers`. No UNK; special + structural + chat tokens
  atomic with fixed low ids (pad=0, bos=1, eos=2, unk=3). NFKC normalizer. Saves
  `tokenizer.json` + `meta.json` (actual vocab, special-token ids, corpus sha256). BOS/EOS
  are added by the packer, not here.
- **Tokenizer analyzer** (`tokenizer/analyze.py`) — token fertility, compression ratio, vocab
  utilization, culinary-term & per-script native-name fragmentation, number/unit efficiency,
  native-script vocab share, sequence-length efficiency.
- **`data/serialize.py`** (PoC-minimal) — canonical store → training text (dishes, recipes with
  `<INGREDIENT>`/`<STEP>` structure, techniques, troubleshooting, glossary); native names only
  as `English (translit, «native»)`.
- **`scripts/fetch_sample.py`** — builds `data/interim/sample/corpus.txt` from the serialized
  canonical store + a permissive English slice (wikitext-103 / wikipedia / fineweb-edu), with a
  canonical-only fallback if offline. **This is the ~300 MB PoC dev sample, not the training corpus.**
- `scripts/train_tokenizer.py`; `tests/test_tokenizer.py` (10 tests — byte round-trip incl.
  4 Dravidian scripts, atomic special tokens, no-UNK, analyzer shape, serialization).
### Changed
- `configs/tokenizer.yaml` — `tamil_script` → `dravidian_script` (4-script, `max_vocab_share`
  0.05 → 0.08); vocab_size annotated as default/PoC (16k vs 24k benchmark); `analysis_report`
  keys renamed for 4 scripts.

## [0.0.5] — 2026-09-02
### Changed
- **Final model size LOCKED at ~150M** (`master_baseline.md` Revision 6) — no longer data-driven.
  **Config B** = d=768, L=20, 12 MHA heads, SwiGLU `ffn_inner_dim` 1984, vocab 16384 →
  **151,225,600 params (+0.8% of 150M)**; accepted tolerance ±5% for implementation fixes.
  `configs/model_150m.yaml` added; validated at the exact count by `scripts/validate_model.py`
  (forward/backward, grad-checkpoint equivalence, CUDA bf16). 89M (Config A) demoted to an
  optional de-risk rung; the 350M rung is removed.
- **Data programme sized to 150M** — target ~15–45B training tokens. General-English base grows
  to ~12–40B tokens **from permissive / openly-licensed corpora (downloaded, not self-crawled)**;
  culinary stays domain-bounded (~0.5–1.5B unique) under the existing integrity gates.
  Indiscriminate web scraping is **rejected** (baseline §8.0) — it would forfeit the clean
  open-source release and eval integrity, and is unnecessary given available open corpora.
- Baseline §5.1 (Config B), §5.1a (Config A de-risk), §5.2 (ladder), §8.0 (data-volume
  strategy), §8.3 (150M volume targets) reworked. `show_model_stats` / `validate_model` /
  `test_smoke` / `test_model` include the 150M config.

## [0.0.4] — 2026-09-02
### Added
- **`src/food_slm/model/transformer.py` — the `FoodSLM` model** (first code milestone, baseline §12).
  Config-driven decoder-only Transformer: RMSNorm + QK-norm, RoPE (NeoX-style), SwiGLU FFN,
  MHA/GQA via `F.scaled_dot_product_attention` (causal, no flash-attn), tied embeddings,
  scaled-normal init (residual projections /√(2L)), z-loss, optional gradient checkpointing.
  `forward(input_ids, labels=None) -> ModelOutput(logits, loss, ce_loss, z_loss)`.
- `scripts/validate_model.py` + `tests/test_model.py` (9 tests) — the **model half of the 15M PoC**:
  analytic vs actual param count (exact at 15M/40M/89M: 15,407,424 / 40,513,280 / 89,807,488),
  forward shape + finiteness, finite grads for every parameter, gradient-checkpointing loss
  equivalence, and single-batch overfit (9.72 → 0.06) proving the learning path. Runs on
  CUDA bf16 — verified PASS on the RTX 2000 Ada.
### Changed
- `ModelConfig.param_count()` now includes the QK-norm gains (`+2·head_dim` per layer).
- `model/__init__.py` exports `FoodSLM`, `ModelOutput`, `build_model`.

## [0.0.3] — 2026-09-02
### Changed
- **Domain expanded: Tamil Nadu vegetarian → South Indian cuisine** (`Documents/master_baseline.md`
  Revision 5, edited in place — the single final reference). 6 regions (TN, Kerala, Karnataka,
  Andhra Pradesh, Telangana, Puducherry); vegetarian + non-vegetarian at full depth; religion &
  community out of scope (neutral practical constraints; region/style tags only); model stays a
  food-domain specialist. **Model size is now data-driven** — ladder 15M→40M→89M→150M→350M with
  the rungs above 89M gated on the 89M evaluation. Timeline 12–16 weeks → 6–12 months.
- Tamil-script policy generalized to a **4-script Dravidian policy** (Tamil, Malayalam, Kannada,
  Telugu). Tokenizer **vocabulary size re-opened for benchmark** (16k vs 24k).
- **Schema v2 implemented** (`src/food_slm/schema/models.py` + `store.py`, `configs/canonical_enums.yaml`,
  spec `Documents/phase5_canonical_schema.md` §0.1):
  - `names: [LocalName]` (lang/native/transliteration, transliteration required) replaces
    `name_ta`/`transliteration`/`ipa`; `term` gets `lang`/`native`/`transliteration`.
  - `cuisine_state` (required on dish/recipe), `region.level` (state|sub_region|style) with a
    parent-chain rule, `diet: DietProfile` on dish/recipe with consistency validation,
    non-veg `ingredient.category` values + allergen classes, step `internal_temp`/`doneness_cue`.
  - `constraint.kind` drops `ritual_religious`/`seasonal_ayurvedic`, adds `cooking_style`.
  - Community/religion terms rejected in labels + structure (per-record) and content prose
    (`store.check_community_tags`, Phase 11 gate); core blocklist in `DISALLOWED_COMMUNITY_TERMS`.
  - `store.check_cuisine_states()` — dish/recipe `cuisine_state` must have a state-level region.
  - `scripts/migrations/v1_to_v2.py` (idempotent); run on the seed store (44 records, +`region:tamil-nadu`).
  - `tests/test_schema.py` rewritten for v2 (23 schema tests; 32 total pass).
- Baseline §4.1 (three data representations) and §4.2 (response format by task type) added.
  `data/instruction/` added (gitignored). `data/README.md` updated.

## [0.0.2] — 2026-09-02
### Added
- **Tamil-script handling policy** (`Documents/tamil_script_policy.md`) — "recognized, never
  required": English-only I/O, native script only in a capped inline form (≤0.5% of pretraining
  tokens, hard gate), popular romanized transliteration as the join key, NFC at ingestion,
  Tamil-script vocab ≤5%, Tamil-heavy sources routed to extraction-only. Knobs recorded in
  `configs/tokenizer.yaml`. `_TamilNamed` schema mixin enforces "native name ⇒ transliteration".
- **Forbidden-entity list frozen** before Phase 4 collection:
  - `data/eval/forbidden_entities.jsonl` (18 seed entries: 11 `withheld_real` for abstention
    probes, 7 `fictitious` for fabrication probes) + `.lock.json` (SHA-256 pin + amendment log).
  - `src/food_slm/data/forbidden.py` — loader, `verify_lock` gate, canonical-store leak check,
    phrase scanner for corpus text.
  - `scripts/freeze_forbidden.py`; `scripts/validate_canonical.py` now runs the leak check;
    `tests/test_forbidden.py` (7 tests). `data/eval/README.md`.
- **Phase 5 canonical knowledge-store schema v1.** Spec: `Documents/phase5_canonical_schema.md`.
  - `src/food_slm/schema/models.py` — 13 record types (source, ingredient, dish, recipe,
    technique, equipment, region, state, outcome, constraint, problem, term, rel) on a common
    envelope (identity + content + structure + provenance + quality + version), `Measurement`
    with canonicalization rules, typed directional relationships, recipe process-DAG validation.
  - `src/food_slm/schema/store.py` — JSONL loader + cross-record referential-integrity checks
    (§8 contract); the schema half of the Phase 11 hard gate.
  - `configs/canonical_enums.yaml`, `configs/measures.yaml` — controlled vocabularies + unit tables.
  - `data/canonical/*.jsonl` — seed store (43 records) built around a worked *ven pongal* example;
    validates end to end.
  - `scripts/validate_canonical.py`; `tests/test_schema.py` (16 tests).
### Changed
- `schema/__init__.py` exports the full v1 surface (was the v0 stub).
- `Documents/master_baseline.md` → Revision 4 (Phase 5 schema v1; Tamil-script policy;
  forbidden-entity list frozen; 4K / bf16 / general-English base corpus hard-locked out of
  "proposed").

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
