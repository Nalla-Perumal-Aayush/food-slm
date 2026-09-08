# Changelog

## [0.0.20] — 2026-09-08
### Added — 40M ablation preparation (baseline Rev 8, `Documents/plan_40m_ablation.md`)
- **Baseline Rev 8** — 15M PoC passed → **40M is the next rung, redefined d=512 / L=14 →
  53.4M** (depth bump: L=8→L=14→L=20). **89M rung now SKIPPED if 40M is clean.** 40M runs on
  the real corpus + retrained tokenizer: general-English = **FineWeb-Edu (ODC-By), ~1B tokens**.
- `configs/model_40m.yaml` → L=10→**L=14**, max_seq_len 2048→4096. `validate_model.py` PASS
  (53,362,944 params, analytic == actual, grad-ckpt loss-identical). `model_89m.yaml` marked
  conditional.
- **`src/food_slm/data/langid.py`** — fastText `lid.176` language ID + Dravidian-script ratio
  gate; `ensure_model()` fetches the 0.9 MB model; scoped NumPy-2 shim for the fastText wheel.
- **`scripts/fetch_corpus.py`** — streams FineWeb-Edu / FineWeb / C4 from HF to a token budget,
  min-length + sampled-langid + exact-dedup filters, writes ~200 MB sharded text with a
  per-shard-SHA provenance manifest. Smoke-tested at 5M tokens.

## [0.0.19] — 2026-09-08
### Added
- **First full 15M PoC training run** (`artifacts/reports/poc_15m_run_2026-09-08.md`) — the
  baseline §12 milestone. 2000 steps on the repacked PoC corpus (wikitext-103 + 128 canonical
  records ×8 = 9.22M tokens), ~57 min on the 70 W GPU at ~38.2k tok/s.
  **val_loss 5.42 → 4.13**, monotone at every eval; grad norm steady 0.4–0.5; z-loss
  ~0.01–0.015; clean exit. **PASS on all three acceptance checks.**
  Samples: fluent wikitext-register English, no repetition collapse; the `<DISH>…</DISH> is a
  <category> from <state>` template + `</QTY>`/`</STEP>` structure reproduced from the
  serialized store; Dravidian script round-trips. Domain knowledge ≈ 0 (28k domain tokens vs
  9.2M) — expected at this scale, not a toolchain fault.
- `scripts/build_poc_corpus.py` — folds newly-promoted canonical records into the PoC sample
  in place (English substrate untouched), repeats the block (default ×8) for structural-token
  signal. Used to repack `data/corpus/poc/` from 44 → 128 canonical records.

## [0.0.18] — 2026-09-08
### Added
- **Seed-store population, batches 6–8 — non-veg + regional style** (Wikipedia, CC-BY-SA).
  First real records exercising the `DietProfile` non-veg flags:
  - **Chicken Chettinad** — `dish:chicken-chettinad` (contains_poultry + contains_dairy),
    `ing:{chicken,kalpasi,poppy-seeds}`, `style:"chettinad"`, dry-roast + wet-grind edges.
  - **Kerala fish curry (Malabar matthi curry)** — `dish:malabar-matthi-curry`
    (contains_seafood), `ing:{sardine,kudampuli,shallot,coconut-oil}`, `equip:meen-chatti`,
    `kudampuli substitutes_for tamarind` (high-risk predicate → calibration sample).
  - **Chettinad** — `region:chettinad` (level `style`, parent `region:tamil-nadu`) +
    `belongs_to_region` edges. Region/style tag only — no community framing.
  Panel (Qwen+Gemma, deepseek-r1 tiebreak) caught two real issues: `ing:sardine` duplicate-lang
  in `names[]` (fixed), and `ing:tapioca` correctly **rejected by the forbidden-entity gate**
  (`fe:0010`, a held-out abstention probe) — pairing re-pointed to rice.
  **Canonical store: 102 → 128 records**, valid. 106 fast tests pass, ruff clean.
### Fixed
- `data/llm.py` `OllamaProvider` — replaced the hard-coded 120 s HTTP timeout with a
  configurable `OLLAMA_TIMEOUT_S` (default **600 s**). The 32B deepseek-r1 tiebreak model
  spills past 16 GB VRAM onto the CPU on the 70 W-capped card and was crashing the panel
  mid-run with `httpx.ReadTimeout`.

## [0.0.17] — 2026-09-03
### Added
- **Seed-store population, batches 4–5 — Puttu + Pesarattu** (Wikipedia, CC-BY-SA).
  Claude-Code extraction, panel unanimous accept (Qwen+Gemma 14/14), human-reviewed, promoted.
  Regional spread beyond the Tamil Nadu staples: Kerala (`dish:puttu`, `ing:rice-flour-coarse`,
  `equip:puttu-kutti`) and coastal Andhra Pradesh (`dish:pesarattu`, `ing:green-gram-whole`).
  8 new `rel` edges, including the first cross-dish `variant_of` (`pesarattu variant_of dosa`)
  and two `belongs_to_region` edges to state-level regions.
  **Canonical store: 86 → 102 records**, valid.

## [0.0.16] — 2026-09-03
### Added
- **Seed-store population, batch 3 — Dosa** (Wikipedia `Dosa_(food)`, CC-BY-SA). Claude-Code
  extraction, panel unanimous accept (Qwen+Gemma 8/8), human-reviewed, promoted. New:
  `dish:dosa`, `tech:griddle-cook`, `equip:tava`, `term:thosai`, and 4 `rel` edges
  (`dosa has_alias thosai`, `dosa requires_technique griddle-cook`,
  `griddle-cook requires_equipment tava`, `fenugreek used_in dosa`).
  **Canonical store: 77 → 86 records**, valid. Calibration log: 4 promote batches
  (`seed-dosa` drew a 1-record sample).

## [0.0.15] — 2026-09-03
### Added
- **Claude-Code extraction path.** `data/pipeline.py::stage_prepared()` stages a batch of
  records extracted by Claude Code (id/type/labels/content/structure) — `_stamp_envelope` adds
  provenance/quality/version, `resolve_batch_refs` runs, the panel scores them, valid ones stage.
  Source text is cached to `canon_staging/_source_text/` for later re-validation.
- `canon_run.py` subcommands: **`stage --batch-file b.json [--no-panel]`** and
  **`validate-staged [--no-heavy]`** (run the panel over every still-`unverified` staged record
  against its cached source text, re-staging the verdict).
- **Seed-store population, batches 1–2** — Idli + Rasam (Wikipedia, CC-BY-SA), all records
  extracted by Claude Code, panel-validated (Qwen+Gemma: 18/19 unanimous accept), human-reviewed,
  promoted. New: `dish:{idli,rasam,milagu-rasam,tomato-rasam}`,
  `ing:{urad-dal-whole,idli-rice,fenugreek-seeds,tamarind,tomato,rasam-powder}`,
  `tech:{ferment,steam,wet-grind,gentle-simmer}`, `equip:wet-grinder`,
  `problem:idli-batter-wont-ferment`, `term:{tiffin,saaru,puli}`.
  **Canonical store: 49 → 77 records**, valid. Calibration log has 3 promote batches.

## [0.0.14] — 2026-09-03
### Added
- **First live pipeline run.** `canon_run.py run` on 3 CC-BY-SA Wikipedia articles (Sambar,
  Rasam, Avial): fetch → license (all `share_alike`) → Qwen-7B extract → Qwen+Gemma validate →
  stage. Reviewed the batch as human: dropped a junk `ing:vegetables`, corrected 4 records
  (transliterations to lowercase, `diet`, `category`), promoted **7 records** into
  `data/canonical/` — 3 `source`, `dish:avial`, `ing:coconut`, `ing:curd`, `term:sambar`.
  Store: 49 → **56 records**, valid.
- `data/eval/calibration_log.jsonl` — the 2.5%-sample audit trail (2 promote batches logged).
### Findings
- Local 7B extraction is mediocre (high schema-invalid rate; the panel accepted a
  category-as-entity). **Confirms the design: Claude Code extracts, local models validate.**
- `needs_human` gate works — nothing auto-promoted (confidence + single-source corroboration).
### Changed
- `test_pipeline*` assertions made robust to a growing store.

## [0.0.13] — 2026-09-02
### Added
- **`data/pipeline.py`** — `process_source(url, kind, *, fetcher, extract_provider, panel, …)`
  chains one source URL through the whole pipeline: fetch → license classify → build a `source`
  record → extract draft records → `resolve_batch_refs` (drop id-refs that resolve in neither
  the store nor this batch; note the drop in `content.summary`) → `validate_records` → stage.
  Nothing reaches `data/canonical/` — that stays `promote_staged`.
- **`scripts/canon_run.py`** — the pipeline driver: `gaps` (ranked coverage gaps), `run
  --sources s.json` (the full chain, providers from `configs/agents.yaml`), `review` (staged
  records by verdict), `promote --ids … --reviewer NAME`.
- The **5 remaining state-level `region` records** (Kerala, Karnataka, Andhra Pradesh,
  Telangana, Puducherry) — a dish's `cuisine_state` must resolve to a state region
  (`store.check_cuisine_states`). Store now 49 records.
- `tests/test_pipeline_run.py` (4).
### Fixed
- `ruff` strict-zip + E702 cleanups.

## [0.0.12] — 2026-09-02
### Added
- **`data/extract.py`** — page text → draft schema-v2 records. `build_extraction_prompt()` (for
  Claude Code to run directly) + `extract_records(text, source_id, provider)` (via any
  `Provider`). Builds `dish` / `ingredient` / `technique` / `term` records, slugs ids, stamps
  provenance, runs each through `parse_record` — invalid ones go to `.invalid`, not the drafts.
- **`data/validate.py`** — `validate_records(drafts, source_text, panel, ...)`:
  schema gate → forbidden-entity gate → **multi-family panel vote** (each model checks the
  record against its source text) → cross-source corroboration. Outcome: unanimous accept +
  corroboration ≥ 2 → `auto_validated`; unanimous reject → `rejected`; else `needs_human`.
  High-risk predicates and `confidence < 0.8` always route to human. Optional `heavy_provider`
  (deepseek-r1) tiebreak on a split.
- `llm.extract_json()` — robust JSON extraction (tolerates code fences + `<think>` blocks).
- `tests/test_extract.py` (8) + `tests/test_validate.py` (10, incl. 1 slow **live Ollama panel**
  test — Qwen + Gemma correctly accept a source-consistent record and reject a contradicted one).
  **105 tests pass** (102 fast + 3 slow).

## [0.0.11] — 2026-09-02
### Added
- **`data/dedup.py`** — `MinHashDeduper` (datasketch LSH over word 3-shingles; verbatim /
  lightly-edited copies), `SemanticDeduper` (fastembed bge-small cosine; paraphrases),
  `RecordDeduper` (draft vs store → new / merge / ambiguous, keyed on name + transliteration +
  ingredient set). `tests/test_dedup.py` (4). `pyproject.toml` registers the `slow` pytest marker.

## [0.0.10] — 2026-09-02
### Added
- **Extraction-pipeline spine — the LLM-independent half, built + tested:**
  - `data/ingest.py` — `Fetcher`: robots.txt (via `urllib.robotparser`) + per-domain rate limit
    + URL-hash response cache + `trafilatura` main-content extraction. One URL at a time, never a
    crawl. **Verified live** on Wikipedia (Bisi Bele Bath → 200, clean text incl. Kannada name;
    2nd fetch from cache).
  - `data/licensing.py` — `classify_license(html, url)` → `license_class` / `usable_for` from CC
    URLs, JSON-LD `license`, permissive/PD/ARR phrases, gov TLDs. Conservative default
    `unknown` → `reference_only`. **Verified**: Wikipedia → `share_alike` (CC-BY-SA).
  - `data/coverage.py` — `list_gaps(canonical_dir, targets)` → coverage of the store vs
    `coverage_targets.yaml`, ranked by the config's `priority_order`.
  - `data/canon_staging.py` — `stage_records` (schema-validate → `data/interim/canon_staging/`)
    and **`promote_staged(ids, reviewer)`** — the sole gate into `data/canonical/`: merges,
    re-validates the whole store, refuses on integrity break, stamps `human_validated`, logs a
    deterministic **2.5% calibration sample** to `data/eval/calibration_log.jsonl`.
  - `tests/test_pipeline.py` (11 tests). 84 total pass, ruff clean.

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
### Fixed
- `train()` gains `checkpoint_dir` + `overrides` params. `scripts/run_poc_checks.py` rewritten
  to use isolated temp checkpoint dirs (was reusing one dir → `resume auto` picked a stale
  checkpoint) and a tiny batch in deterministic-CPU mode so the check finishes in minutes.

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
