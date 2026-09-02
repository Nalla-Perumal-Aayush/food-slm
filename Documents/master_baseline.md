# Master Project Baseline
## From-Scratch Small Language Model — South Indian Cuisine (vegetarian & non-vegetarian)

**Status:** Phases 1–3 frozen. Domain, architecture, tokenization, data-integrity framework, schema v2, extraction pipeline, and evaluation strategy frozen through Revision 7. **Revision 6 (2026-09-02): the final SLM is LOCKED at ~150M parameters (Config B — d=768, L=20, 12 heads, SwiGLU 1984), tolerance ±5% for implementation fixes. No longer data-driven — 150M is the release target. Data collection is scaled to whatever 150M needs (~15–45B training tokens), sourced from permissive/open corpora + the culinary pipeline — see §8.** This document is the **single final reference** — edited in place, not versioned as a separate file.
**Next stage:** Phase 4 — Data Source Identification & Collection Strategy (see §8 and §13), executed against the schema v2 deltas in §3.1.

---

## Revision History

| Rev | Date | Change |
|---|---|---|
| 7 | 2026-09-02 | **Extraction/synthesis pipeline + evaluation strategy designed.** `Documents/mcp_extraction_pipeline.md` — domain-agnostic **extract · synthesize · validate** engine; MCP server with 11 tools; Claude orchestrator + sub-agents; **self-hosted scraping only** (httpx+trafilatura / self-hosted Chromium; no scraping SaaS; robots.txt + rate-limit + cache); license classifier gates every source; agents write only to `data/interim/canon_staging/`, `promote_staged(reviewer)` is the sole path into `data/canonical/`; **facts not prose** (nothing scraped enters `data/corpus/`). `Documents/evaluation_strategy.md` — **AI-primary, human-anchored**: ground-truth anchors (forbidden-entity list + ~500–1000 golden QA pairs from primary sources + ~300–500 golden validation records) built once by a human; multi-family AI panels (GPT-4o + Gemini + local Qwen2.5-7B decide; Claude is tiebreak only — anti-circularity) do the scaling; **2.5% random human calibration** per batch/round, halt if AI-vs-human agreement < 90%. `configs/coverage_targets.yaml` (domain target matrix), `configs/agents.yaml` (role→provider roster), `src/food_slm/data/llm.py` (provider-agnostic multi-model client + Panel/consensus). |
| 6 | 2026-09-02 | **Final model size LOCKED at ~150M parameters** (was data-driven). **Config B** — d=768, L=20, 12 MHA heads (head_dim 64), SwiGLU `ffn_inner_dim` 1984, vocab 16384 → **151,225,600 params (+0.8% of 150M)**; accepted tolerance **±5%** for implementation fixes (vocab lock, FFN rounding). `configs/model_150m.yaml` created; validated by `scripts/validate_model.py`. 89M (Config A) demoted to an optional mid-scale de-risk / ablation rung. 350M rung removed. **Data collection scaled to 150M's needs** (~15–45B training tokens) — the general-English base grows to ~12–40B tokens from **permissive / openly-licensed corpora (downloaded, not self-crawled)**; culinary stays authentic + grounded-synthetic under the existing integrity gates. "Scrape everything" is **not** adopted — it would forfeit the clean open-source release (license contamination) and eval integrity; §8 explains why it is also unnecessary. Timeline for the 150M run: ~1.5–3 weeks on the 70 W GPU. |
| 5 | 2026-09-02 | **Domain expanded: Tamil Nadu vegetarian → South Indian cuisine.** Geography = Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, Telangana, Puducherry. Diet = **vegetarian + non-vegetarian at full depth** (seafood, poultry, red meat, egg). **Religion / community out of scope** — dietary constraints modelled practically and neutrally; dishes tagged by region/style, never by community or religion. Model stays a **food-domain specialist** (general-English corpus is a language substrate only). **Model size is data-driven** — scale ladder 15M → 40M → 89M (Config A) → 150M → 350M, the rungs above 89M gated on 89M eval curves; no pre-commitment. Tamil-script policy generalised to a **4-script Dravidian policy** (Tamil, Malayalam, Kannada, Telugu). **Vocabulary size re-opened for benchmark** (16k → 24k candidate). **Schema v2 deltas** specified (§3.1) — pending code. Eval sets + forbidden-entity list to be rebuilt at the new scope. MCP-orchestrated structured-extraction pipeline moved onto the critical path. **Timeline 12–16 weeks → 6–12 months.** |
| 4 | 2026-09-02 | **Tamil-script question resolved** (`Documents/tamil_script_policy.md`): "recognized, never required" — English-only I/O, native script only in a fixed inline form capped at ≤0.5% of pretraining tokens (hard gate), popular romanized transliteration is the join key, NFC at ingestion, Tamil-script vocab ≤5%, Tamil-heavy sources are extraction-only. **Forbidden-entity list frozen** before data collection (`data/eval/forbidden_entities.jsonl` + `.lock.json`, 18 seed entries, SHA-pinned, leak-checked in `scripts/validate_canonical.py`). Re-affirmed and hard-locked (no longer "proposed"): **4K context**, **bf16**, **general-English base corpus** (domain-only pretraining relaxation confirmed). |
| 3 | 2026-09-02 | **Phase 5 canonical schema finalized as v1** — `Documents/phase5_canonical_schema.md` + `src/food_slm/schema/{models,store}.py` + `configs/canonical_enums.yaml` / `configs/measures.yaml`. 13 record types, common envelope, measurement normalization, typed directional relationships, recipe process-DAG, referential-integrity validator (Phase 11 gate half). Seed store (`data/canonical/*.jsonl`, ven pongal worked example) ships and validates. |
| 2 | 2026-09-01 | Model scale locked at **~89M (Config A)**; context flipped **6K/4K → 4K target / 2K fallback** (6K = future stretch); **general-English base corpus adopted**; **agent-generated synthetic culinary data** formalized as a pipeline; **"genuine multi-step reasoning" re-scoped** to structured/compositional reasoning over represented patterns; **capability tiers re-ranked** (abstention → Core; novel-combination + generalization-beyond-memorized → aspirational); **MHA locked** (GQA benchmark dropped); vocab locked at **~16k byte-level BPE**; **15M "Proof of Concept" model** added as the first rung of the scale ladder; verified hardware recorded. |
| 1 | Aug 2026 | Initial freeze: problem, domain, ontology, hard infra constraints; architecture/tokenizer/training/eval left as *derived*. |

---

## 1. Problem Definition & System Requirements

| Field | Final Configuration |
|---|---|
| Project | True from-scratch Small Language Model |
| Initialization | Random initialization, no pretrained weights |
| Domain | **South Indian cuisine** — Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, Telangana, Puducherry; **vegetarian + non-vegetarian**. Food-domain specialist only (see §2). |
| Interaction language | English (input and output). Native Dravidian scripts are *recognised, never required* (§6). |
| Reasoning | **Structured / compositional culinary reasoning over patterns represented in training** (see §9.1) |
| RAG | Independent during core development; RAG is a separate later layer |
| Development period | **6–12 months** (revised from 12–16 weeks with the domain expansion) |
| **Model scale** | **LOCKED at ~150M (Config B, §5.1).** 151.2M params (+0.8% of the 150M target); tolerance **±5%** for implementation fixes. Ladder: 15M PoC → 40M ablations → 89M (Config A) optional de-risk → **150M release**. |
| **Context length** | **4K tokens (hard-locked).** 2K is a throughput fallback only; trained via 1K→2K→4K curriculum. 6K = post-v1 stretch. |
| Inference latency | 10–15 seconds maximum target (met by a wide margin at ≤350M) |
| Deployment | Hugging Face + GitHub, open source |
| Initial inference | Local GPU |
| Training precision | **bf16 (hard-locked)** — Ada-supported; preferred over fp16 for from-scratch stability |

### 1.1 Verified Hardware (2026-09-01)

| Component | Spec'd | Verified actual |
|---|---|---|
| GPU | NVIDIA RTX 2000 Ada, 16 GB | RTX 2000 Ada, 16 GB, driver 573.44 / CUDA 12.8, **70 W power cap** (the real throughput limiter), bf16-capable |
| CPU | Intel Core i9 | i9-14900K (24C / 32T) |
| System RAM | 32 GB preferred | 64 GB DDR5-5600 |
| Storage | ≥2 TB | 512 GB NVMe (OS, near-full) + 2 TB HDD (~1.3 TB unallocated → **must be partitioned as a dedicated data volume before Phase 6**). At the 150M / ~15–45B-token scale the project footprint is **~100–220 GB** (raw text ~40–120 GB + packed shards ~20–60 GB + checkpoints ~40 GB); the HDD partition is now mandatory. All data / caches / checkpoints stay on **D:**, never C:. |
| iGPU | — | Intel UHD 770 present → drive displays from iGPU to free VRAM |

The **70 W GPU power cap**, not the 16 GB VRAM, is the binding compute constraint. The **150M release run** costs roughly **1.5–3 weeks** at a ~15–45B-token budget (bf16, gradient checkpointing on for the 4K phase). The 15M / 40M / 89M rungs cost ~1 hour / ~half a day / ~1–3 days respectively and exist to keep failed experiments cheap.

---

## 2. Domain Scope & Capability Definition

### 2.1 Domain scope (Rev 5)

**Geography:** Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, Telangana, Puducherry. Culturally-adjacent coastal cuisines (Goa, coastal Maharashtra) and the Dravidian diaspora are **out of scope for v1**.

**Diet:** vegetarian **and** non-vegetarian at full depth — seafood, poultry, mutton/lamb, beef, pork, egg — all treated as culinary items.

**Religion & community: out of scope.**
- Dietary constraints are modelled **practically and neutrally** (`allium-free`, `vegan`, `egg-free`, `no-shellfish`, `contains-beef`, `contains-pork`, `diabetic-friendly`, `jain-style` as a cooking pattern) — never as religious doctrine.
- Dishes are tagged by **region and cooking style** (Malabar, Chettinad, Kongunadu, Udupi, Rayalaseema, Coorg, Hyderabadi-Deccan, "temple-style / saatvik" as an allium-free style) — **never** by community, caste, or religion.
- The model does not explain, attribute, or make claims about religious practice.

**In scope:** dishes, ingredients, recipes & procedures, cooking techniques, equipment, measurements/conversions, recipe scaling, ingredient substitution, troubleshooting, regional variation & comparison, culinary terminology (4 Dravidian languages ↔ English), food storage, culinary food safety (including non-veg handling).

**Out of scope:** nutrition / calorie data, medical or health advice, non-South-Indian cuisine, religious / theological / community content, general-purpose knowledge, general conversation, programming, multilingual conversation, appliance control, RAG (later layer), vision / multimodal.

### 2.2 Domain topics & their canonical record types

| Topic area | Record type(s) | Notes |
|---|---|---|
| Named dishes | `dish` | Categories: breakfast/tiffin, rice dishes, sambar / kuzhambu / pulusu, rasam / saaru, kootu / koottu, poriyal / thoran / palya, chutney / thogayal / pachadi, podi, biryani / pulao, non-veg curries, seafood, egg dishes, sweets / payasam, snacks, breads, pickles, beverages, festival meals (sadya, etc.) |
| Ingredients | `ingredient` | Incl. non-veg categories: `seafood`, `poultry`, `red_meat`, `egg` |
| Full procedures | `recipe` | Grouped ingredient lists w/ normalized quantities, ordered steps, the **process DAG**, timing, equipment, yield, variations |
| Techniques | `technique` | Tempering, dry/wet grinding, fermentation, pressure cooking, steaming, frying, braising, curry-base methods, biryani layering, dosa/appam making … |
| Equipment | `equipment` | Vessels, grinders, steamers, pans, presses, moulds |
| Regions & styles | `region` | 6 regions → sub-regions → cooking styles (no community tags) |
| Ingredient states | `state` | Intermediate states consumed/produced by the recipe DAG |
| Expected results | `outcome` | Texture / flavour / aroma / appearance / doneness |
| Dietary constraints | `constraint` | Practical & neutral only (see §2.1) |
| Troubleshooting | `problem` | Symptom → diagnosis → fix → prevention |
| Glossary | `term` | Tamil / Malayalam / Kannada / Telugu ↔ English |
| Provenance | `source` | Every fact traces to a source record |
| Typed edges | `rel` | Closed predicate vocabulary linking all of the above |

### 2.3 Capability hierarchy

| Tier | Capabilities |
|---|---|
| Core | Recipe understanding/generation, ingredients, procedures, measurements, terminology, basic (seen) substitution, basic troubleshooting, **calibrated abstention / low hallucination** |
| Advanced | Multi-step reasoning over represented patterns, constraint-based modification, ingredient-function reasoning, **cross-regional comparison** |
| Secondary | Multi-turn context, equipment selection, storage guidance, comparative culinary analysis, recipe scaling / unit arithmetic (weak at small scale — pair with a calculation template) |
| Aspirational (not a v1 success condition) | Novel ingredient combinations, generalization substantially beyond memorized recipes, coherent long-context (>2K) multi-turn |
| Future | Vision, cooking-state recognition, multimodal interaction, appliance control |

**Capability validation principle:** every claimed capability must trace `Capability → Expected Behaviour → Training Representation → Evaluation Task`.

---

## 3. Culinary Knowledge Ontology

**Core entities:** Dish, Ingredient, Recipe, Technique, Equipment, Region, Measurement, Ingredient State, Culinary Outcome, Constraint, Problem/Symptom, Term — **each with its own attribute schema** (a `dish` record ≠ an `ingredient` record ≠ a `recipe` record; they share only the common envelope, §3.2). E.g. Dish: names, aliases, category, cuisine_state, regions, style, primary ingredients, techniques, variants, serving context, expected outcome; Recipe: dish, servings, grouped ingredients + normalized quantities, ordered process-DAG steps, techniques, equipment, timing, conditions, outcome.

**Relationships are typed and directional**, e.g.:
```
Ingredient ──used_in──> Dish
Ingredient A ──substitutes_for──> Ingredient B
Technique A ──precedes──> Technique B
Technique ──transforms──> Ingredient State
Dish A ──variant_of──> Dish B
Dish ──belongs_to_region──> Region
```

**Recipe process model (state transitions):**
```
Input State → Preparation → Intermediate State → Cooking/Transformation → New State → Final Dish → Expected Outcome
```

**Reasoning supported by the ontology:** recipe modification, ingredient substitution, scaling, troubleshooting, technique reasoning, constraint reasoning, cross-regional reasoning, outcome prediction, novel-combination reasoning *(the last is aspirational — see §2.3)*.

The canonical knowledge store built from this ontology is also the **sole grounding source for agent-generated synthetic data** (§8.2). Schema **v2 is implemented** — `src/food_slm/schema/{models,store}.py`, `configs/canonical_enums.yaml`, spec `Documents/phase5_canonical_schema.md` §0.1. Migration `scripts/migrations/v1_to_v2.py` (idempotent) has been run on the seed store.

### 3.1 Schema v2 deltas (South India expansion)

| Area | v1 | v2 |
|---|---|---|
| Native name | single `name_ta` (Tamil only) | **`names[]`** — list of `{lang: ta\|ml\|kn\|te\|en, script: "...", transliteration: "..."}`; a native script entry requires its transliteration (popular romanization, not ISO 15919) |
| Cuisine dimension | implicit (all TN) | **`cuisine_state`** enum on every dish/recipe/ingredient/technique: `tamil_nadu \| kerala \| karnataka \| andhra_pradesh \| telangana \| puducherry \| pan_south` |
| Region hierarchy | flat `region:` | `region` gains `level` (`state \| sub_region \| style`) + `parent_region`; styles (Malabar, Chettinad, Udupi, Rayalaseema, Coorg, Hyderabadi-Deccan …) are `level: style` |
| Diet | vegetarian only | `dish`/`recipe` gain **`diet_class`** (`vegetarian \| contains_egg \| contains_dairy \| seafood \| poultry \| red_meat`); `ingredient.category` adds `seafood \| poultry \| red_meat \| egg` |
| Constraint kinds | incl. `ritual_religious` | **`ritual_religious` removed.** `constraint.kind` ∈ `dietary \| allergy \| preference \| medical \| cooking_style` — `no-onion-no-garlic` is reframed `cooking_style` ("allium-free / saatvik style"), no doctrinal rationale |
| Community/religion tags | — | **explicitly disallowed** — the schema validator rejects any `structure` field or label matching a community/religion vocabulary; regional/style tags only |
| Non-veg technique params | — | `technique`/`recipe` step params gain optional `internal_temp` / `doneness_cue` for meat & seafood; food-safety notes carried in `content` |

Schema-version string is `"2"`. `scripts/migrations/v1_to_v2.py` (idempotent) wraps `name_ta`/`term_ta` into `names[]`, backfills `cuisine_state`, infers `diet` from ingredient categories / allergen classes, sets `region.level` + parent chain (adding a state-level region where missing), and reframes `constraint.kind`. Records that trip the community-tag rule after migration are surfaced by `scripts/validate_canonical.py` for manual reframing.

### 3.2 Common record envelope (all types)

Identity (`id`, `type`, `schema_version`, `labels`) + **`content`** (human-readable prose) + **`structure`** (typed / enum / normalized fields the pipeline reasons over) + **`provenance`** + **`quality`** + **`version`**. Full field spec: `phase5_canonical_schema.md` §3.

---

## 4. Data Architecture Principles

The ontology is not the training dataset — it's an intermediate structured representation.

```
Raw Sources → Data Acquisition → Cleaning → Normalization → Deduplication
→ Ontology Mapping (canonical store) → Serialization + Synthesis → Training Text → Tokenizer → Packed Shards → SLM
```

### 4.1 Three data representations

| # | Representation | Location | Format | Role |
|---|---|---|---|---|
| 1 | **Canonical knowledge store** | `data/canonical/*.jsonl` (versioned) | JSON Lines, one file per record type, common envelope (§3.2) | Source of truth / fact oracle. Grounding source for all synthesis; validation reference; basis for entity-disjoint splits. **Not** trained on directly. |
| 2 | **Training corpus** | `data/corpus/*` (gitignored, regenerable) | (a) assembled text → (b) packed token-ID shards (`uint16`/`uint32` memmap `.bin`/`.npy`) + JSON manifest (source mix, hashes, split, curriculum length) | What the model trains on. Derived from #1 (serialized + synthesized) + general-English base + authentic culinary prose. |
| 3 | **Instruction / SFT data** | `data/instruction/*.jsonl` | `{"messages": [...], "provenance": {...}}` chat format, chat tokens | Capability + behavioural tuning. ~60% agent-built by serializing #1 into Q&A. |

### 4.2 Response format by task type

The model outputs natural English prose. Consistency of shape comes from training on consistently-serialized examples (a **convention**, not constrained decoding). The structural tokens (`<DISH> <INGREDIENT> <STEP>` …) mark these spans in training data. Each task's shape mirrors the schema of its underlying record type:

| Task | Response shape | Backing record type |
|---|---|---|
| Recipe request | intro line → grouped ingredient list w/ quantities → numbered steps → tips / common mistakes → serving suggestion | `recipe` |
| Ingredient substitution | substitute + ratio + what changes + caveat (1–3 sentences) | `rel` (`substitutes_for` attributes) |
| Scaling | adjusted quantity table + non-linear notes (spice, cooking time, vessel) | `recipe` + measurement rules |
| Troubleshooting | symptom → likely cause → fix → prevention | `problem` |
| Technique explanation | what it is → how → sensory "done" cues → common mistakes | `technique` |
| Regional comparison | dimension-by-dimension parallel (ingredients / method / texture / accompaniments) | 2× `dish` + `rel` |
| Definition / factual | concise 1–3 sentences | `term` / `ingredient` |
| Out-of-scope / unknown | explicit abstention | — |

**Decided data principles:**
- AI-assisted collection/processing and AI-generated synthetic data are permitted (as augmentation, not replacement). Synthetic culinary data is generated only by serializing/paraphrasing canonical records, is fact-validated against canonical entities, is provenance-tagged, and is ratio-capped per capability (§8.2).
- An **MCP-orchestrated structured-extraction pipeline** (`Documents/mcp_extraction_pipeline.md`) is on the critical path — Claude orchestrator + sub-agents, **self-hosted scraping only**. It extracts *structured facts* into the canonical store; it does **not** scrape prose into `data/corpus/`. Copyrighted recipe sites are `reference_only`: facts are extracted and re-expressed; expressive prose is not trained on. Agents write only to `data/interim/canon_staging/`; `promote_staged(reviewer)` is the sole path into `data/canonical/`. Validation is **AI-primary, human-anchored** (`Documents/evaluation_strategy.md`): multi-family AI panels + cross-source corroboration decide; a **2.5% random human calibration sample** per batch, halt if agreement < 90%. The same engine runs backwards for synthesis (§8.2).
- Provenance, deduplication, and quality filtering are mandatory.
- Train/validation/test leakage prevention is mandatory. Splitting is **entity-disjoint** (a dish/recipe in train must not appear in val/test), not row-random — required because the model is expected to memorize.
- A **train × eval n-gram contamination scan** + a **forbidden-entity scan** are hard gates before any training run.
- Recipe measurements use standardized units (§ measurement normalization); regional representation is evidence-driven and balanced across the 6 regions and across veg/non-veg.
- Copyright/licensing must be explicitly tracked per source record; prefer public-domain/permissively licensed material; avoid sources with unclear training/redistribution rights. The general-English base corpus (§8.1) is **permissively licensed only**.

---

## 5. SLM Architecture

**Frozen:** Decoder-only autoregressive Transformer.

### 5.1 Config B — LOCKED release architecture (~150M)

The final released model. `configs/model_150m.yaml`. Same architecture family as Config A —
only the dimensions differ.

| Component | Locked value |
|---|---|
| Hidden dimension (d) | 768 |
| Layers (L) | 20 |
| Attention heads | 12 (head dim 64) |
| Attention type | **MHA** (`num_kv_heads` = 12; GQA/MQA benchmark dropped) |
| FFN | SwiGLU, inner dim **1984** (multiple of 64, ≈ 2.58 d) |
| Activation | SiLU (within SwiGLU) |
| Positional encoding | RoPE (base 10000), trained at 1K → 2K → 4K |
| Normalization | Pre-Norm RMSNorm + **QK-norm** |
| Logit stabilization | **z-loss** (≈ 1e-4) |
| Regularization | dropout 0.05–0.1 on residual + FFN |
| Weight tying | Enabled (input/output embeddings tied) |
| Vocabulary | 16384 (24k under benchmark, §6) |
| **Parameter count** | **151,225,600** (~138.6M non-embedding + ~12.6M tied embedding at vocab 16384) = **+0.8 % of the 150M target**. Tolerance **±5 %** (142.5M–157.5M). If the tokenizer benchmark locks vocab at **24k** (+6.3M embedding → ~157.5M, at the edge): trim `ffn_inner_dim` → 1792 or L → 19 to re-centre. Any such change is an allowed "fix" inside the ±5 % band. |

### 5.1a Config A — 89M (optional de-risk / ablation rung)

`configs/model_89m.yaml`. d=640, L=16, 10 heads, SwiGLU 1728, vocab 16384 → 89,807,488 params.
Not a release target under Rev 6. Used, if time allows, as an intermediate scale between the
40M ablations and the 150M run to catch scaling surprises cheaply.

### 5.2 Scale ladder

| Rung | Size | Role |
|---|---|---|
| **PoC** | **~15M** (d=320, L=8) | **Proof of Concept — validates the whole toolchain end to end** (tokenizer → packing → training loop → checkpoint/resume → eval harness) on ~1-hour runs. First milestone (§12). Domain-agnostic. |
| Ablation | ~40M (d=512, L=10) | Cheap sweeps — data mix, dropout, LR, curriculum steps, tokenizer algorithm. |
| De-risk | ~89M (Config A) | Optional. One intermediate scale-check before committing the multi-week 150M run. |
| **Release** | **~150M (Config B)** | **The final model.** Every arch / tokenizer / data decision is exercised at 15M → 40M (→ 89M) before the 150M run. |

**Sizing is locked** — 150M ±5%. If the 150M eval shows the model is *data*-limited (plausible for a bounded domain), the response is more / better data, **not** a bigger model. There is no 350M rung.

**Still derived (require empirical validation):** optimizer constants, global batch size and gradient-accumulation factor, repetition/epoch count, final training-token budget, vocab-size choice (§6), and the exact FFN/layer trim if vocab locks at 24k.

---

## 6. Tokenization Architecture

**Locked:** **byte-level BPE** (no UNK; Dravidian proper nouns survive via byte fallback). English-only I/O.

**Re-opened for benchmark (Rev 5):** **vocabulary size — 16k vs 24k** (candidate 24k, given 4 Dravidian scripts + a ~5× larger English culinary/regional vocabulary than the TN-veg scope). Decide on the assembled South India corpus. Also still benchmarked: BPE vs Unigram *algorithm*. The byte-level property is fixed.

**Dravidian-script handling — 4-script policy (Rev 5, generalizes `Documents/tamil_script_policy.md`):** policy is *"recognized, never required"* for **Tamil, Malayalam, Kannada, Telugu**. The model never emits native script and is never required to read it; a stray native token in a prompt is handled by byte fallback but is out-of-distribution. Native script appears in training text only as `English Name (transliteration, «native script»)` on first mention — e.g. `Puttu (puttu, പുട്ട്)`, `Bisi Bele Bath (bisi bele bath, ಬಿಸಿ ಬೇಳೆ ಬಾತ್)` — **capped at ≤ 0.5 % of pretraining tokens (hard gate at corpus assembly)**, measured across all four scripts combined. The **popular romanized transliteration** (not ISO 15919) is the join key and is required for every entity that carries a native name (per-language romanization guides to be written, one per script). Ingestion normalizes native script to **NFC**; the tokenizer's NFKC runs on top. Tokenizer sees native script only at true corpus frequency (no injected balancing text); **native-script tokens ≤ 8 % of vocab** (raised from 5% for four scripts; still a guardrail — down-sample if exceeded). Source documents > 15 % native-script characters are **extraction-only** — facts populate the canonical store in English, prose never enters `data/corpus/`.

**Structural/control tokens:** ~10–15 justified tokens (`<DISH> <INGREDIENT> <QTY> <UNIT> <STEP> <TECHNIQUE> <TIME> <TEMP> <REGION> <OUTCOME> <CONSTRAINT>` + BOS/EOS/PAD and instruction-format markers) used for the instruction / serialization format only — **not** a compression feature. Each must prove utility before inclusion. No per-script structural token.

**Selection criteria for the benchmarks:** token fertility, compression ratio, vocabulary utilization, culinary-term fragmentation, number/unit efficiency, per-script native-name fragmentation, sequence-length efficiency, training throughput, and — decisively — **downstream SLM performance**.

---

## 7. Training Architecture

| Component | Recommendation |
|---|---|
| Objective | Causal next-token prediction |
| Optimizer | AdamW (β 0.9 / 0.95, weight decay 0.1, grad clip 1.0) |
| LR schedule | Warmup (2–3%) + cosine decay to ~10% of peak |
| Precision | **bf16** |
| Batch strategy | Micro-batch + gradient accumulation to a ~0.5M-token global batch |
| Gradient checkpointing | Off at ≤2K context; on for the 4K extension phase |
| Efficient attention | PyTorch SDPA (no flash-attn dependency on native Windows) |
| Sequence packing | Yes |
| Checkpointing | Periodic + best-validation checkpoint |
| Experiment tracking | Required |
| Reproducibility | Required (seeds, configs, data manifest hashes) |
| Training duration | Defined in tokens/steps, not arbitrary epochs |

### 7.1 Curriculum (Rev 2)

```
general / procedural English (language acquisition)
  → culinary prose (domain adaptation)
  → structured recipes (format learning)
  → instruction data (capability learning)
  → RoPE context extension 2K → 4K
  → SFT (culinary instruction only)
  → abstention / behavioural tuning
```

Repeated data is planned for ~4–6 epochs on the culinary portion (safe zone for data-constrained training); val-loss curves are monitored for memorization onset.

---

## 8. Training Data Architecture

```
Training inputs
   ├─ General-English Base Corpus   → language acquisition (substrate only)
   ├─ Pretraining Corpus (culinary) → domain language learning (authentic + synthetic, from the canonical store)
   ├─ Instruction Data              → capability + behavioural learning
   └─ Evaluation Benchmark          → independent testing (built before corpus freeze; no synthetic)
```

### 8.0 Data-volume strategy for 150M (Rev 6)

**Target: ~15–45B training tokens seen** (≈ 100–300 tokens/param — the healthy range for a
small model trained past Chinchilla-optimal). Compute-optimal alone (20 tok/param ≈ 3B) would
under-train a from-scratch model.

**Acquisition timing.** The bulk corpus is downloaded at **Phase 8**, *after* the tokenizer is
trained and the packing code works, the Phase 4 source list is licence-vetted, and the
dedicated HDD partition exists. The **15M PoC and 40M ablations run on a small development
sample** (~200–500 MB / ~100–300M tokens) — a streamed slice of a permissive corpus plus the
seed recipes — because they are judged for pipeline correctness, not model quality. Do **not**
download tens of GB before the mix ratios (§8.3) are fixed.

**"Scrape all of internet" is not adopted — and is not needed.**

- *Not needed:* the general-English base can reach 15–45B tokens by **downloading existing
  permissively-/openly-licensed corpora** — FineWeb-Edu (~1.3T tokens, ODC-By), C4 (~175B,
  ODC-By), Wikipedia (~4B), Project Gutenberg public-domain books (~5–10B), permissive
  Stack-Exchange / how-to text, OpenWebMath. A 150M model's whole budget is a small slice of
  these. Self-crawling raw web would only reproduce — worse — the cleaning, dedup, quality
  filtering and de-duplication those datasets already did.
- *Not adopted:* training on indiscriminately-scraped web text would (a) **forfeit the clean
  open-source release** — unclear-rights text makes the weights legally unshippable as open
  weights, and (b) **destroy evaluation integrity** — a crawler that ingests a page about a
  held-out dish silently contaminates the benchmark. The §4 data principles (permissive-only
  base, provenance, entity-disjoint splits, contamination + forbidden-entity gates) are **not
  relaxed**.
- *What crawling legitimately buys:* culinary **breadth** — coverage of distinct dishes /
  techniques / regional variants. That is captured by the **MCP structured-extraction pipeline**
  (facts → canonical records, re-expressed, provenance-tagged), not by raw token volume. Recipe
  sites are almost all all-rights-reserved → `reference_only`; their prose never enters
  `data/corpus/`. The domain is bounded (~15–40k distinct dishes) — 10× more crawling yields
  ~10× more *duplicated* prose, not 10× more knowledge.

### 8.1 General-English base corpus

- **Purpose:** give the model genuine English syntax and basic reasoning patterns a domain corpus cannot teach, and cut the required synthetic-culinary volume to a safe level. **Substrate only** — the model stays a food-domain specialist; SFT is culinary-only.
- **Size target:** **~12–40B tokens** for the 150M release (dominant share of the corpus; the culinary portion is domain-bounded).
- **Sources:** permissively-/openly-licensed only (see §8.0). **Download, don't scrape.** No unclear-rights sources.
- Used only in the early curriculum stages.

### 8.2 Synthetic culinary data pipeline (formalized — Rev 2)

- Agents generate **only** by serializing / paraphrasing records from the canonical knowledge store — never free-form "write a recipe".
- Techniques: template + slot-fill, multi-phrasing paraphrase, persona / difficulty / question-type variation.
- Every generated culinary **fact** is validated against a canonical entity; mismatches are rejected.
- Semantic dedup vs authentic data and vs other synthetic data.
- Provenance tag on every record: `synthetic=true`, generator, seed canonical record id.
- **Ratio caps per capability** — no capability may be 100% synthetic. With the general-English base in place, synthetic culinary is held to ≤ ~1–2× the authentic culinary volume.
- Synthetic data never enters the evaluation set.

### 8.3 Volume targets (Rev 6 — 150M release)

| Bucket | Target | Notes |
|---|---|---|
| Canonical store records | ~10k–25k (`configs/coverage_targets.yaml` sums to ~20k) | comprehensive dish/ingredient/technique/region coverage, 6 regions, veg + non-veg |
| Authentic culinary tokens (unique) | ~150–500M | bounded by what exists + is licensable/extractable |
| Synthetic culinary tokens (unique) | ~0.3–1B | capped ≤ 1–2× authentic (§8.2) |
| **Unique culinary tokens** | **~0.5–1.5B** | the domain ceiling — not raised by more crawling |
| General-English base (unique) | **~12–40B** | permissive/open, downloaded (§8.0) |
| **Total training tokens seen** | **~15–45B** | culinary repeated ~4–6 epochs, base ~1–2 epochs |
| Instruction / SFT examples | ~100k–300k (≈ 50–150M tokens) | ~60% agent-built from the canonical store |
| **On-disk footprint** | **~100–220 GB** | raw text ~40–120 GB + packed shards ~20–60 GB + checkpoints ~40 GB |

The culinary portion is **domain-bounded** (~0.5–1.5B unique tokens); the general-English base
carries the token count up to what 150M needs. This split is why 150M — not larger — is the
locked size: beyond ~150–200M the extra capacity has no *culinary* signal to learn and just
memorizes.

---

## 9. Evaluation Architecture

```
Model → Automated Metrics + AI-Judge Panel + 2.5% Human Calibration → Final Assessment
```

**AI-primary, human-anchored** (`Documents/evaluation_strategy.md`). Human effort is bounded:
build ~1,500 golden items once (forbidden-entity list + golden QA set from primary sources +
golden validation records), then a **2.5% random calibration sample** per eval round (halt if
AI-vs-human agreement < 90%). The deciding judge panel is **GPT-4o + Gemini + a local model**
(never Claude alone — Claude does extraction/synthesis, so it is a tiebreak cross-check only).
Forbidden-entity probes and automated metrics (perplexity, generalization gap, format adherence,
reproducibility) need no judge at all.

**Metric families:** perplexity, validation loss, factual accuracy, ingredient correctness, procedure correctness, structured multi-step reasoning accuracy, instruction adherence, substitution accuracy, scaling accuracy, **per-region accuracy (all 6 regions reported separately)**, **veg / non-veg parity (accuracy gap between the two must be small)**, response-format adherence (§4.2), relevance, coherence, **hallucination rate, abstention correctness**, safety violation rate (incl. non-veg handling), **religion/community-leakage rate (must be ~0 — the model must not attribute dishes to communities or explain religious practice)**, consistency, **generalization gap (seen-dish vs unseen-dish accuracy)**, AI-as-a-Judge score, human evaluation score.

### 9.1 Re-scoped reasoning success criterion (Rev 2)

Success requires **structured / compositional reasoning over patterns represented in training** (e.g. applying a known substitution rule under a stated constraint, sequencing known techniques, comparing two documented regional variants). It does **not** require emergent reasoning over novel chains the model was never shown; that is aspirational (§2).

### 9.2 Integrity requirements

- Held-out **"forbidden entity" list** (dishes/ingredients that must never appear in any training data) frozen before data collection — makes hallucination probes genuine. Mechanism live (`data/eval/forbidden_entities.jsonl` + `.lock.json`, SHA-pinned, amendments logged; `food_slm.data.forbidden.verify_lock` gates every training run; leak-checked against the canonical store by `scripts/validate_canonical.py` and against the assembled corpus by the Phase 11 gate). **Rev 5: the list must be rebuilt and re-frozen at the South India scope** — target ~80–150 entries spanning all 6 regions and veg + non-veg, both `withheld_real` (abstention) and `fictitious` (fabrication). The 18 TN-veg seed entries carry over.
- Evaluation set built **before** the final corpus is frozen, balanced across the 6 regions and veg/non-veg, with a dedicated religion/community-leakage probe set. The **golden QA set (~500–1000 pairs) and golden validation set (~300–500 records) come from primary human sources, never AI generation** (`Documents/evaluation_strategy.md` §2).
- Entity-disjoint train/val/test split; n-gram contamination scan + forbidden-entity scan as hard pre-training gates.

Exact metric definitions, test-set sizes, scoring rubrics, and thresholds are finalized during the dedicated evaluation phase.

---

## 10. Deployment Architecture

```
Hugging Face Model → Tokenizer → Inference Engine → User Query → SLM → English Response
```

- Local GPU initially; ≤10–15s target latency (comfortably met at 150M).
- Quantization and compression applied after the baseline model is established.
- Instruction format and canonical store are designed with retrieval-fusion slots so the later RAG layer is a drop-in (RAG itself is out of core-development scope).
- GitHub for source/documentation; Hugging Face for model distribution.
- Reproducible model configuration; dataset/model provenance documented.

---

## 11. Decision Status

### 🔒 Frozen now

True from-scratch SLM · Random initialization, no pretrained weights · Decoder-only Transformer · **Config B architecture — the LOCKED release (~150M ±5%): d=768, L=20, 12 MHA heads, SwiGLU 1984, RoPE, pre-norm RMSNorm + QK-norm + z-loss, tied embeddings, dropout 0.05–0.1 → 151.2M params** · **Ladder 15M PoC → 40M ablations → 89M (Config A) optional de-risk → 150M release; no 350M rung** · **Target ~15–45B training tokens; general-English base from permissive/open corpora (downloaded, not crawled); indiscriminate scraping rejected (§8.0)** · **Byte-level BPE** (no UNK) · **South Indian culinary domain — TN, Kerala, Karnataka, Andhra Pradesh, Telangana, Puducherry; vegetarian + non-vegetarian** · **Religion / community out of scope (neutral practical constraints; region/style tags only)** · **Food-domain specialist (general-English = substrate only)** · English interaction · **Dravidian 4-script policy "recognized, never required" (≤0.5% of tokens, transliteration bridge, NFC)** · **Structured/compositional reasoning success criterion** · **Capability tiers (abstention = Core)** · **4K context (hard-locked); 2K = throughput fallback; 1K→2K→4K curriculum** · **bf16 (hard-locked)** · **General-English base corpus (permissive only, download not scrape)** · **Formalized synthetic-data pipeline with per-capability ratio caps** · **MCP extract·synthesize·validate engine (self-hosted scraping, facts not prose, `promote_staged` sole path into the store)** · **AI-primary human-anchored evaluation (multi-family panels; 2.5% human calibration; golden sets from primary sources)** · **Entity-disjoint splitting + contamination scan + forbidden-entity scan (hard gates)** · **Common record envelope + per-type content/structure schemas** · Typed directional relationships · Recipe process-DAG · Standardized measurement normalization · Three-stage evaluation (automated + AI judge + human) · Provenance/licensing requirements · RTX 2000 Ada 16GB / i9-14900K / 64GB RAM · 6–12 month timeline · Independent of RAG in core dev · Open-source via Hugging Face + GitHub

### 🟡 Proposed — pending experimental validation

**Vocabulary size 16k vs 24k** · BPE vs Unigram tokenizer *algorithm* · Config B FFN/layer trim **iff** vocab locks at 24k (stay inside ±5%) · Learning rate, global batch size, gradient-accumulation factor · Repetition/epoch count and final training-token budget within the ~15–45B range · Context-extension schedule details · Whether the 89M de-risk rung is run · Quantization level · Exact evaluation thresholds

### 📝 Pending code (specified, not yet implemented)

Per-script romanization guides (Malayalam, Kannada, Telugu) · Rebuilt forbidden-entity list at South India scope (~100–150 entries) · **MCP server + tools** (`data/{mcp_server,licensing,ingest,extract,dedup}.py`, `scripts/canon_run.py`) · Golden QA + golden validation sets (Phase 10) · Eval **suites / judge / report** (`eval/suites.py`, `judge.py`, `report.py`) · Curriculum stage ordering + RoPE context extension (`train/curriculum.py`)

*Done (Fork A toolchain):* schema **v2** · **`FoodSLM` model** validated at 15M / 40M / 89M / **150M (Config B)** · **byte-level BPE tokenizer** · **canonical→text serializer** · **PoC dev-sample fetcher** · **sequence packing** · **training loop** (bf16 AdamW, warmup+cosine, grad-accum, z-loss) · **bit-exact checkpoint/resume** · **eval perplexity + generation**. 15M PoC runs end-to-end on the 70 W GPU.

*Done (pipeline design):* `Documents/mcp_extraction_pipeline.md` + `Documents/evaluation_strategy.md` · `configs/coverage_targets.yaml` + `configs/agents.yaml` · `src/food_slm/data/llm.py` (provider-agnostic multi-model client + Panel/consensus, `FakeProvider` for tests).

---

## 12. Proof-of-Concept Model (15M) — role and implications

The 15M model is **not a throwaway toy** — it is the project's first implementation milestone and de-risking instrument.

**Progress (Fork A):** ✅ **model** validated at 15M/40M/89M/**150M (Config B)**. ✅ **tokenizer** — byte-level BPE + analyzer. ✅ **serializer** + **PoC sample fetcher**. ✅ **sequence packing** — deterministic `uint16` shards + `PackedDataset`. ✅ **training loop** (`train/loop.py`) — bf16 AdamW, warmup+cosine, grad-accum, grad-clip, z-loss; **bit-exact checkpoint/resume** (`train/checkpoint.py` — model+optim+RNG + deterministic `BatchSampler`); periodic val eval. ✅ **eval** — `perplexity.py` + `generate.py` (temperature/top-k/top-p). **15M PoC verified end-to-end on the RTX 2000 Ada** (loss 9.77→7.0/60 steps, ~38k tok/s; interrupt→resume continues cleanly). **Remaining:** populate a real seed corpus + run the full ~1-hour 15M PoC and eyeball samples; then 40M ablations.

**What it proves:**
- The full pipeline runs end to end: canonical JSONL → tokenizer training → sequence packing → training loop → checkpoint/resume → evaluation harness → sample generation.
- The tokenizer choices (byte-level, structural tokens, script handling) behave correctly on real culinary text.
- The training/eval code is correct and reproducible (seed → identical loss curve).
- Rough throughput on the 70 W GPU is measured, feeding the batch-size, token-budget, and rung-feasibility decisions.

**What it does NOT prove:** final model quality. A 15M model on a small slice will be barely coherent — that is expected. Its outputs are judged for *pipeline correctness*, not culinary competence. **The PoC is domain-agnostic** — it validates the toolchain identically whether the domain is TN-veg or all of South India, so it is not blocked on the schema v2 work.

**Implication:** every architecture, tokenizer, data-format, and training-loop decision is exercised at 15M first, then 40M for cheap ablations (and optionally 89M) — never first at the 150M release scale. This keeps failed experiments to ~1-hour runs and is what makes the multi-week 150M run a single, well-understood commitment.

---

## 13. Reorganized Phase 4+ Data Block

The 51-phase roadmap is unchanged in overall order; the data block (Phases 4–16) is reorganized around the canonical store and the synthetic pipeline:

```
Phase 3  Ontology
Phase 4  Data Source ID + Licensing            (South India authentic culinary + permissive general-English)
         ✅ pipeline + eval strategy DESIGNED — Documents/mcp_extraction_pipeline.md,
            Documents/evaluation_strategy.md, configs/coverage_targets.yaml, configs/agents.yaml
Phase 5  Canonical Schema — ✅ v2 IMPLEMENTED (src/food_slm/schema/, migration run on seed store)
Phase 6  Authentic Ingestion → Clean → Normalize (Dravidian NFC) → Dedup   (self-hosted scraping)
Phase 7  Culinary Knowledge Extraction → populate canonical store
         (MCP: list_gaps → propose_sources → [human OK] → register_source → fetch → extract →
          resolve → validate [AI panel] → stage → [2.5% human calibration] → promote_staged)
Phase 7b Synthetic Data Generation Pipeline        (same engine backwards: synthesize → validate)
Phase 8  Pretraining Corpus Assembly               (general-EN base + culinary authentic + synthetic;
                                                    ≤0.5% native-script token gate)
Phase 9  Instruction Dataset Construction          (mostly agent-built from canonical store)
Phase 10 Evaluation Dataset Construction           (before corpus freeze; rebuilt forbidden-entity
                                                    list; per-region + veg/non-veg + community-leakage
                                                    + generalization-gap probes)
Phase 11 Dataset Quality Validation + n-gram contamination scan + forbidden-entity scan (hard gates)
Phase 12 Train / Val / Test Split                  (entity-disjoint)
Phases 13–16 continue (corpus/instruction/eval construction detail, quality validation)
```

RAG integration (40–41), quantization/compression (44), alignment beyond abstention tuning, and any Smart-Glasses work remain **out of the v1 critical path** (Phases 1–37).

---

*Rev 5 expanded the domain to South Indian cuisine (6 regions, veg + non-veg). **Rev 6 locks the final model at ~150M parameters (Config B, ±5%)** and sizes the data programme to it (~15–45B training tokens; general-English base from permissive/open corpora, downloaded not scraped). **Rev 7 designs the extract·synthesize·validate engine** (`Documents/mcp_extraction_pipeline.md`, self-hosted scraping, facts-not-prose) and the **AI-primary, human-anchored evaluation** (`Documents/evaluation_strategy.md`, multi-family panels, 2.5% human calibration, golden sets from primary sources). Frozen: domain, architecture (Config B), byte-level tokenization, 4K context, bf16, corpus strategy, data-integrity framework, extraction pipeline design, evaluation strategy, reasoning criterion, capability tiers, schema v2. Derived (empirical): vocabulary size, tokenizer algorithm, training hyperparameters — decided bottom-up on 15M → 40M → (89M) before the single 150M run.*
