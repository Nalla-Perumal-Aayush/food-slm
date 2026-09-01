# Master Project Baseline
## From-Scratch Small Language Model — Tamil Nadu Vegetarian Cuisine

**Status:** Phases 1–3 frozen. Architecture / tokenization / training / data baseline **revised and re-frozen** (Revision 2, 2026-09-01). Model scale, context length, corpus strategy, reasoning success criterion, and capability tiers are now locked.
**Next stage:** Phase 4 — Data Source Identification & Collection Strategy (reorganized — see §8 and §13).

---

## Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | Aug 2026 | Initial freeze: problem, domain, ontology, hard infra constraints; architecture/tokenizer/training/eval left as *derived*. |
| 2 | 2026-09-01 | Model scale locked at **~89M (Config A)**; context flipped **6K/4K → 4K target / 2K fallback** (6K = future stretch); **general-English base corpus adopted**; **agent-generated synthetic culinary data** formalized as a pipeline; **"genuine multi-step reasoning" re-scoped** to structured/compositional reasoning over represented patterns; **capability tiers re-ranked** (abstention → Core; novel-combination + generalization-beyond-memorized → aspirational); **MHA locked** (GQA benchmark dropped); vocab locked at **~16k byte-level BPE**; **15M "Proof of Concept" model** added as the first rung of the scale ladder; verified hardware recorded. |

---

## 1. Problem Definition & System Requirements

| Field | Final Configuration |
|---|---|
| Project | True from-scratch Small Language Model |
| Initialization | Random initialization, no pretrained weights |
| Domain | Tamil Nadu vegetarian cuisine |
| Interaction language | English (input and output) |
| Reasoning | **Structured / compositional culinary reasoning over patterns represented in training** (re-scoped in Rev 2 — see §9.1) |
| RAG | Independent during core development; RAG is a separate later layer |
| Development period | 12–16 weeks |
| **Model scale** | **~89M parameters (Config A, §5). Scale ladder: 15M PoC → 40M → 89M target.** |
| **Context length** | **4K tokens target; 2K fallback if VRAM/throughput-constrained; trained via 1K→2K→4K curriculum. 6K = post-v1 stretch goal.** |
| Inference latency | 10–15 seconds maximum target (expected to be met by a wide margin at this scale) |
| Deployment | Hugging Face + GitHub, open source |
| Initial inference | Local GPU |
| Training precision | **bf16** (Ada-supported; preferred over fp16 for from-scratch stability) |

### 1.1 Verified Hardware (2026-09-01)

| Component | Spec'd | Verified actual |
|---|---|---|
| GPU | NVIDIA RTX 2000 Ada, 16 GB | RTX 2000 Ada, 16 GB, driver 573.44 / CUDA 12.8, **70 W power cap** (the real throughput limiter), bf16-capable |
| CPU | Intel Core i9 | i9-14900K (24C / 32T) |
| System RAM | 32 GB preferred | 64 GB DDR5-5600 |
| Storage | ≥2 TB | 512 GB NVMe (OS, near-full) + 2 TB HDD (~1.3 TB unallocated → to be partitioned as a dedicated data volume) |
| iGPU | — | Intel UHD 770 present → drive displays from iGPU to free VRAM |

The **70 W GPU power cap**, not the 16 GB VRAM, is the binding compute constraint. At ~89M the model + optimizer state is ~1.6 GB; VRAM is comfortable. Expect a full pretraining run of ~1–2 days.

---

## 2. Domain Scope & Capability Definition

**In scope:** recipes, ingredients, preparation, cooking techniques, measurements/conversions, recipe scaling, ingredient substitution, troubleshooting, regional variations, culinary terminology, equipment, food storage, culinary food safety.

**Out of scope:** medical diagnosis, disease treatment, clinical nutrition, general health consultation, general-purpose knowledge, general programming, general multilingual conversation, autonomous appliance control, general non-food reasoning.

**Capability hierarchy (re-ranked in Rev 2)**

| Tier | Capabilities |
|---|---|
| Core | Recipe understanding/generation, ingredients, procedures, measurements, terminology, basic (seen) substitution, basic troubleshooting, **calibrated abstention / low hallucination** |
| Advanced | Multi-step reasoning over represented patterns, constraint-based modification, ingredient-function reasoning, regional comparison |
| Secondary | Multi-turn context, equipment selection, storage guidance, comparative culinary analysis, recipe scaling / unit arithmetic (weak at this scale — pair with a calculation template) |
| Aspirational (not a v1 success condition) | Novel ingredient combinations, generalization substantially beyond memorized recipes, coherent long-context (>2K) multi-turn |
| Future | Vision, cooking-state recognition, multimodal interaction, appliance control |

**Capability validation principle:** every claimed capability must trace: `Capability → Expected Behaviour → Training Representation → Evaluation Task`.

---

## 3. Culinary Knowledge Ontology

**Core entities:** Dish, Ingredient, Recipe, Technique, Equipment, Region, Measurement, Culinary Outcome, Constraint, Problem/Symptom — each with its own attribute schema (e.g. Dish: name, aliases, category, region, ingredients, techniques, variants, serving context, expected outcome; Recipe: dish, servings, ingredients, quantities, ordered steps, techniques, equipment, timing, conditions, outcome).

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

**Reasoning supported by the ontology:** recipe modification, ingredient substitution, scaling, troubleshooting, technique reasoning, constraint reasoning, regional reasoning, outcome prediction, novel-combination reasoning.

The canonical knowledge store built from this ontology (§14 of the consolidated doc / Phase 5) is also the **sole grounding source for agent-generated synthetic data** (§8.2).

---

## 4. Data Architecture Principles

The ontology is not the training dataset — it's an intermediate structured representation.

```
Raw Sources → Data Acquisition → Cleaning → Normalization → Deduplication
→ Ontology Mapping → Structured Culinary Knowledge → Training Representation → Tokenizer → SLM
```

**Decided data principles:**
- AI-assisted collection/processing and AI-generated synthetic data are permitted (as augmentation, not replacement). Synthetic culinary data is generated only by serializing/paraphrasing canonical records, is fact-validated against canonical entities, is provenance-tagged, and is ratio-capped per capability (§8.2).
- Provenance, deduplication, and quality filtering are mandatory.
- Train/validation/test leakage prevention is mandatory. Splitting is **entity-disjoint** (a dish/recipe in train must not appear in val/test), not row-random — required because the model is expected to memorize.
- A **train × eval n-gram contamination scan** is a hard gate before any training run.
- Recipe measurements use standardized units; regional representation is evidence-driven.
- Copyright/licensing must be explicitly tracked; prefer public-domain/permissively licensed material; avoid sources with unclear training/redistribution rights. The general-English base corpus (§8.1) is **permissively licensed only**.

---

## 5. SLM Architecture

**Frozen:** Decoder-only autoregressive Transformer.

### 5.1 Config A — LOCKED reference architecture (~89M)

| Component | Locked value |
|---|---|
| Hidden dimension (d) | 640 |
| Layers (L) | 16 |
| Attention heads | 10 (head dim 64) |
| Attention type | **MHA** (GQA/MQA benchmark dropped — negligible benefit at this scale) |
| FFN | SwiGLU, inner dim ≈ 1728 (multiple of 64) |
| Activation | SiLU (within SwiGLU) |
| Positional encoding | RoPE (base 10000), trained at 2K then extended to 4K |
| Normalization | Pre-Norm RMSNorm + **QK-norm** |
| Logit stabilization | **z-loss** (≈ 1e-4) |
| Regularization | dropout 0.05–0.1 on residual + FFN (data-constrained repeated-epoch regime) |
| Weight tying | Enabled (input/output embeddings tied) |
| Vocabulary | ~16k (§6) |
| Parameter count | ~89M total (~79M non-embedding + ~10M tied embedding) |

### 5.2 Scale ladder

| Rung | Size | Purpose |
|---|---|---|
| **PoC** | **~15M** (e.g. d=320, L=8, heads=8) | **Proof of Concept — validates the entire toolchain end to end** (tokenizer → packing → training loop → checkpointing → eval harness) on a ~1-hour run before any large-scale data collection or training. First implementation milestone (see §15). |
| Mid | ~40M | Ablations (data mix, dropout, LR, curriculum steps) at low cost. |
| **Target** | **~89M (Config A)** | The released model. Architecture decisions are validated on PoC/Mid then committed here. |

**Still derived (require empirical validation):** exact FFN rounding, exact optimizer constants, exact global batch size and gradient-accumulation factor (from VRAM/throughput experiments), exact repetition/epoch count, final training-token budget.

---

## 6. Tokenization Architecture

**Locked:** vocabulary **~16k**, **byte-level BPE** (no UNK; Tamil proper nouns survive via byte fallback). English-only I/O makes a larger vocab wasteful for a narrow domain (16k×640 embedding ≈ 12% of parameters).

**Still benchmarked:** BPE vs Unigram *algorithm* on the final corpus — but vocab size and the byte-level property are fixed.

**Structural/control tokens:** ~10–15 justified tokens (`<DISH> <INGREDIENT> <QTY> <UNIT> <STEP> <TECHNIQUE> <TIME> <TEMP> <REGION> <OUTCOME> <CONSTRAINT>` + BOS/EOS/PAD and instruction-format markers) used for the instruction / serialization format only — **not** treated as a compression feature. Each must prove utility before inclusion.

**Selection criteria for the algorithm benchmark:** token fertility, compression ratio, vocabulary utilization, culinary-term fragmentation, number/unit efficiency, Tamil-name fragmentation, sequence-length efficiency, training throughput, and — decisively — **downstream SLM performance**.

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
Culinary Data
   ├─ General-English Base Corpus  → Language acquisition (NEW, Rev 2)
   ├─ Pretraining Corpus (culinary) → Domain language learning
   ├─ Instruction Data              → Capability learning
   └─ Evaluation Benchmark          → Independent testing (built before corpus freeze)
```

### 8.1 General-English base corpus (NEW — Rev 2)

- **Purpose:** give ~89M parameters genuine English syntax and basic reasoning patterns that a narrow culinary corpus cannot teach, and cut the required synthetic-culinary volume to a safe level.
- **Size target:** ~1.5–3B tokens.
- **Sources:** permissively-licensed / public-domain only (e.g. Wikipedia, public-domain books, openly-licensed how-to / procedural text). No unclear-rights sources.
- Used only in the early curriculum stages; SFT is culinary-only.

### 8.2 Synthetic culinary data pipeline (formalized — Rev 2)

- Agents generate **only** by serializing / paraphrasing records from the canonical knowledge store — never free-form "write a recipe".
- Techniques: template + slot-fill, multi-phrasing paraphrase, persona / difficulty / question-type variation.
- Every generated culinary **fact** is validated against a canonical entity; mismatches are rejected.
- Semantic dedup vs authentic data and vs other synthetic data.
- Provenance tag on every record: `synthetic=true`, generator, seed canonical record id.
- **Ratio caps per capability** — no capability may be 100% synthetic. With the general-English base in place, synthetic culinary is held to ≤ ~1–2× the authentic culinary volume.
- Synthetic data never enters the evaluation set.

### 8.3 Volume targets (89M model)

| Bucket | Target |
|---|---|
| Training tokens seen (culinary + base, total) | ~0.9–1.2B culinary-equivalent + base corpus |
| Unique culinary tokens | ~150–300M |
| Authentic culinary (realistic ceiling) | ~20–50M |
| Synthetic culinary | ~100–250M (capped per §8.2) |
| Instruction / SFT examples | ~30k–100k (≈ 15–50M tokens), ~60% agent-built from canonical store |

---

## 9. Evaluation Architecture

```
Model → Automated Metrics + AI Judge + Human Validation → Final Assessment
```

**Metric families:** perplexity, validation loss, factual accuracy, ingredient correctness, procedure correctness, structured multi-step reasoning accuracy, instruction adherence, substitution accuracy, scaling accuracy, regional accuracy, relevance, coherence, **hallucination rate, abstention correctness**, safety violation rate, consistency, **generalization gap (seen-dish vs unseen-dish accuracy)**, AI-as-a-Judge score, human evaluation score.

### 9.1 Re-scoped reasoning success criterion (Rev 2)

Success requires **structured / compositional reasoning over patterns represented in training** (e.g. applying a known substitution rule under a stated constraint, sequencing known techniques, comparing two documented regional variants). It does **not** require emergent reasoning over novel chains the model was never shown; that is aspirational (§2).

### 9.2 Integrity requirements

- Held-out **"forbidden entity" list** (dishes/ingredients that must never appear in any training data) frozen before data collection — makes hallucination probes genuine.
- Evaluation set built **before** the final corpus is frozen.
- Entity-disjoint train/val/test split; n-gram contamination scan as a hard pre-training gate.

Exact metric definitions, test-set sizes, scoring rubrics, and thresholds are finalized during the dedicated evaluation phase.

---

## 10. Deployment Architecture

```
Hugging Face Model → Tokenizer → Inference Engine → User Query → SLM → English Response
```

- Local GPU initially; ≤10–15s target latency (comfortably met at ~89M).
- Quantization and compression applied after the baseline model is established.
- Instruction format and canonical store are designed with retrieval-fusion slots so the later RAG layer is a drop-in (RAG itself is out of core-development scope).
- GitHub for source/documentation; Hugging Face for model distribution.
- Reproducible model configuration; dataset/model provenance documented.

---

## 11. Decision Status

### 🔒 Frozen now

True from-scratch SLM · Random initialization, no pretrained weights · Decoder-only Transformer · **Config A architecture (~89M): d=640, L=16, 10 MHA heads, SwiGLU≈1728, RoPE, pre-norm RMSNorm + QK-norm + z-loss, tied embeddings, dropout 0.05–0.1** · **Scale ladder 15M PoC → 40M → 89M** · **Vocabulary ~16k byte-level BPE** · Tamil Nadu vegetarian culinary domain · English interaction · **Structured/compositional reasoning success criterion** · **Re-ranked capability tiers (abstention = Core)** · **4K context target / 2K fallback via 1K→2K→4K curriculum; 6K = future** · **bf16 training** · **General-English base corpus (permissive only)** · **Formalized synthetic-data pipeline with per-capability ratio caps** · **Entity-disjoint splitting + contamination-scan gate + forbidden-entity list** · RTX 2000 Ada 16GB / i9-14900K / 64GB RAM / 2TB HDD · 12–16 week timeline · Independent of RAG in core dev · Open-source via Hugging Face + GitHub · Culinary ontology structure and typed directional relationships · Recipe process graph · Three-stage evaluation (automated + AI judge + human) · Provenance/licensing requirements · Standardized measurements

### 🟡 Proposed — pending experimental validation

BPE vs Unigram tokenizer *algorithm* · Exact FFN rounding · Learning rate, global batch size, gradient-accumulation factor · Repetition/epoch count and training-token budget · Context-extension schedule details · Quantization level · Exact evaluation thresholds

---

## 12. Proof-of-Concept Model (15M) — role and implications

The 15M model is **not a throwaway toy** — it is the project's first implementation milestone and de-risking instrument.

**What it proves:**
- The full pipeline runs end to end: canonical JSONL → tokenizer training → sequence packing → training loop → checkpoint/resume → evaluation harness → sample generation.
- The tokenizer choices (16k, byte-level, structural tokens) behave correctly on real culinary text.
- The training/eval code is correct and reproducible (seed → identical loss curve).
- Rough throughput on the 70 W GPU is measured, feeding the batch-size and token-budget decisions for 89M.

**What it does NOT prove:** final model quality. A 15M model on a small slice will be barely coherent — that is expected. Its outputs are judged for *pipeline correctness*, not culinary competence.

**Implication for the bigger implementation:** every architecture, tokenizer, data-format, and training-loop decision is exercised at 15M first, then 40M for cheap ablations, then committed at 89M. No decision is first tested at 89M. This keeps failed experiments to ~1-hour runs instead of ~2-day runs and is the main reason the 12–16 week window is feasible.

---

## 13. Reorganized Phase 4+ Data Block

The 51-phase roadmap is unchanged in overall order; the data block (Phases 4–16) is reorganized around the canonical store and the synthetic pipeline:

```
Phase 3  Ontology
Phase 4  Data Source ID + Licensing            (authentic culinary + permissive general-English)
Phase 5  Canonical Schema (JSONL knowledge store)   ← grounding source for all synthesis
Phase 6  Authentic Ingestion → Clean → Normalize → Dedup
Phase 7  Culinary Knowledge Extraction → populate canonical store
Phase 7b NEW  Synthetic Data Generation Pipeline    (grounded, validated, dedup'd, ratio-capped)
Phase 8  Pretraining Corpus Assembly                (general-English base + culinary authentic + culinary synthetic)
Phase 9  Instruction Dataset Construction           (mostly agent-built from canonical store)
Phase 10 Evaluation Dataset Construction            (before corpus freeze; forbidden-entity list; hallucination + generalization-gap probes)
Phase 11 Dataset Quality Validation + n-gram contamination scan (hard gate)
Phase 12 Train / Val / Test Split                   (entity-disjoint)
Phases 13–16 continue as before (schema finalization, corpus/instruction/eval construction detail, quality validation)
```

RAG integration (40–41), quantization/compression (44), alignment beyond abstention tuning, and any Smart-Glasses work are explicitly **v2** — outside the v1 critical path (Phases 1–37).

---

*This revision freezes model scale, context length, corpus strategy, the reasoning success criterion, and capability tiers, while leaving only the hyperparameters and tokenizer algorithm that genuinely require empirical evidence as derived decisions — validated bottom-up on the 15M PoC and 40M ablation models before commitment at 89M.*
