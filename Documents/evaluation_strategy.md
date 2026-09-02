# Evaluation Strategy — AI-Primary, Human-Anchored

**Status:** design frozen 2026-09-02. Tracks `master_baseline.md` §9. Applies to **both** data
validation (extraction pipeline) and model evaluation (Phases 31–37).

---

## 1. Principle

Evaluation runs on AI, anchored to a small amount of non-AI ground truth. **It must not be a
closed loop:** if AI extracts the data, AI validates it, and AI grades the model with no external
anchor, the model learns the judge's blind spots and every systematic AI error is extracted →
validated → trained → scored as correct (model collapse).

**Net human effort:** build ~1,500 golden items **once** + **2.5 % calibration sampling** ongoing
(≈ 2–4 hrs/week). Everything else is AI.

---

## 2. Ground-truth anchors — human-verified, built once, small

| Anchor | Size | Source | Purpose |
|---|---|---|---|
| **Forbidden-entity list** | ~100–150 (18 today) | hand-curated | known-answer hallucination / abstention probes |
| **Golden eval set** | ~500–1,000 QA pairs | **primary sources** (cookbooks, gov / agri docs) — never AI-generated; entity-disjoint; all 6 regions × veg/non-veg × every task type (§4.2 of baseline) | reference answers the AI judge scores against |
| **Golden validation set** | ~300–500 canonical records | a human marks each correct / incorrect | calibrates the data-validation panel |

These are the only items that *require* a human. They are frozen before the corpus (baseline §9.2).

---

## 3. AI does the scaling

### 3.1 Data validation (extraction pipeline)

Panel of **different model families** — Claude + GPT-4o + local Qwen2.5-7B — each independently
returns `accept | reject | uncertain` + a reason, against the schema and the source text. Plus
**cross-source corroboration**: a fact confirmed by ≥ 3 independent sources is reliable
regardless of any model's opinion (weight it above the panel).

- unanimous `accept` **and** `corroboration_count ≥ 2` → `auto_validated`, promotable without
  per-record human review
- any split, or a high-risk relationship (`substitutes_for`, `fixes_problem`,
  `produces_outcome`), or `confidence < 0.8` → human review queue

### 3.2 Model evaluation (Phases 31–37)

- **AI-judge panel** (§5) scores model outputs against the golden eval-set references, runs the
  full §9 metric suite, and does pairwise "which answer is better" comparisons — structured
  rubric, structured output.
- **Forbidden-entity probes** are scored against their *known* answers (did the model abstain on
  a `withheld_real` entity? did it refuse to invent a `fictitious` one?) — fully reliable, no
  judge subjectivity.
- **Automated metrics** (perplexity, generalization gap, format adherence, consistency,
  reproducibility) need no model at all.
- The AI judge is itself continuously scored against the golden sets; if its agreement with the
  golden labels falls, the judge is what's broken.

---

## 4. Calibration loop — the only recurring human cost

Each extraction batch / eval round:

1. A human checks a **random 2.5 %** of AI decisions (promoted records / judged outputs).
2. Track AI-vs-human agreement.
3. If agreement < **90 %** → **halt**. Diagnose (bad prompt? drifted panel? gap in golden set?),
   re-anchor, re-run the batch. Do not resume until agreement recovers.
4. Log every calibration round (`data/eval/calibration_log.jsonl`) — sample, agreement rate,
   action taken.

At ~50k lifetime records, 2.5 % ≈ 1,250 human checks total — spread over the timeline, trivial.

---

## 5. Anti-circularity measures

- **Judge family ≠ generator family.** Claude does extraction + synthesis, so the *deciding*
  judge/validator votes are **GPT-4o + Gemini 2.0 Pro + a local model**; Claude is only a
  tiebreak cross-check, never the deciding vote on its own training data.
- Golden sets come from **primary human sources**, never AI generation.
- **Synthetic data never enters the eval set** (hard rule, baseline §8.2).
- **Generalization gap** (seen-dish vs unseen-dish accuracy) runs automatically and is the
  canary — a widening gap means the model is memorizing or overfitting the judge.
- Entity-disjoint splits + the n-gram contamination scan + the forbidden-entity scan are hard
  pre-training gates regardless of any AI verdict.

---

## 6. Agent roster (evaluation) — ALL-FREE STACK

No paid API keys. Full roster: `configs/agents.yaml`.

| Role | Agent | Cost |
|---|---|---|
| AI-judge panel — deciding votes | **Qwen2.5-7B + Llama-3.1-8B + Gemma-2-9B** via Ollama (3 independent lineages) | free, local |
| AI-judge heavyweight cross-check | **deepseek-r1:32b** | free, local |
| Automated metrics · forbidden-entity probes | — (code) | free, local |
| Optional extra votes | Gemini free tier / Groq / GitHub Models (free signup) | free |

**Anti-circularity holds:** Claude Code does extraction + synthesis; the judge panel is Qwen +
Llama + Gemma + deepseek-r1 — no Claude-lineage model votes. Everything runs on the local GPU
(idle during eval) / CPU. Nothing but public source text leaves the machine.

---

## 7. What still needs a human — and always will

- Building the ~1,500 golden items (once).
- The 2.5 % calibration sample.
- "Does this recipe actually work when cooked?" — no AI can verify this; the golden eval set's
  recipes are sanity-checked by someone who cooks.
- Final release sign-off.
