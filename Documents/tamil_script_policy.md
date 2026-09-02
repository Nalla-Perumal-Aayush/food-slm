# Tamil-Script Handling Policy (v1)

**Status:** frozen 2026-09-02. Resolves the open tokenizer question in `master_baseline.md` §6.
**Applies to:** the tokenizer, the ingestion/normalization pipeline, corpus assembly, the
canonical store, and evaluation.

---

## 1. The problem

The model's interface contract is **English-only** — the user writes English, the model
replies in English. But Tamil script (தமிழ்) is unavoidable in the *material*:

- dish, ingredient, technique and glossary names have native-script forms (`name_ta`);
- authentic sources include Tamil-language recipe blogs and cookbooks;
- users may paste a Tamil dish name into an otherwise-English prompt.

A byte-level BPE tokenizer with no `UNK` can *always* encode Tamil (UTF-8, 3 bytes/codepoint),
so nothing breaks — but with zero Tamil merges a dish name like `வெண் பொங்கல்` costs ~25 tokens.
The question is how much to invest in handling it, and how to keep it from leaking into places
it shouldn't.

---

## 2. Decision — "recognized, never required"

Tamil script is treated as **proper-noun reference material attached to English names**, never
as running prose the model reads or writes.

| | Policy |
|---|---|
| Model **output** | Never emits Tamil script. Not a capability; not evaluated as one. |
| Model **input** | Never *required* to contain Tamil script. A stray Tamil token in a prompt must be handled gracefully (byte-level guarantees no crash) but is out-of-distribution. |
| Tamil script in **training text** | Allowed only in a fixed inline form, capped at **≤ 0.5 % of pretraining tokens** (§4). |
| Tamil script in **the canonical store** | Kept in `name_ta` as reference. Not the join key. |
| The **join key** between English text and Tamil identity | The **popular romanized transliteration** (§3). |

Rationale: the domain is narrow and the compute budget is small (70 W GPU). Spending vocabulary
slots or context length on a script the model never produces is a poor trade. But discarding
Tamil identity entirely would make the store less useful for grounding and for human review, and
would fail queries that arrive transliterated.

---

## 3. Transliteration is the canonical bridge

Every Tamil-origin entity (`dish`, `ingredient`, `technique`, `term`, and Tamil `region` names)
**must** carry a `transliteration` value. It is the form that appears in running English text and
the primary lookup/alias key.

### 3.1 Romanization standard — practical, not scholarly

Use the **common romanization people actually type and search**, not ISO 15919 with diacritics:

| Use | Not |
|---|---|
| `kuzhambu` | `kuḻampu` |
| `poriyal` | `poriyal` (ok) / `poṟiyal` |
| `vengaya sambar` | `veṅkāya cāmpār` |
| `paruppu` | `paruppu` (ok) |
| `thayir` | `tayir` |
| `pongal` | `poṅkal` |

Conventions:
- `zh` for ழ (the retroflex approximant): `kuzhambu`, `pazham`, `thakkali`.
- `th` for தdental, `t`/`tt` for ட retroflex: `thengai`, `pattani`.
- Double a consonant for gemination as commonly written: `muttai`, `appalam`, `pulli`.
- Drop the inherent-vowel `a` at word end where common usage does: `sambar` not `sambara`.
- No diacritics, no macrons. ASCII only.
- When a widely-accepted English name exists, that is `name_en` and the transliteration is still
  recorded: `name_en: "Drumstick"`, `transliteration: "murungakkai"`, `name_ta: "முருங்கைக்காய்"`.

A `transliteration_scientific` (ISO 15919) field **may** be added later if disambiguation
demands it. It is **not** in scope for v1 and must not be collected now.

### 3.2 Aliases

Common alternative romanizations go in `slug_aliases` / entity `aliases`
(`kuzhambu` / `kulambu` / `kolambu`) so transliterated queries resolve regardless of the user's
spelling.

---

## 4. Tamil script in the pretraining corpus

### 4.1 Inline form (the only allowed shape)

On **first mention** of a Tamil-named entity in a document:

```
English Name (transliteration, தமிழ்)
```

e.g. `Ven Pongal (ven pongal, வெண் பொங்கல்)`, `drumstick (murungakkai, முருங்கைக்காய்)`.

**Subsequent mentions in the same document:** English name only. Never bare native script,
never native script without the English name beside it.

### 4.2 The glossary corpus

`terminology.jsonl` serialized to text is the one place native-script density is higher
(one line per term, each showing the script). This is small and bounded and is fine.

### 4.3 Hard cap

**Tamil-script codepoints ≤ 0.5 % of total pretraining tokens**, checked at corpus-assembly
time (`food_slm.data.assemble`). Over the cap ⇒ assembly fails; reduce native-script inclusion
(drop it from non-first mentions, trim the glossary weighting).

### 4.4 Tamil-language *source documents*

A source that is predominantly Tamil script (a Tamil recipe blog) is
`usable_for: [grounding]` or `[reference_only]` — **its facts are extracted into the canonical
store in English; its prose is never tokenized into `data/corpus/`.**

`food_slm.data.langid` adds a script gate: a document whose Tamil-script character ratio exceeds
**15 %** is routed to extraction-only and marked `corpus_eligible: false`, regardless of license.
(A permissive license makes the *facts* usable; it does not put Tamil prose in the corpus.)

---

## 5. Tokenizer training

1. **Include Tamil at its true frequency, no more.** The tokenizer-training sample is: all
   `name_ta` values in the canonical store + `terminology.jsonl` native forms + the capped
   (≤ 0.5 %) inline occurrences already present in the pretraining sample. Do **not** inject
   extra Tamil text to "balance" the tokenizer.
2. Expected: BPE learns **~200–500 Tamil merges** organically (frequent grapheme clusters —
   consonant + pulli, consonant + vowel sign, common food-vocabulary syllables).
3. **Guardrail: Tamil-script tokens ≤ 5 % of the final 16 k vocab (~820 slots).** If the
   algorithm benchmark exceeds this, down-sample Tamil in the tokenizer corpus rather than
   growing the vocab.
4. **Do not** hand-add Tamil tokens or a Tamil structural token. `<DISH>…</DISH>` wraps the
   English name; native script, if present, sits as plain text inside that span.
5. `nfkc_normalize: true` stays. On text already NFC-normalized at ingestion (§6) NFKC is
   effectively idempotent for modern Tamil, and it still fixes the Latin/punctuation
   compatibility cases that matter more.

### 5.1 Analysis-report targets (informational, not gates)

`src/food_slm/tokenizer/analyze.py` already emits `tamil_name_fragmentation`. Targets:

| Metric | Target |
|---|---|
| median tokens per canonical dish-name Tamil form | ≤ 6 |
| Tamil-script share of vocab | ≤ 5 % |
| Tamil-script share of pretraining tokens | ≤ 0.5 % (this one **is** a gate, §4.3) |

If Tamil fragmentation is poor but the token-share gate passes, **accept it** — do not spend
vocab or engineering effort chasing Tamil fertility.

---

## 6. Unicode normalization (ingestion)

`food_slm.data.normalize` applies, to every field and document, in order:

1. `ftfy.fix_text` (mojibake repair).
2. **NFC** normalization (canonical composition — the correct default for Tamil).
3. Strip `ZWJ` (U+200D) and `ZWNJ` (U+200C). Tamil rarely needs them; default is removal.
4. Normalize deprecated / compatibility forms: decompose the `ஸ்ரீ` (SHRII) ligature to its
   sequence; map any Tamil-numeral or fraction compatibility characters to ASCII digits.
5. Collapse repeated combining marks; drop isolated combining marks with no base.
6. Standard whitespace / quote / dash normalization (language-agnostic).

The tokenizer then sees already-NFC text and applies NFKC on top (§5.5).

---

## 7. Evaluation

- **Never penalize** an English answer for omitting Tamil script.
- **Do penalize** Tamil script attached to a non-Tamil concept, or an invented native-script
  spelling — that is a hallucination (feeds the hallucination-rate metric, §9 of the baseline).
- Add a small probe set: transliterated-name queries (`"how do I make kara kuzhambu"`) must
  resolve to the right entity as reliably as the English-name form — measures whether the
  transliteration bridge (§3) actually works.

---

## 8. What this changes in the repo

| Artifact | Change |
|---|---|
| `configs/tokenizer.yaml` | record the Tamil policy knobs (script caps, normalization form). |
| `Documents/phase5_canonical_schema.md` | `transliteration` REQUIRED for Tamil-origin entities; add `aliases` guidance; note `transliteration_scientific` is explicitly deferred. |
| `src/food_slm/data/langid.py` | 15 % Tamil-script ratio ⇒ `corpus_eligible: false`. |
| `src/food_slm/data/normalize.py` | the §6 pipeline. |
| `src/food_slm/data/assemble.py` | the ≤ 0.5 % pretraining-token gate. |
| `src/food_slm/tokenizer/analyze.py` | the §5.1 targets. |
