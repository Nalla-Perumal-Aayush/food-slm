# Phase 5 — Canonical Knowledge-Store Schema (v2)

**Status:** authoritative for `data/canonical/**`. Executable form: `src/food_slm/schema/{models,store}.py`.
**Owner:** Aayush · **Date:** 2026-09-02 · **Tracks:** `master_baseline.md` Revision 5, §3 / §3.1 / §4 / §8.2.

> **v2 (South Indian domain) — implemented 2026-09-02.** Migration `scripts/migrations/v1_to_v2.py`
> (idempotent) is run on the seed store. The base sections below describe the shared design; the
> v2-specific field changes are consolidated in **§0.1** and mirror `master_baseline.md` §3.1.

## 0.1 v2 changes vs v1

| Area | v1 | v2 |
|---|---|---|
| Domain | Tamil Nadu vegetarian | South Indian, 6 regions, vegetarian + non-vegetarian |
| Native name | `name_ta` + `transliteration` + `ipa` on each named structure | **`names: [LocalName]`** — `LocalName{lang: ta\|ml\|kn\|te, native, transliteration (required), primary}`; ≤1 primary, no duplicate lang |
| `term` record | `term_ta` + `transliteration` | `lang` + `native` + `transliteration` (single language) |
| Cuisine dimension | implicit | **`cuisine_state`** `tamil_nadu\|kerala\|karnataka\|andhra_pradesh\|telangana\|puducherry\|pan_south` — **required** on `dish`/`recipe`, optional on `ingredient`/`technique` |
| `region` | flat, `parent_region` only | **`level`** `state\|sub_region\|style` (required) + `cuisine_state`; non-state levels need a `parent_region`; state level must not have one |
| Diet | vegetarian only | **`diet: DietProfile`** on `dish`/`recipe` — booleans `vegetarian, vegan, contains_dairy, contains_egg, contains_seafood, contains_poultry, contains_red_meat` with consistency validation. `ingredient.category` adds `seafood\|poultry\|red_meat\|egg` |
| `constraint.kind` | `dietary\|ritual_religious\|allergy\|preference\|medical\|seasonal_ayurvedic` | **`dietary\|allergy\|preference\|medical\|cooking_style`** (`ritual_religious`/`seasonal_ayurvedic` removed) |
| Community / religion | — | **rejected** — the per-record validator fails on a community/religion word in any label or `structure` value; `store.check_community_tags()` extends the scan to `content` prose (Phase 11 gate). Core blocklist in `models.DISALLOWED_COMMUNITY_TERMS`; use regional / cooking-style tags instead |
| Process-step params | heat / duration / temperature | + **`internal_temp`** (Measurement) + **`doneness_cue`** (str) for meat & seafood |
| `dish.style` / `recipe.style` | — | free-text cooking style (`chettinad`, `malabar`, `udupi`, …) |
| `schema_version` | `"1"` | `"2"` |

`transliteration_scientific` (ISO 15919) remains **explicitly deferred** — do not collect it.

---

## 0. Purpose and non-goals

The canonical store is the **single structured source of truth** for South Indian
cuisine used by this project. It is:

- the **grounding source for 100% of synthetic data** (§8.2 of the baseline — agents may only
  serialize/paraphrase records that exist here);
- the **fact oracle** for per-fact validation of synthetic text and for hallucination scoring;
- the basis for **entity-disjoint** train/val/test splitting (splits are computed over entity IDs
  defined here, never over rows).

It is **not** the training corpus. Records here are converted to training text in Phase 8; the
store itself is never fed to the model verbatim as a file.

Non-goals: multilingual *content bodies* (English prose only; native scripts kept only as
`LocalName.native` / `term.native` reference strings), nutrition / calorie modelling, image data,
religious / community attribution, RAG index structures.

---

## 1. Physical layout

One JSONL file per record type under `data/canonical/` (all versioned in git):

| File | Record `type` | Holds |
|---|---|---|
| `sources.jsonl` | `source` | Every origin a fact can be attributed to. |
| `ingredients.jsonl` | `ingredient` | Ingredients + their culinary properties. |
| `dishes.jsonl` | `dish` | Named dishes (abstract — not a specific recipe). |
| `recipes.jsonl` | `recipe` | Concrete procedures for a dish, incl. the process graph. |
| `techniques.jsonl` | `technique` | Cooking techniques (tempering, dry-roasting, …). |
| `equipment.jsonl` | `equipment` | Tools/vessels. |
| `regions.jsonl` | `region` | Geographic / cultural regions within Tamil Nadu. |
| `states.jsonl` | `state` | Named ingredient/dish states used by process graphs. |
| `outcomes.jsonl` | `outcome` | Expected sensory/textural end results. |
| `constraints.jsonl` | `constraint` | Dietary / ritual / preference constraints. |
| `problems.jsonl` | `problem` | Failure symptoms ("idli batter didn't ferment"). |
| `terminology.jsonl` | `term` | Culinary glossary entries (Tamil term ↔ English gloss). |
| `relationships.jsonl` | `rel` | All typed directional edges between the above. |

**One record per line.** UTF-8. Keys sorted. No trailing whitespace. Files are append-friendly
but MUST be re-sorted by `id` before commit (a `make canonical-sort` target will do this).

---

## 2. Entity ID scheme

```
<prefix>:<slug>
```

| Prefix | Type | Example |
|---|---|---|
| `src`     | source     | `src:sharmis-passions` |
| `ing`     | ingredient | `ing:urad-dal-whole` |
| `dish`    | dish       | `dish:ven-pongal` |
| `recipe`  | recipe     | `recipe:ven-pongal--classic` |
| `tech`    | technique  | `tech:tadka` |
| `equip`   | equipment  | `equip:idli-steamer` |
| `region`  | region     | `region:kongu-nadu` |
| `state`   | state      | `state:dal-soft-mashed` |
| `outcome` | outcome    | `outcome:fluffy-non-sticky` |
| `constraint` | constraint | `constraint:no-onion-no-garlic` |
| `problem` | problem    | `problem:idli-batter-not-fermented` |
| `term`    | term       | `term:kuzhambu` |
| `rel`     | relationship | `rel:00042` (zero-padded counter, 5 digits) |

**Slug rules:** lowercase; ASCII only (transliterate Tamil, drop diacritics); words joined by `-`;
recipe slugs use `<dish-slug>--<variant-slug>` (double dash separates dish from variant).
**IDs are immutable and never reused.** A wrong/merged record is marked `status: "merged"` with
`merged_into` pointing at the survivor; it stays in the file for referential integrity.

---

## 3. Common record envelope

Every record in every file carries these blocks. This is the "20-field" core
(5 identity + 4 content-pointer + 3 provenance + 4 quality + 4 version).

```jsonc
{
  // --- identity (5) ---
  "id":             "dish:ven-pongal",
  "type":           "dish",
  "schema_version": "1",
  "labels":         ["Ven Pongal", "Khara Pongal"],   // display names, first = canonical
  "slug_aliases":   [],                                 // other slugs that redirect here

  // --- content: free-text, English, human-readable (varies per type, see §4) ---
  "content": { /* type-specific */ },

  // --- structure: typed/enumerated/normalized fields the reasoner uses (see §4) ---
  "structure": { /* type-specific */ },

  // --- provenance (3) ---
  "provenance": {
    "source_ids":     ["src:sharmis-passions"],   // -> sources.jsonl; [] only for hand_authored
    "source_urls":    ["https://..."],
    "acquired_on":    "2026-09-10",
    "transformations":["scraped", "manual-normalize", "unit-canonicalize"],
    "synthetic":      false,
    "generator":      null,        // e.g. "claude-sonnet-5" when synthetic
    "seed_record_ids":[]           // canonical records a synthetic record was grounded on
  },

  // --- quality (4) ---
  "quality": {
    "confidence":         0.9,                    // 0-1 subjective reliability
    "validation_status":  "human_validated",      // see §5
    "corroboration_count":2,                      // # independent sources agreeing
    "notes":              null
  },

  // --- version (4) ---
  "version": {
    "revision":   3,
    "created_on": "2026-09-10",
    "updated_on": "2026-09-28",
    "status":     "active"        // active | deprecated | merged
  },
  "merged_into": null             // set iff status == "merged"
}
```

### Rules

- `synthetic: true` ⇒ `generator` set, `seed_record_ids` non-empty, and the record MUST NOT
  introduce any fact (ingredient, quantity, step, relationship) absent from its seed records.
  Synthetic records are **excluded from `data/eval/**` by construction**.
- `validation_status: "rejected"` records are kept (for the training of abstention behaviour and
  for negative mining) but are never serialized into the positive training corpus.
- `confidence < 0.5` OR `corroboration_count == 0` for a non-hand-authored record ⇒ excluded from
  synthetic-data grounding until validated.

---

## 4. Per-type `content` / `structure` fields

Enumerated values marked **[enum]** are closed vocabularies defined in §7. Adding a value =
schema change (bump a `terminology`/enum table + this doc).

> **v2 note:** the per-type field lists in §4.x below are the **v1** shape. For v2 apply the
> deltas in **§0.1** (native names → `names[]`; add `cuisine_state`, `diet`, `region.level`,
> non-veg categories; `constraint.kind` change; community-tag rejection; step `internal_temp` /
> `doneness_cue`). **`src/food_slm/schema/models.py` is the authoritative field list.**

**Native-name rule (all named types — ingredient, dish, technique, equipment, region):**
names live in `names: [LocalName]`; every `LocalName` carries a `native` string and a **required**
`transliteration` (popular romanization, not ISO 15919 — see `Documents/tamil_script_policy.md` §3
and `master_baseline.md` §6). A `term` record carries a single `lang` / `native` / `transliteration`.
Alternative romanizations go in entity `aliases` / `slug_aliases`.

### 4.1 `source`
- `content`: `title`, `authors` (list), `description`, `notes`
- `structure`:
  - `kind` **[enum]**: `recipe_blog` `cookbook` `gov_doc` `agri_extension` `cultural_doc`
    `academic` `wikipedia` `video_transcript` `community_forum` `hand_authored`
  - `license` — SPDX id or explicit statement string (required)
  - `license_class` **[enum]**: `public_domain` `permissive` `share_alike` `noncommercial`
    `unknown` `all_rights_reserved`
  - `usable_for` **[enum list]**: `training` `grounding` `reference_only` — set from `license_class`
  - `authority` **[enum]**: `primary` `secondary` `tertiary`
  - `region_focus` (list of `region:` ids, optional)
  - `accessed_on` (date)

### 4.2 `ingredient`
- `content`: `summary` (1–3 sentences), `description`, `selection_notes`, `storage_notes`,
  `preparation_notes`
- `structure`:
  - `name_ta`, `transliteration`, `ipa` (all optional) — **but if `name_ta` is set,
    `transliteration` is required** (Tamil-script policy §3; popular romanization, not ISO 15919).
    `transliteration_scientific` (ISO 15919) is explicitly **deferred** — do not collect it in v1.
  - `category` **[enum]**: `dal_legume` `rice_grain` `flour` `vegetable` `leafy_green` `fruit`
    `spice_whole` `spice_ground` `herb` `tempering_agent` `souring_agent` `sweetener` `fat_oil`
    `dairy` `nut_seed` `flavouring` `leavening` `salt` `water` `other`
  - `culinary_roles` **[enum list]**: `base` `thickener` `souring` `tempering` `aromatic`
    `sweetening` `binding` `garnish` `fat` `leavening` `colouring` `bulking`
  - `forms` — list of `{form [enum], notes}`; form ∈ `whole` `split` `skinned` `ground` `paste`
    `roasted` `soaked` `sprouted` `fresh` `dried` `powder` `flattened`
  - `default_unit` **[enum]** — canonical measurement unit normally used (see §6)
  - `flavour_profile` **[enum list]**: `sweet` `sour` `salty` `bitter` `umami` `pungent`
    `astringent` `nutty` `earthy` `neutral`
  - `common_substitutes` — list of `ing:` ids (detail/ratios live in `relationships.jsonl`)
  - `allergen_class` **[enum list]**: `nuts` `dairy` `gluten` `legume` `sesame` `mustard` `none`
  - `seasonality` (list of months or `year_round`)

### 4.3 `dish`
- `content`: `summary`, `description`, `serving_context`, `cultural_notes`, `history`
- `structure`:
  - `name_ta`, `transliteration`
  - `category` **[enum]**: `tiffin` `rice_dish` `kuzhambu` `sambar` `rasam` `kootu` `poriyal`
    `masiyal` `thogayal` `chutney` `podi` `pachadi` `appalam_vadagam` `sweet` `snack` `beverage`
    `pickle` `bread` `festival_special`
  - `course` **[enum list]**: `breakfast` `lunch` `dinner` `tiffin` `side` `main` `accompaniment`
    `dessert` `festival`
  - `primary_ingredients` — list of `ing:` ids
  - `typical_techniques` — list of `tech:` ids
  - `regions` — list of `region:` ids
  - `expected_outcome_ids` — list of `outcome:` ids
  - `constraints_typically_met` — list of `constraint:` ids (e.g. many temple dishes are
    `constraint:no-onion-no-garlic`)
  - `variant_of` — `dish:` id or null
  - `festival_associations` (list of strings)

### 4.4 `recipe` — the process graph lives here
- `content`: `summary`, `intro`, `tips` (list), `common_mistakes` (list), `serving_suggestions`
- `structure`:
  - `dish_id` — `dish:` id (required)
  - `variant_label` (e.g. "hotel style", "Chettinad", "no coconut")
  - `yield` — `{servings: int|null, quantity: <measurement>|null}`
  - `total_time` — `<measurement>` (minutes), `active_time` — `<measurement>`
  - `difficulty` **[enum]**: `easy` `moderate` `advanced`
  - `ingredients` — ordered list of `<recipe-ingredient>` (§4.10)
  - `equipment_ids` — list of `equip:` ids
  - `steps` — ordered list of `<process-step>` (§4.11)  ← the DAG
  - `states` — list of `state:` ids introduced/consumed by this recipe
  - `outcome_ids` — list of `outcome:` ids
  - `constraints_met` — list of `constraint:` ids
  - `scalable` (bool), `make_ahead` (bool)

### 4.5 `technique`
- `content`: `summary`, `description`, `how_to`, `sensory_cues` (how you know it's done),
  `failure_modes`
- `structure`:
  - `name_ta`, `transliteration`
  - `family` **[enum]**: `heat_dry` `heat_fat` `heat_moist` `mechanical` `fermentation`
    `soaking_hydration` `tempering` `resting` `assembly`
  - `typical_equipment_ids` — list of `equip:` ids
  - `inputs_state_change` — list of `{from_state: state:id|null, to_state: state:id}`
  - `typical_params` — `{heat_level [enum], duration: <measurement>, temperature: <measurement>}`;
    heat_level ∈ `low` `medium-low` `medium` `medium-high` `high`
  - `precedes` / `follows` — lists of `tech:` ids (ordering hints; authoritative edges in `rel`)

### 4.6 `equipment`
- `content`: `summary`, `description`, `alternatives_notes`
- `structure`: `name_ta`, `transliteration`, `category` **[enum]** (`vessel` `stove` `grinder`
  `steamer` `pan` `press` `utensil` `storage`), `substitutes` (list of `equip:` ids),
  `required_for_techniques` (list of `tech:` ids)

### 4.7 `region`
- `content`: `summary`, `description`, `culinary_signature`
- `structure`: `name_ta`, `transliteration`, `parent_region` (`region:` id | null),
  `districts` (list of strings), `signature_dishes` (list of `dish:` ids),
  `signature_ingredients` (list of `ing:` ids)

### 4.8 `state`
- `content`: `description`, `sensory_markers`
- `structure`: `applies_to` **[enum]** (`ingredient` `mixture` `dish`),
  `properties` — dict of `{texture, colour, temperature_band, moisture, structure}` (free strings),
  `reachable_from` (list of `state:` ids), `via_technique_ids` (list of `tech:` ids)

### 4.9 `outcome`
- `content`: `description`, `positive_markers`, `negative_markers`
- `structure`: `dimension` **[enum list]** (`texture` `flavour` `aroma` `appearance`
  `temperature` `consistency` `doneness`), `measurable_by` (free strings)

### 4.9b `constraint`
- `content`: `description`, `rationale`
- `structure`:
  - `kind` **[enum]**: `dietary` `ritual_religious` `allergy` `preference` `medical` `seasonal_ayurvedic`
  - `excludes_ingredients` (list of `ing:` ids), `excludes_categories` (list of ingredient
    category enums), `requires` (free strings)
  - `common_substitution_strategy` (free string)

### 4.9c `problem`
- `content`: `description`, `symptoms`, `diagnosis`, `fixes` (list), `prevention` (list)
- `structure`:
  - `scope` **[enum]**: `ingredient_prep` `fermentation` `cooking` `texture` `flavour` `storage`
  - `related_dish_ids` / `related_technique_ids` / `related_ingredient_ids`
  - `root_causes` (list of free strings), `fix_relationship_ids` (list of `rel:` ids using
    predicate `fixes_problem`)

### 4.9d `term`
- `content`: `gloss` (English), `usage_notes`, `example_sentence`
- `structure`: `term_ta`, `transliteration`, `part_of_speech`, `maps_to_id` (any entity id | null),
  `category` **[enum]** (`ingredient` `dish` `technique` `equipment` `measure` `general`)

### 4.10 `<recipe-ingredient>` sub-object
```jsonc
{
  "ingredient_id": "ing:urad-dal-whole",
  "quantity":      { /* <measurement>, see §6 */ },
  "form":          "whole",              // [enum] from ingredient.forms
  "preparation":   "soaked 30 min",      // free text
  "role_in_recipe":"base",               // [enum] culinary_roles
  "optional":      false,
  "substitutable": true,
  "group":         "for the tempering"   // section header, nullable
}
```

### 4.11 `<process-step>` sub-object — nodes of the recipe DAG
```jsonc
{
  "step_id":     "s3",                       // unique within the recipe
  "order":       3,                          // presentation order
  "phase":       "temper",                   // [enum] prep|soak|grind|ferment|temper|cook|assemble|finish|rest|serve
  "action":      "Add cumin and pepper to the hot ghee and let them crackle.",
  "action_verb": "temper",                   // [enum] §7.4 (controlled verb)
  "input_ingredient_ids": ["ing:cumin-seeds", "ing:black-pepper-whole"],
  "input_state_ids":      ["state:ghee-hot"],
  "output_state_ids":     ["state:tempering-ready"],
  "equipment_id":         "equip:kadai",
  "technique_id":         "tech:tadka",
  "params": {
    "heat_level":     "medium",              // [enum]
    "duration":       { /* <measurement> minutes */ },
    "until_condition":"seeds crackle and turn aromatic"
  },
  "depends_on":  ["s2"],                     // step_ids that must complete first (DAG edges)
  "parallel_ok": false
}
```
**DAG rules:** `depends_on` must reference earlier `step_id`s only (no cycles); every
`input_state_id` must be an `output_state_id` of some prior step or a state with
`reachable_from: []` (a starting state); the final step's `output_state_ids` must include a state
whose `applies_to == "dish"`.

---

## 5. `validation_status` lifecycle

```
unverified ──auto checks pass──▶ auto_validated ──human review──▶ human_validated
     │                                │                                  │
     └────────────── any check fails ─┴──────────────────────────────────┘
                                      ▼
                                  rejected  (kept, never in positive corpus)
```

- **auto checks** (Phase 11 gate, run by `food_slm.data.contamination` + a schema validator):
  referential integrity (every id referenced exists), enum conformance, measurement
  canonicalization present, DAG acyclic, no synthetic fact outside seed set, unit sanity ranges.
- **human_validated** required for: every `dish`, every `recipe`, every `rel` with predicate
  `substitutes_for` / `fixes_problem` / `produces_outcome`, and any record with `confidence < 0.8`.
- Synthetic records inherit `auto_validated` at best; they reach `human_validated` only by
  sampling audit.

---

## 6. Measurement normalization

Every quantity is an object, never a bare number or string:

```jsonc
{
  "original_text": "1 heaped tsp",
  "value":         1.0,
  "unit":          "tsp",            // as-written, [enum] §7.3
  "value_canonical": 6.0,            // converted
  "unit_canonical":  "g",           // SI-ish base per dimension
  "dimension":     "mass",           // [enum] mass|volume|count|time|temperature|length|qualitative
  "approximate":   true,
  "range":         null,             // [lo, hi] in canonical unit, or null
  "qualitative":   null              // [enum] to_taste|a_pinch|a_few|handful|as_needed|garnish — set iff dimension==qualitative
}
```

**Canonical bases:** mass → `g`, volume → `ml`, count → `count`, time → `min`,
temperature → `C`, length → `cm`.

**Conversion policy:**
- Volume→mass conversions for dry ingredients use a per-ingredient density table
  (`configs/measures.yaml`, keyed by `ingredient_id` + `form`). If no density is known,
  keep `dimension: "volume"` and leave `value_canonical` in `ml`. Never guess density.
- "tsp/tbsp/cup" use the Indian metric convention: tsp = 5 ml, tbsp = 15 ml, cup = 240 ml
  (recorded in `configs/measures.yaml` so it's overridable).
- Vague quantities ("a pinch", "to taste") → `dimension: "qualitative"`, `value: null`.
- Ranges ("2–3 cups") → `value` = midpoint, `range` = `[lo, hi]` canonical, `approximate: true`.

---

## 7. Controlled vocabularies (closed sets)

Kept machine-readable in `configs/canonical_enums.yaml` (the schema validator loads it; this doc
is the human copy). Changing any list is a schema change → bump `schema_version` minor.

- **7.1 record types:** the 13 in §1.
- **7.2 relationship predicates** (`relationships.jsonl`, all directional subject→object):
  `used_in` · `substitutes_for` · `precedes` · `transforms_state` · `variant_of` ·
  `belongs_to_region` · `requires_technique` · `requires_equipment` · `pairs_with` ·
  `part_of` · `causes_problem` · `fixes_problem` · `violates_constraint` · `satisfies_constraint` ·
  `produces_outcome` · `has_alias` · `derived_from` · `seasonal_in`
  Each `rel` record may carry `attributes` (e.g. `substitutes_for` → `{ratio, context, caveat}`;
  `used_in` → `{typical_quantity: <measurement>, essential: bool}`).
- **7.3 measurement units:** mass `mg g kg` · volume `ml l tsp tbsp cup` · count `count clove
  sprig piece` · time `sec min hr` · temperature `C` · qualitative (see §6).
- **7.4 controlled action verbs (process steps):** `soak` `drain` `rinse` `grind` `blend`
  `mix` `knead` `ferment` `rest` `heat` `temper` `saute` `fry` `deep_fry` `roast` `dry_roast`
  `boil` `simmer` `pressure_cook` `steam` `add` `combine` `mash` `season` `garnish` `cool`
  `serve` `reduce` `strain` `cover` `flip`
- **7.5 phases:** `prep` `soak` `grind` `ferment` `temper` `cook` `assemble` `finish` `rest` `serve`
- **7.6 license classes → usable_for:** `public_domain`/`permissive` → `[training, grounding]`;
  `share_alike` → `[training, grounding]` (with attribution obligations tracked);
  `noncommercial` → `[grounding, reference_only]`; `unknown`/`all_rights_reserved` →
  `[reference_only]` (facts may be re-derived from a permissive source before use).

---

## 8. Referential-integrity contract (enforced by the validator)

1. Every `*_id` / `*_ids` value resolves to an existing record whose `status != "merged"`
   (or, if merged, is auto-rewritten to `merged_into` and flagged).
2. `recipe.dish_id` → a `dish`; `dish.variant_of` → a `dish` or null.
3. Every `rel.subject_id` / `rel.object_id` type is legal for that predicate (predicate→type
   matrix in `configs/canonical_enums.yaml`).
4. Recipe DAG: acyclic; states conserved (see §4.11 DAG rules); every `technique_id` /
   `equipment_id` used in a step also appears in the recipe's `equipment_ids` /
   (technique reachable).
5. No `dish` or `recipe` appears in both a training-eligible source and `data/eval/**`
   (entity-disjoint split, checked in Phase 12).
6. Every non-`hand_authored` record has ≥1 `source_ids`; every `synthetic` record has ≥1
   `seed_record_ids` and every atomic fact traces to a seed.

---

## 9. Minimal seed set (Phase 6/7 target before first 15M PoC run)

To exercise the pipeline the store needs, at minimum:
- 15 `sources`, 60 `ingredients`, 40 `dishes`, 50 `recipes` (≥1 per dish, some with 2 variants),
  25 `techniques`, 15 `equipment`, 8 `regions`, 40 `states`, 25 `outcomes`, 10 `constraints`,
  20 `problems`, 120 `terminology`, ~400 `relationships`.
This is the "toolchain-correctness" bar, not the release bar (§12 of the baseline).

---

## 10. Versioning of the schema itself

`schema_version` is `"MAJOR.MINOR"` as a string, starting `"1"` (== `"1.0"`).
- MINOR bump: added optional field, added enum value, relaxed a constraint. Old records stay valid.
- MAJOR bump: removed/renamed field, tightened a constraint, changed an ID scheme. Requires a
  migration script under `scripts/migrations/` and a `transformations` entry on every touched record.
The `schema/models.py` Pydantic models are the executable form of this document and must move together with it.
