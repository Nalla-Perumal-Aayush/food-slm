# data/eval/

Held-out evaluation material. **Versioned.** **Never contains synthetic data.**
Built *before* the training corpus is frozen (baseline §9.2).

| File | What |
|---|---|
| `forbidden_entities.jsonl` | The frozen forbidden-entity list — culinary entities that must **never** appear in any training data (canonical store, corpus, synthetic, SFT). Reserved for hallucination / abstention probes. |
| `forbidden_entities.lock.json` | SHA-256 pin + amendment history for the list. `food_slm.data.forbidden.verify_lock` is a hard gate before any training run. |
| `forbidden_entities.amendments.log` | Append-only log of every (re)freeze, with rationale. |

## Forbidden-entity list

Two kinds:

- **`withheld_real`** — genuine Tamil Nadu vegetarian dishes/ingredients deliberately
  excluded from collection, to measure whether the model *abstains* ("I don't have a
  reliable recipe for that") instead of confabulating.
- **`fictitious`** — plausible-sounding invented dishes / ingredients / techniques /
  claims / pairings, to catch confident fabrication.

### Rules

1. The list is **frozen before Phase 4 (data collection)**. Freeze with
   `python scripts/freeze_forbidden.py --note "..."`.
2. Every entity here is **banned everywhere in training data**. Enforced by:
   - `scripts/validate_canonical.py` — checks the canonical store on every run.
   - the Phase 11 n-gram contamination gate — checks the assembled corpus + synthetic output.
3. Changing the list after freeze requires re-running `freeze_forbidden.py --note "..."`;
   the reason is logged and the lock's `amendments` array grows. Tooling refuses to train
   if the file no longer matches the lock and the change was not logged.
4. Keep surface forms broad — include common misspellings and romanization variants so a
   leak is caught however it is written. Add `tamil_forms` where a native-script spelling
   might appear in a scraped source.

### Probe usage (later, during the evaluation phase)

For each entity, the eval harness asks the model about it directly and scores:
`withheld_real` → did it abstain / express uncertainty? `fictitious` → did it refuse to
invent, or did it fabricate a confident recipe/claim? Both feed the hallucination-rate
and abstention-correctness metrics (baseline §9).
