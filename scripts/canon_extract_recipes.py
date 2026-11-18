"""Recipe extraction pass into the canonical store (150M dataset plan).

    python scripts/canon_extract_recipes.py                       # all matched docs
    python scripts/canon_extract_recipes.py --limit 50              # bound one run
    python scripts/canon_extract_recipes.py --max-minutes 240       # bound by wall clock
    python scripts/canon_extract_recipes.py --only-region kerala
    OLLAMA_NUM_GPU=0 python scripts/canon_extract_recipes.py        # CPU-only, safe
                                                                     # alongside GPU training

Same document set and dish-matching as canon_extract_dishes.py (Phase 2), but runs
each matched doc through `food_slm.data.pipeline.process_cached_text_recipe`
instead of the flat dish/ingredient/technique/term extractor -- producing full
`recipe` records (step DAG, typed ingredient quantities, diet profile taken from
the dish list, not re-derived by the LLM) rather than loose entity mentions.

`recipe` is the single largest remaining canonical-store gap (target ~5,000): this
pass is what fills it.

Resumable: `_recipe_slug_done()` checks staged `recipe:{dish-slug}-{source-slug}*`
ids (the exact id shape `extract_recipe._build_recipe` generates) rather than
canon_extract_dishes.py's `already_staged()`, which checks *source*-level staging
left behind by the earlier flat pass -- that would incorrectly mark every doc as
"already done" for recipe purposes, since Phase 2 already staged a `source:` record
for each one. Progress snapshot every 10 docs, same shape as canon_extract_dishes.py.

Runs on local Ollama only (qwen2.5:7b + gemma2:9b), no paid API, no network beyond
localhost.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Real failure (took the whole multi-day run down): stdout redirected to a log file on
# Windows defaults to the system codepage (cp1252), not UTF-8 -- a dish/source title
# containing an unencodable character (native-script text, a mangled scrape artifact)
# crashes every print() with UnicodeEncodeError and kills the process outright, not
# just that one doc. reconfigure() (Python 3.7+) forces UTF-8 regardless of how stdout
# is attached; errors="replace" means a genuinely bad byte degrades to a substitution
# character in the log instead of taking the process down.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon_extract_dishes import (  # noqa: E402
    _interleave_by_dish, collect_docs, load_dish_index, load_source_blocklist,
)
from food_slm.data.extract import slugify  # noqa: E402
from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.llm import Panel, build_provider, load_agents_config  # noqa: E402
from food_slm.data.pipeline import process_cached_text_recipe  # noqa: E402
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
STAGING = REPO / "data" / "interim" / "canon_staging"
EVAL = REPO / "data" / "eval"
AGENTS = REPO / "configs" / "agents.yaml"
PROGRESS = REPO / "artifacts" / "logs" / "canon_extract_recipes_progress.json"


def _recipe_ids_done() -> set[str]:
    """Every staged `recipe` record's filename stem -- `stage_records()` writes each
    one to `{staging_dir}/recipe/{id-without-'recipe:'-prefix}.json`, and
    `extract_recipe._build_recipe` builds that id-suffix as
    `{dish-slug}[-{variant-slug}]-{source-slug}`. We only need the (dish, source)
    pair to know a doc's recipe attempt is done, so match on prefix/suffix below."""
    d = STAGING / "recipe"
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.json")}


def _recipe_done_for(dish_id: str, source_id: str, done: set[str]) -> bool:
    dish_slug = dish_id.split(":", 1)[1]
    source_slug = source_id.split(":", 1)[1] if ":" in source_id else slugify(source_id)
    suffix = f"-{source_slug}"
    return any(stem.startswith(dish_slug) and stem.endswith(suffix) for stem in done)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dish-list", type=Path, default=REPO / "configs" / "dish_list_v1.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max docs this run (0 = no limit)")
    ap.add_argument("--max-minutes", type=float, default=0, help="wall-clock budget (0 = none)")
    ap.add_argument("--only-region", default=None)
    ap.add_argument("--max-docs-per-dish", type=int, default=5,
                    help="cap docs per dish name so a generic-name dish (Coconut "
                         "Rice, Tomato Chutney...) with many matched/false-positive "
                         "docs can't dominate a run's whole doc budget before other "
                         "dishes get any coverage; extra docs are deferred to a "
                         "later run, not dropped -- same as canon_extract_dishes.py")
    args = ap.parse_args()

    dish_index = load_dish_index(args.dish_list)
    print(f"dish index: {len(dish_index)} name/alias entries")
    docs = collect_docs(dish_index)
    if args.only_region:
        docs = [d for d in docs if d["dish"]["region"] == args.only_region]
    docs.sort(key=lambda d: (-d["dish"].get("prominence", 1), d["kind"] != "wikipedia"))
    blocklist = load_source_blocklist()
    done = _recipe_ids_done()
    todo = [d for d in docs
            if d["source_id"] not in blocklist
            and not _recipe_done_for(d["dish"]["id"], d["source_id"], done)]
    print(f"{len(docs)} dish-matched docs total, {len(docs) - len(todo)} already have a "
         f"staged recipe attempt or are blocklisted, {len(todo)} to process")
    todo = _interleave_by_dish(todo, args.max_docs_per_dish)

    roles = load_agents_config(AGENTS)
    extract_p = build_provider(roles["extraction_local"][0])
    panel = Panel([build_provider(s) for s in roles["data_validation"]])
    heavy = build_provider(roles["data_validation_heavy"][0])
    forbidden = load_forbidden(EVAL)
    store = load_store(CANON, strict=False)

    stats = {"docs_done": 0, "docs_no_recipe": 0, "docs_invalid": 0, "docs_error": 0,
            "staged": 0, "by_decision": {}, "dishes_touched": set()}
    t0 = time.time()
    consecutive_errors = 0
    for i, doc in enumerate(todo):
        if args.limit and stats["docs_done"] >= args.limit:
            print(f"--limit {args.limit} reached")
            break
        if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
            print(f"--max-minutes {args.max_minutes} reached")
            break
        dish = doc["dish"]
        print(f"[{i + 1}/{len(todo)}] {dish['canonical_name']} ({dish['region']})  "
             f"<- {doc['title'][:60]!r}")
        try:
            run = process_cached_text_recipe(
                source_id=doc["source_id"], source_url=doc["url"], kind=doc["kind"],
                text=doc["text"], license_class=doc["license_class"], title=doc["title"],
                dish_id=dish["id"], dish_name=dish["canonical_name"],
                cuisine_state=dish["region"], dish_diet=dish.get("diet") or {},
                extract_provider=extract_p, panel=panel, staging_dir=STAGING, store=store,
                forbidden=forbidden, heavy_provider=heavy)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR {type(exc).__name__}: {str(exc)[:200]}")
            stats["docs_error"] += 1
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print(f"    {consecutive_errors} consecutive failures -- something is "
                     f"broken (Ollama down/hung?), not just a slow document. Stopping "
                     f"rather than fail through the rest of the backlog.")
                _write_progress(stats, stats["docs_done"], len(todo), t0, final=True)
                return 1
            continue
        consecutive_errors = 0
        stats["docs_done"] += 1
        n_extracted = len(run.extraction.drafts) if run.extraction else 0
        n_invalid = len(run.extraction.invalid) if run.extraction else 0
        if n_extracted == 0:
            stats["docs_no_recipe"] += 1
        if n_invalid:
            stats["docs_invalid"] += 1
        stats["staged"] += len(run.staged)
        stats["dishes_touched"].add(dish["canonical_name"])
        for v in run.verdicts:
            stats["by_decision"][v.decision] = stats["by_decision"].get(v.decision, 0) + 1
        print(f"    extracted={n_extracted} invalid={n_invalid} staged={len(run.staged)}")

        if (i + 1) % 10 == 0:
            _write_progress(stats, i + 1, len(todo), t0)

    _write_progress(stats, stats["docs_done"], len(todo), t0, final=True)
    print(f"\n=== done: {stats['docs_done']} docs, {stats['staged']} records staged, "
         f"{len(stats['dishes_touched'])} distinct dishes touched ===")
    print(f"docs_no_recipe={stats['docs_no_recipe']} docs_invalid={stats['docs_invalid']} "
         f"docs_error={stats['docs_error']}")
    print(f"by_decision: {stats['by_decision']}")
    print("next: python scripts/canon_run.py promote-ready --reviewer <name>")
    return 0


def _write_progress(stats: dict, done: int, total: int, t0: float, *, final: bool = False) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    PROGRESS.write_text(json.dumps({
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "final": final, "docs_done": done, "docs_total_this_run": total,
        "elapsed_s": round(elapsed, 1),
        "docs_per_min": round(done / (elapsed / 60), 2) if elapsed > 0 else 0,
        "staged": stats["staged"], "docs_error": stats["docs_error"],
        "docs_no_recipe": stats["docs_no_recipe"], "docs_invalid": stats["docs_invalid"],
        "by_decision": stats["by_decision"], "dishes_touched": sorted(stats["dishes_touched"]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
