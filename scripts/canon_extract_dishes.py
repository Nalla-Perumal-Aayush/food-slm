"""Phase 2: per-dish targeted extraction into the canonical store (150M dataset plan).

    python scripts/canon_extract_dishes.py                      # all matched docs, prominence-first
    python scripts/canon_extract_dishes.py --limit 50            # bound one run
    python scripts/canon_extract_dishes.py --max-minutes 240     # bound by wall clock instead
    python scripts/canon_extract_dishes.py --only-region kerala

Walks the already-collected raw text (blogs, Wikipedia, recipe datasets), matches each
document to a dish in `configs/dish_list_v1.yaml`, and — for matched docs, highest
prominence first — runs it through the real extract -> validate -> stage chain
(`food_slm.data.pipeline.process_cached_text`): local qwen7b extracts structured
records, the qwen7b+gemma9b panel checks each one against the source text, records are
staged to `data/interim/canon_staging/`. A record whose id recurs across multiple
sources (e.g. a dish covered by both a blog post and a Wikipedia article) has its
`provenance.source_ids` merged and `corroboration_count` grows accordingly — that
cross-source agreement is what later lets `canon_run.py promote-ready` move it into
`data/canonical/` without needing a fresh LLM vote on the merge.

Unmatched docs (no dish in the source text) are skipped — this pass is deliberately
dish-driven, not a bulk scrape-everything pass; see plan_40m_ablation.md / Rev 9 notes
on why the automated-harvest-first approach was abandoned.

Resumable: a doc whose synthesized source_id is already staged is skipped, so a killed
run picks back up where it left off. Progress snapshot written every 10 docs.

Runs on local Ollama only (qwen2.5:7b + gemma2:9b), no paid API, no network beyond
localhost. Meant to run for hours in the background while the GPU is otherwise idle.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.dish_list import normalise_name  # noqa: E402
from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.llm import Panel, build_provider, load_agents_config  # noqa: E402
from food_slm.data.pipeline import process_cached_text  # noqa: E402
from food_slm.data.recipe_transform import to_prose  # noqa: E402
from food_slm.schema import load_store  # noqa: E402
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
STAGING = REPO / "data" / "interim" / "canon_staging"
EVAL = REPO / "data" / "eval"
AGENTS = REPO / "configs" / "agents.yaml"
BLOGS = REPO / "data" / "interim" / "domain_150m" / "blogs"
WIKI = REPO / "data" / "interim" / "domain_150m" / "wiki"
RECIPES = REPO / "data" / "interim" / "recipes_150m"
PROGRESS = REPO / "artifacts" / "logs" / "canon_extract_progress.json"

_SLUG = re.compile(r"[^a-z0-9]+")

# Generic English food words that should never be used alone as a dish-match key,
# even if they slipped in as a single-word alias somewhere in data/dish_seed/*.yaml
# (e.g. "Poha", "Dal") — they'd match thousands of unrelated non-SI blog/recipe
# titles. Multi-word names/aliases are never affected by this list.
_GENERIC_SINGLE_WORD = {
    "peanut", "peanuts", "dal", "poha", "rice", "coconut", "milk", "sugar", "egg",
    "eggs", "bread", "tea", "coffee", "fish", "chicken", "mutton", "beef", "pork",
    "oil", "ghee", "curry", "curries", "chutney", "pickle", "salad", "cake", "soup",
    "kheer", "malai", "roti", "flour", "onion", "tomato", "potato", "cream", "sauce",
    "gravy", "masala", "spice", "spices", "juice", "cookies", "biscuit", "biscuits",
}


def _slug(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")[:60] or "doc"


# --- dish index --------------------------------------------------------------

def load_dish_index(path: Path) -> list[tuple[str, re.Pattern, dict]]:
    doc = load_yaml(path)
    entries = []
    for d in doc["dishes"]:
        names = [d["canonical_name"], *(d.get("also_known_as") or [])]
        for n in names:
            norm = normalise_name(n)
            if len(norm) < 3:
                continue
            if " " not in norm and norm in _GENERIC_SINGLE_WORD:
                continue
            entries.append((norm, d))
    entries.sort(key=lambda e: -len(e[0]))     # longest / most specific match wins
    return [(norm, re.compile(rf"(?<!\w){re.escape(norm)}(?!\w)"), d) for norm, d in entries]


def match_dish(title: str, index: list[tuple[str, re.Pattern, dict]]) -> dict | None:
    norm = normalise_name(title)
    if not norm:
        return None
    for _, pat, dish in index:
        if pat.search(norm):
            return dish
    return None


# --- document sources ---------------------------------------------------------

def _iter_txt_docs(path: Path):
    """Blank-line-separated (title, body) blocks, matching how fetch_food_blogs.py /
    fetch_wikipedia_dump.py write their .txt output."""
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
        if line.startswith("# ---"):
            continue
        if line.strip() == "":
            if len(buf) >= 2:
                yield buf[0].strip(), "\n".join(buf[1:]).strip()
            buf = []
        else:
            buf.append(line)
    if len(buf) >= 2:
        yield buf[0].strip(), "\n".join(buf[1:]).strip()


def collect_docs(dish_index) -> list[dict]:
    docs = []

    for p in sorted(BLOGS.glob("*.txt")):
        site = p.stem
        for title, body in _iter_txt_docs(p):
            if len(body) < 300:
                continue
            dish = match_dish(title, dish_index)
            if not dish:
                continue
            docs.append({"source_id": f"src:{site}-{_slug(title)}",
                        "url": f"https://{site.replace('_', '')}.example/#{_slug(title)}",
                        "kind": "recipe_blog", "license_class": "unknown",
                        "title": title, "text": f"{title}\n{body}", "dish": dish})

    wiki_core = WIKI / "core_si.txt"
    if wiki_core.exists():
        for title, body in _iter_txt_docs(wiki_core):
            if len(body) < 200:
                continue
            dish = match_dish(title, dish_index)
            if not dish:
                continue
            url = f"https://en.wikipedia.org/wiki/{title.strip().replace(' ', '_')}"
            docs.append({"source_id": f"src:en-wikipedia-org-{_slug(title)}", "url": url,
                        "kind": "wikipedia", "license_class": "share_alike",
                        "title": title, "text": f"{title}\n{body}", "dish": dish})

    for name in ("core_si.jsonl", "pan_indian.jsonl"):
        p = RECIPES / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                dish = match_dish(row.get("title", ""), dish_index)
                if not dish:
                    continue
                sid = f"src:{row.get('source', 'recipe')}-{_slug(row.get('sid') or row['title'])}"
                docs.append({"source_id": sid, "url": row.get("url") or sid,
                            "kind": "recipe_blog", "license_class": "unknown",
                            "title": row["title"], "text": to_prose(row), "dish": dish})
    return docs


def _interleave_by_dish(docs: list[dict], cap: int) -> list[dict]:
    """Round-robin across dishes (highest prominence first), capping each dish at
    `cap` docs this run, so a handful of generic-name dishes (Dosa, Poriyal, Rasam)
    can't consume the whole doc budget before the long tail gets a single doc.
    Docs beyond the cap are simply dropped from *this run's* list — they're still
    on disk and still match, so a later run (already_staged filters what's done by
    then) will pick them up once the long tail has some coverage too."""
    from collections import defaultdict

    by_dish: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_dish[d["dish"]["canonical_name"]].append(d)
    order = sorted(by_dish, key=lambda name: -by_dish[name][0]["dish"].get("prominence", 1))
    capped = {name: lst[:cap] for name, lst in by_dish.items()}
    out: list[dict] = []
    i = 0
    while True:
        added = False
        for name in order:
            lst = capped[name]
            if i < len(lst):
                out.append(lst[i])
                added = True
        if not added:
            break
        i += 1
    return out


def already_staged(source_id: str) -> bool:
    slug = source_id.split(":", 1)[1]
    return (STAGING / "source" / f"{slug}.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dish-list", type=Path, default=REPO / "configs" / "dish_list_v1.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max docs this run (0 = no limit)")
    ap.add_argument("--max-minutes", type=float, default=0, help="wall-clock budget (0 = none)")
    ap.add_argument("--only-region", default=None)
    ap.add_argument("--max-docs-per-dish", type=int, default=5,
                    help="cap corroborating docs per dish so a few generic-name "
                         "dishes (Dosa, Poriyal, Rasam...) don't starve the long "
                         "tail's doc budget within one run; extra docs are simply "
                         "deferred to a later run, not dropped")
    args = ap.parse_args()

    dish_index = load_dish_index(args.dish_list)
    print(f"dish index: {len(dish_index)} name/alias entries")
    docs = collect_docs(dish_index)
    if args.only_region:
        docs = [d for d in docs if d["dish"]["region"] == args.only_region]
    docs.sort(key=lambda d: (-d["dish"].get("prominence", 1), d["kind"] != "wikipedia"))
    todo = [d for d in docs if not already_staged(d["source_id"])]
    print(f"{len(docs)} dish-matched docs total, {len(docs) - len(todo)} already staged, "
         f"{len(todo)} to process")
    todo = _interleave_by_dish(todo, args.max_docs_per_dish)

    roles = load_agents_config(AGENTS)
    extract_p = build_provider(roles["extraction_local"][0])
    panel = Panel([build_provider(s) for s in roles["data_validation"]])
    heavy = build_provider(roles["data_validation_heavy"][0])
    forbidden = load_forbidden(EVAL)
    store = load_store(CANON, strict=False)

    stats = {"docs_done": 0, "docs_skipped_empty": 0, "docs_error": 0, "staged": 0,
            "by_type": {}, "by_decision": {}, "dishes_touched": set()}
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
        print(f"[{i + 1}/{len(todo)}] {dish['canonical_name']} ({dish['region']}, "
             f"prom={dish.get('prominence')})  <- {doc['title'][:60]!r}")
        try:
            run = process_cached_text(
                source_id=doc["source_id"], source_url=doc["url"], kind=doc["kind"],
                text=doc["text"], license_class=doc["license_class"], title=doc["title"],
                extract_provider=extract_p, panel=panel, staging_dir=STAGING, store=store,
                forbidden=forbidden, heavy_provider=heavy)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR {type(exc).__name__}: {str(exc)[:200]}")
            stats["docs_error"] += 1
            consecutive_errors += 1
            if consecutive_errors >= 5:
                # Isolated timeouts (one slow doc) are normal and handled; 5 in a row
                # means something is actually broken (Ollama hung/down, not just one
                # big document) — burning through the rest of the backlog at this
                # point would just fail every doc near-instantly and *look* alive
                # (the progress file would keep updating) while producing nothing.
                # Exit non-zero so the file goes stale and the watchdog's restart
                # (a fresh process, fresh Ollama connection) actually fixes it,
                # instead of masking the outage.
                print(f"    {consecutive_errors} consecutive failures — something is "
                     f"broken (Ollama down/hung?), not just a slow document. Stopping "
                     f"rather than fail through the rest of the backlog.")
                _write_progress(stats, stats["docs_done"], len(todo), t0, final=True)
                return 1
            continue
        consecutive_errors = 0
        s = run.summary()
        if s["extracted"] == 0:
            stats["docs_skipped_empty"] += 1
        stats["docs_done"] += 1
        stats["staged"] += len(run.staged)
        stats["dishes_touched"].add(dish["canonical_name"])
        for v in run.verdicts:
            stats["by_decision"][v.decision] = stats["by_decision"].get(v.decision, 0) + 1
        for rid in run.staged:
            t = rid.split(":", 1)[0]
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
        print(f"    extracted={s['extracted']} invalid={s['invalid']} "
             f"staged={s['staged']} verdicts={s['verdicts']}")

        if (i + 1) % 10 == 0:
            _write_progress(stats, i + 1, len(todo), t0)

    _write_progress(stats, stats["docs_done"], len(todo), t0, final=True)
    print(f"\n=== done: {stats['docs_done']} docs, {stats['staged']} records staged, "
         f"{len(stats['dishes_touched'])} distinct dishes touched ===")
    print(f"by_type: {stats['by_type']}")
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
        "docs_skipped_empty": stats["docs_skipped_empty"],
        "by_type": stats["by_type"], "by_decision": stats["by_decision"],
        "dishes_touched": sorted(stats["dishes_touched"]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
