"""Recipe generation for source-gap dishes -- canon_extract_recipes.py's counterpart
for the ~half of dishes with zero matched source documents.

    python scripts/canon_generate_recipes.py                    # all gap dishes
    python scripts/canon_generate_recipes.py --limit 20
    python scripts/canon_generate_recipes.py --only-region kerala

Uses openai/gpt-oss-120b via Groq's free tier (agents.yaml's recipe_generation_remote
role) to generate a recipe from general knowledge for each dish canon_extract_recipes.py
has no document to extract from, then validates it with the SAME local qwen7b+gemma9b
panel (+deepseek-r1 heavy tiebreak) used everywhere else -- via
food_slm.data.pipeline.process_dish_generation, in "generation" mode (a plausibility
check, since there's no source text to check fidelity against). Every result is staged
as `synthetic: true` and can only ever reach `needs_human`, never auto-promote --
corroboration is necessarily 0 with no source_ids, so these always need a human look
before reaching data/canonical/, unlike a well-corroborated extracted recipe.

Needs GROQ_API_KEY in the environment (free signup at console.groq.com, no billing
account). Requests are paced with a small fixed delay and back off on a real 429 from
Groq (reading Retry-After) rather than hand-estimating token budgets; a same-day 429
that persists after backoff means the daily cap (TPD/RPD) is exhausted -- the run stops
cleanly rather than sleeping for hours, and just needs a rerun (tomorrow, or once the
cap resets) to pick up the rest.

Resumable: a dish with an already-staged `recipe:{dish}-{generator-slug}` record is
skipped, same mechanism as canon_extract_recipes.py's own resume check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

# See canon_extract_recipes.py's identical fix: stdout redirected to a log file on
# Windows defaults to cp1252, and a dish/generated-content title with an unencodable
# character crashes every print() and kills the whole process, not just that one dish.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canon_extract_dishes import collect_docs, load_dish_index  # noqa: E402
from canon_extract_recipes import _recipe_done_for, _recipe_ids_done  # noqa: E402
from food_slm.data.extract import slugify  # noqa: E402
from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.llm import Panel, build_provider, load_agents_config  # noqa: E402
from food_slm.data.pipeline import process_dish_generation  # noqa: E402
from food_slm.schema import load_store  # noqa: E402
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
STAGING = REPO / "data" / "interim" / "canon_staging"
EVAL = REPO / "data" / "eval"
AGENTS = REPO / "configs" / "agents.yaml"
PROGRESS = REPO / "artifacts" / "logs" / "canon_generate_recipes_progress.json"

GENERATOR_LABEL = "openai/gpt-oss-120b (Groq)"
# TPM 8000 / ~1.9k tokens-per-call observed -> a real 429 is the authoritative backstop,
# but this fixed spacing keeps normal operation well clear of it without hand-tracking
# token counts across calls.
PACE_SECONDS = 15.0


def find_gap_dishes(dish_list_path: Path, *, only_region: str | None = None) -> list[dict]:
    """Dishes in dish_list_v1.yaml with zero documents matched by canon_extract_dishes's
    own collect_docs() -- i.e. dishes canon_extract_recipes.py can never generate a
    recipe for, no matter how long it runs, because there's nothing to extract from."""
    dish_index = load_dish_index(dish_list_path)
    docs = collect_docs(dish_index)
    matched_ids = {d["dish"]["id"] for d in docs}
    all_dishes = load_yaml(dish_list_path)["dishes"]
    gap = [d for d in all_dishes if d["id"] not in matched_ids]
    if only_region:
        gap = [d for d in gap if d["region"] == only_region]
    gap.sort(key=lambda d: -d.get("prominence", 1))
    return gap


def _complete_with_backoff(provider, system: str, prompt: str, *, max_retries: int = 3):
    """Groq 429s carry a Retry-After header; a real rate-limit signal is more reliable
    than hand-estimating token budgets. Returns (text, daily_cap_hit: bool)."""
    import httpx

    for attempt in range(max_retries):
        try:
            return provider.complete(system, prompt, temperature=0.0, json_out=True), False
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            retry_after = float(exc.response.headers.get("retry-after", 60))
            if retry_after > 300:
                # A short-lived (RPM/TPM) 429 backs off in seconds; a multi-minute
                # Retry-After means the DAILY cap (TPD/RPD) is exhausted -- no point
                # sleeping for hours inside this process.
                return "", True
            print(f"    429 rate-limited, backing off {retry_after:.0f}s "
                 f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_after)
    return "", True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dish-list", type=Path, default=REPO / "configs" / "dish_list_v1.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max dishes this run (0 = no limit)")
    ap.add_argument("--only-region", default=None)
    args = ap.parse_args()

    gap = find_gap_dishes(args.dish_list, only_region=args.only_region)
    done = _recipe_ids_done()
    gen_source_id = f"gen:{slugify(GENERATOR_LABEL)}"
    todo = [d for d in gap if not _recipe_done_for(d["id"], gen_source_id, done)]
    print(f"{len(gap)} source-gap dishes total, {len(gap) - len(todo)} already generated, "
         f"{len(todo)} to process")

    roles = load_agents_config(AGENTS)
    gen_provider = build_provider(roles["recipe_generation_remote"][0])
    panel = Panel([build_provider(s) for s in roles["data_validation"]])
    heavy = build_provider(roles["data_validation_heavy"][0])
    forbidden = load_forbidden(EVAL)
    store = load_store(CANON, strict=False)

    # Groq calls happen inside process_dish_generation via gen_provider.complete();
    # wrap it once so every call gets 429 backoff without threading that logic through
    # process_dish_generation's generic Provider interface.
    def _wrapped_complete(system, prompt, *, temperature=0.0, json_out=False):
        text, capped = _complete_with_backoff(gen_provider, system, prompt)
        if capped:
            raise RuntimeError("DAILY_CAP")
        return text

    provider_proxy = types.SimpleNamespace(complete=_wrapped_complete, spec=gen_provider.spec)

    stats = {"dishes_done": 0, "dishes_no_recipe": 0, "dishes_invalid": 0, "dishes_error": 0,
            "staged": 0, "by_decision": {}, "dishes_touched": []}
    t0 = time.time()
    for i, dish in enumerate(todo):
        if args.limit and stats["dishes_done"] >= args.limit:
            print(f"--limit {args.limit} reached")
            break
        print(f"[{i + 1}/{len(todo)}] {dish['canonical_name']} ({dish['region']}, "
             f"prom={dish.get('prominence')})")
        try:
            run = process_dish_generation(
                dish_id=dish["id"], dish_name=dish["canonical_name"], cuisine_state=dish["region"],
                dish_diet=dish.get("diet") or {}, generator_label=GENERATOR_LABEL,
                generation_provider=provider_proxy, panel=panel, staging_dir=STAGING, store=store,
                forbidden=forbidden, heavy_provider=heavy)
        except RuntimeError as exc:
            if str(exc) == "DAILY_CAP":
                print("    Groq daily cap (TPD/RPD) hit -- stopping cleanly. "
                     "Rerun this script once the cap resets to pick up the rest.")
                _write_progress(stats, stats["dishes_done"], len(todo), t0, final=True)
                # Distinct from 0 (fully done) and 1 (crashed) -- a bare retry-on-
                # nonzero wrapper would otherwise either treat this as "finished"
                # (if it only checks ==0) or burn through retries uselessly before
                # the cap resets tomorrow (if it treats every nonzero the same as
                # a crash). Confirmed live: exactly this ambiguity left a prior
                # wrapper run idle for a day after a cap-hit it misread as done.
                return 2
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR {type(exc).__name__}: {str(exc)[:200]}")
            stats["dishes_error"] += 1
            continue

        stats["dishes_done"] += 1
        n_extracted = len(run.extraction.drafts) if run.extraction else 0
        n_invalid = len(run.extraction.invalid) if run.extraction else 0
        if n_extracted == 0:
            stats["dishes_no_recipe"] += 1
        if n_invalid:
            stats["dishes_invalid"] += 1
        stats["staged"] += len(run.staged)
        stats["dishes_touched"].append(dish["canonical_name"])
        for v in run.verdicts:
            stats["by_decision"][v.decision] = stats["by_decision"].get(v.decision, 0) + 1
        print(f"    extracted={n_extracted} invalid={n_invalid} staged={len(run.staged)}")

        if (i + 1) % 10 == 0:
            _write_progress(stats, i + 1, len(todo), t0)
        time.sleep(PACE_SECONDS)

    _write_progress(stats, stats["dishes_done"], len(todo), t0, final=True)
    print(f"\n=== done: {stats['dishes_done']} dishes, {stats['staged']} records staged ===")
    print(f"by_decision: {stats['by_decision']}  (generated recipes always need_human -- "
         "never auto-promote -- since corroboration is necessarily 0)")
    print("next: review with the Recipe Extraction dashboard, then "
         "python scripts/canon_run.py promote-ready --reviewer <name>")
    return 0


def _write_progress(stats: dict, done: int, total: int, t0: float, *, final: bool = False) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    PROGRESS.write_text(json.dumps({
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "final": final, "dishes_done": done, "dishes_total_this_run": total,
        "elapsed_s": round(elapsed, 1), "staged": stats["staged"],
        "dishes_error": stats["dishes_error"], "dishes_no_recipe": stats["dishes_no_recipe"],
        "dishes_invalid": stats["dishes_invalid"], "by_decision": stats["by_decision"],
        "dishes_touched": stats["dishes_touched"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
