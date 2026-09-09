"""Walk Wikipedia food categories and fetch article extracts for the 150M domain corpus
(plan Level 1, method B1). CC-BY-SA 4.0. No GPU.

    python scripts/fetch_wikipedia_food.py --depth 2 --full

Seeds cover South Indian cuisine specifically plus global food vocabulary. Writes
data/interim/domain_150m/wiki/{core_si,general_food}.txt (forbidden-scanned, deduped)
+ a manifest with per-title provenance.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.langid import ensure_model, is_english  # noqa: E402
from food_slm.data.recipe_sources import _SI_DISHES, _SI_INGREDIENTS  # noqa: E402
from food_slm.data.wiki import UA, fetch_extracts, walk_categories  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

SI_SEEDS = [
    "South Indian cuisine", "Tamil cuisine", "Cuisine of Kerala", "Cuisine of Karnataka",
    "Andhra cuisine", "Telugu cuisine", "Cuisine of Tamil Nadu", "Hyderabadi cuisine",
    "Udupi cuisine", "Chettinad cuisine", "Mangalorean cuisine", "Kodava cuisine",
    "Indian rice dishes", "Dosa", "Indian breads", "Indian snack foods",
    "Fermented foods", "Lentil dishes", "South Indian snacks",
]
GENERAL_FOOD_SEEDS = [
    "Cuisine of India", "Indian cuisine", "Spices", "Herbs", "Cooking techniques",
    "Staple foods", "Edible legumes", "Rice", "Culinary fruits", "Leaf vegetables",
    "Condiments", "Curries", "Stews", "Street food",
]
EXCLUDE = ["restaurant", "chef", "television", "company", "brand", "list of", "wikipedia",
           "chain", "manufacturer", "controvers", "outbreak", "by country"]


def _tier(title: str, text: str) -> str:
    hay = (title + " " + text[:600]).lower()
    if _SI_DISHES.search(hay) or sum(1 for s in _SI_INGREDIENTS if s in hay) >= 3 or \
            re.search(r"south india|tamil nadu|kerala|karnataka|andhra|telangana|"
                      r"chettinad|udupi|malabar", hay):
        return "core_si"
    return "general_food"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "wiki")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--per-cat", type=int, default=600)
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--full", action="store_true", help="also fetch full articles (slow)")
    args = ap.parse_args()

    import httpx

    ensure_model()
    forbidden = load_forbidden(REPO / "data" / "eval")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "_cache"

    with httpx.Client(headers={"User-Agent": UA}, timeout=30.0) as client:
        print("walking SI categories...")
        si_titles = walk_categories(client, SI_SEEDS, max_depth=args.depth,
                                    per_cat_limit=args.per_cat, exclude_substrings=EXCLUDE)
        print(f"  {len(si_titles)} SI articles")
        print("walking general-food categories...")
        gen_titles = walk_categories(client, GENERAL_FOOD_SEEDS, max_depth=args.depth,
                                     per_cat_limit=args.per_cat, exclude_substrings=EXCLUDE)
        gen_titles -= si_titles
        print(f"  {len(gen_titles)} general-food articles")

        titles = sorted(si_titles | gen_titles)
        print(f"fetching {len(titles)} extracts...")
        texts = fetch_extracts(client, titles, cache, intro_only=True)
        if args.full:
            long = [t for t, x in texts.items() if len(x) >= args.min_chars]
            texts |= fetch_extracts(client, long, cache, intro_only=False)

    seen: set[str] = set()
    stats = {"core_si": [], "general_food": []}
    writers = {t: (args.out / f"{t}.txt").open("w", encoding="utf-8") for t in stats}
    for w in writers.values():
        w.write("# --- Wikipedia food articles (CC-BY-SA 4.0) ---\n")
    skipped = {"short": 0, "non_en": 0, "forbidden": 0, "dup": 0}
    for title in titles:
        text = texts.get(title, "").strip()
        if len(text) < args.min_chars:
            skipped["short"] += 1
            continue
        key = " ".join(text.split())[:300].lower()
        if key in seen:
            skipped["dup"] += 1
            continue
        seen.add(key)
        if forbidden.scan_text(text):
            skipped["forbidden"] += 1
            continue
        if not is_english(text):
            skipped["non_en"] += 1
            continue
        tier = _tier(title, text)
        writers[tier].write(text + "\n\n")
        stats[tier].append({"title": title, "chars": len(text)})
    for w in writers.values():
        w.close()

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "en.wikipedia.org category walk + extracts", "license": "CC-BY-SA-4.0",
        "depth": args.depth, "full": args.full,
        "kept": {t: len(v) for t, v in stats.items()},
        "chars": {t: sum(a["chars"] for a in v) for t, v in stats.items()},
        "skipped": skipped,
        "titles": {t: [a["title"] for a in v] for t, v in stats.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                            + "\n", encoding="utf-8")
    for t, v in stats.items():
        print(f"  {t:<13} {len(v):>5} articles  ~{sum(a['chars'] for a in v) // 4:,} tok")
    print(f"  skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
