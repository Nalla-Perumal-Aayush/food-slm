"""Stream the English Wikipedia dump and keep food / cuisine articles for the 150M domain
corpus (plan Level 1, method B1). CC-BY-SA 4.0. No API rate limits, no GPU.

    python scripts/fetch_wikipedia_dump.py

Streams wikimedia/wikipedia (20231101.en), keeps an article if its title or lead matches
a food signature, tiers it core_si / general_food, forbidden-scans and dedups, and writes
data/interim/domain_150m/wiki/{core_si,general_food}.txt + a manifest.
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
from food_slm.data.recipe_sources import _SI_DISHES, _SI_INGREDIENTS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

_TITLE_FOOD = re.compile(
    r"\b(cuisine|dish|curry|bread|rice|chutney|pickle|spice|masala|biryani|pulao|"
    r"dosa|idli|sambar|rasam|dal|dhal|paratha|roti|naan|halwa|payasam|kheer|ladoo|"
    r"soup|stew|salad|dessert|snack|pancake|dumpling|fritter|porridge|noodle|"
    r"pilaf|kebab|tikka|korma|vindaloo|thali|sadya)\b", re.I)
_LEAD_FOOD = re.compile(
    r"\bis (a|an) (indian |south indian |tamil |kerala |karnataka |andhra |telugu |"
    r"fermented |steamed |deep-?fried |savou?ry |sweet |spicy |traditional |popular |"
    r"type of |variety of |kind of |staple )?"
    r"(dish|food|curry|bread|snack|dessert|beverage|drink|preparation|delicacy|"
    r"cuisine|stew|soup|condiment|spice|spice blend|flatbread|pancake|dumpling|"
    r"rice dish|lentil|legume|vegetable|fruit|grain|millet|pulse)\b", re.I)
_SI_REGION = re.compile(
    r"south india|tamil nadu|\bkerala\b|karnataka|andhra pradesh|telangana|"
    r"chettinad|udupi|mangalore|malabar|coorg|hyderabad|dravidian", re.I)
# cheap prefilter: a handful of common food words; only articles passing this run the
# full (expensive) signature checks. Cuts ~99% of Wikipedia before the regex work.
_FOOD_HINT = re.compile(
    r"\b(food|dish|cuisine|recipe|cook|curry|rice|bread|spice|meal|eaten|"
    r"ingredient|flavou?r|sauce|soup|dessert|snack|fried|baked|boiled|"
    r"vegetable|fruit|meat|sweet|drink|beverage)\b", re.I)


def _is_food(title: str, lead_lc: str) -> bool:
    return bool(_TITLE_FOOD.search(title) or _LEAD_FOOD.search(lead_lc)
               or _SI_DISHES.search(title.lower() + " " + lead_lc[:300])
               or sum(1 for s in _SI_INGREDIENTS if s in lead_lc) >= 3)


def _tier(title: str, lead_lc: str) -> str:
    hay = title.lower() + " " + lead_lc[:800]
    if _SI_REGION.search(hay) or _SI_DISHES.search(hay) or \
            sum(1 for s in _SI_INGREDIENTS if s in hay) >= 3:
        return "core_si"
    return "general_food"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "wiki")
    ap.add_argument("--max-chars-per-article", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N scanned (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    forbidden = load_forbidden(REPO / "data" / "eval")
    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    writers = {t: (args.out / f"{t}.txt").open("w", encoding="utf-8")
               for t in ("core_si", "general_food")}
    for w in writers.values():
        w.write("# --- Wikipedia food articles (CC-BY-SA 4.0, dump 20231101.en) ---\n")

    scanned = kept = forbid = 0
    stats = {"core_si": {"n": 0, "chars": 0},
             "general_food": {"n": 0, "chars": 0}}
    seen_titles: set[str] = set()
    for row in ds:
        scanned += 1
        if args.limit and scanned > args.limit:
            break
        if scanned % 200_000 == 0:
            print(f"  scanned {scanned:,}  kept {kept:,}  "
                  f"(si {stats['core_si']['n']:,} / gen {stats['general_food']['n']:,})")
        title = (row.get("title") or "").strip()
        text = row.get("text") or ""
        if not title or len(text) < 400 or title in seen_titles:
            continue
        lead_lc = text[:1500].lower()
        if not _FOOD_HINT.search(title.lower()) and not _FOOD_HINT.search(lead_lc):
            continue
        if not _is_food(title, lead_lc):
            continue
        seen_titles.add(title)
        text = text.strip()[:args.max_chars_per_article]
        if forbidden.scan_text(text[:4000]):
            forbid += 1
            continue
        tier = _tier(title, lead_lc)
        writers[tier].write(f"{title}\n{text}\n\n")
        kept += 1
        stats[tier]["n"] += 1
        stats[tier]["chars"] += len(text)

    for w in writers.values():
        w.close()
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "wikimedia/wikipedia 20231101.en (dump stream)", "license": "CC-BY-SA-4.0",
        "scanned": scanned, "kept": kept, "forbidden_hits": forbid,
        "tiers": {t: {"articles": s["n"], "est_tokens": s["chars"] // 4}
                  for t, s in stats.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    for t, s in stats.items():
        print(f"  {t:<13} {s['n']:>6} articles  ~{s['chars'] // 4:,} tok")
    print(f"scanned {scanned:,}, forbidden {forbid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
