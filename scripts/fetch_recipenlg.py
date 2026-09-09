"""Stream the RecipeNLG corpus (corbt/all-recipes, ~2.2M recipes, pre-formatted prose)
and split by South-Indian relevance for the 150M domain corpus (Level 1, the volume lever).

    python scripts/fetch_recipenlg.py

Rows are already clean text ("Title\\n\\nIngredients:\\n- ...\\n\\nDirections:\\n- ..."), so
no parsing — just tier (title + ingredient block -> relevance_tier), forbidden-scan, dedup,
write data/interim/domain_150m/recipenlg/{core_si,pan_indian,general}.txt + a manifest.

Scraped source: weights only, not redistributed (150M data plan). No GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.recipe_sources import relevance_tier  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _parse(text: str) -> dict | None:
    lines = text.splitlines()
    if not lines:
        return None
    title = lines[0].strip()
    ings, in_ing = [], False
    for ln in lines[1:]:
        low = ln.strip().lower()
        if low.startswith("ingredients"):
            in_ing = True
            continue
        if low.startswith(("directions", "method", "instructions")):
            break
        if in_ing and ln.strip().lstrip("-* ").strip():
            ings.append(ln.strip().lstrip("-* ").strip())
    if not title or len(ings) < 2:
        return None
    return {"title": title, "cuisine_raw": "", "ingredients": ings, "tags": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "recipenlg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-general", type=int, default=1_200_000,
                    help="cap the general tier (RecipeNLG is huge and Western-heavy)")
    args = ap.parse_args()

    from datasets import load_dataset

    forbidden = load_forbidden(REPO / "data" / "eval")
    args.out.mkdir(parents=True, exist_ok=True)
    writers = {t: (args.out / f"{t}.txt").open("w", encoding="utf-8")
               for t in ("core_si", "pan_indian", "general")}
    for w in writers.values():
        w.write("# --- RecipeNLG (scraped; weights only) ---\n")

    ds = load_dataset("corbt/all-recipes", split="train", streaming=True)
    seen: set[str] = set()
    n = forbid = dupes = 0
    kept = {"core_si": 0, "pan_indian": 0, "general": 0}
    chars = {"core_si": 0, "pan_indian": 0, "general": 0}
    regions: dict[str, int] = {}
    for row in ds:
        n += 1
        if args.limit and n > args.limit:
            break
        if n % 200_000 == 0:
            print(f"  scanned {n:,}  kept {sum(kept.values()):,}  {dict(kept)}")
        text = (row.get("input") or "").strip()
        if len(text) < 80:
            continue
        parsed = _parse(text)
        if parsed is None:
            continue
        tier, regs = relevance_tier(parsed)
        if tier == "general" and kept["general"] >= args.max_general:
            continue
        key = hashlib.sha1(" ".join(text.split())[:200].lower().encode()).hexdigest()  # noqa: S324
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        if forbidden.scan_text(text):
            forbid += 1
            continue
        writers[tier].write(text + "\n\n")
        kept[tier] += 1
        chars[tier] += len(text)
        for r in regs:
            regions[r] = regions.get(r, 0) + 1

    for w in writers.values():
        w.close()
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "corbt/all-recipes (RecipeNLG)", "license": "research-only scrape",
        "scanned": n, "duplicates": dupes, "forbidden_hits": forbid,
        "kept": kept, "est_tokens": {t: chars[t] // 4 for t in chars},
        "core_si_regions": dict(sorted(regions.items(), key=lambda x: -x[1])),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    for t in kept:
        print(f"  {t:<11} {kept[t]:>8,} recipes  ~{chars[t] // 4:,} tok")
    print(f"scanned {n:,}, dup {dupes:,}, forbidden {forbid}")
    print(f"core_si regions: {manifest['core_si_regions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
