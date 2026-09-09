"""Pull recipe datasets from HF, normalise, and split by South-Indian relevance tier
(150M dataset plan, Level 1/2).

    python scripts/ingest_recipe_datasets.py --sources food_com archanas epicurious

Writes data/interim/recipes_150m/{core_si,pan_indian,general}.jsonl (common row shape +
tier + region_tags + provenance) and a manifest. Nothing here touches the GPU.

The scraped sources are copyright-encumbered: used to train weights, not redistributed
(see the 150M data plan). Provenance (source, sid, url, license) is on every row.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.recipe_sources import SOURCES, normalise, relevance_tier  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--out", type=Path, default=REPO / "data" / "interim" / "recipes_150m")
    ap.add_argument("--limit", type=int, default=0, help="max rows per source (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    writers = {t: (args.out / f"{t}.jsonl").open("w", encoding="utf-8")
               for t in ("core_si", "pan_indian", "general")}
    stats: dict[str, Counter] = {s: Counter() for s in args.sources}
    region_counts: Counter = Counter()
    t0 = time.time()

    for src in args.sources:
        spec = SOURCES[src]
        print(f"\n=== {src}  ({spec['hf']}) ===")
        ds = load_dataset(spec["hf"], split="train", streaming=spec["streaming"])
        n = 0
        for raw in ds:
            n += 1
            if args.limit and n > args.limit:
                break
            row = normalise(src, raw)
            if row is None:
                stats[src]["unusable"] += 1
                continue
            tier, regions = relevance_tier(row)
            row["tier"] = tier
            row["region_tags"] = regions
            row["license"] = spec["license"]
            row["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            writers[tier].write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[src][tier] += 1
            for r in regions:
                region_counts[r] += 1
            if n % 50_000 == 0:
                print(f"  {n:,} rows  ({dict(stats[src])})")
        print(f"  done: {n:,} rows  {dict(stats[src])}")

    for w in writers.values():
        w.close()

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "sources": {s: {"hf": SOURCES[s]["hf"], "license": SOURCES[s]["license"],
                        "counts": dict(stats[s])} for s in args.sources},
        "tier_totals": {t: sum(stats[s][t] for s in args.sources)
                        for t in ("core_si", "pan_indian", "general")},
        "core_si_region_tags": dict(region_counts.most_common()),
        "note": "scraped sources: weights only, not redistributed (150M data plan)",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                            + "\n", encoding="utf-8")
    print(f"\n=== totals ===\n{json.dumps(manifest['tier_totals'], indent=2)}")
    print(f"core_si regions: {json.dumps(manifest['core_si_region_tags'])}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
