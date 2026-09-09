"""MinHash-dedup the domain corpus tiers before packing (150M dataset plan, integrity).

    python scripts/dedup_domain.py

Reads every *.txt under data/interim/domain_150m/ (recipes/, wiki/, plus the 40M
domain slice), dedups documents across all of them with datasketch MinHash-LSH, and
writes data/interim/domain_150m/deduped/<tier>.txt + a report. CPU only.

Tiers are inferred from the file path: files under a dir/name containing "core_si" ->
core_si; "pan_indian" -> pan_indian; "general" or "wiki" (non-si) -> general_food;
"canonical" -> canonical.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.dedup import MinHashDeduper  # noqa: E402
from food_slm.train.packing import iter_text_documents  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _tier_of(path: Path) -> str:
    s = str(path).lower()
    if "canonical" in s:
        return "canonical"
    if "core_si" in s or "domain_40m" in s:   # domain_40m/corpus.txt = hand-picked SI articles
        return "core_si"
    if "pan_indian" in s:
        return "pan_indian"
    return "general_food"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data" / "interim" / "domain_150m")
    ap.add_argument("--also", type=Path, nargs="*",
                    default=[REPO / "data" / "interim" / "domain_40m"])
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    out = args.root / "deduped"
    out.mkdir(parents=True, exist_ok=True)

    def _keep(p: Path) -> bool:
        return "deduped" not in str(p) and "_cache" not in str(p)

    files = sorted(p for p in args.root.rglob("*.txt") if _keep(p))
    for extra in args.also:
        files += sorted(p for p in extra.rglob("*.txt") if _keep(p))
    if not files:
        print(f"no .txt found under {args.root} / {args.also}")
        return 1

    deduper = MinHashDeduper(threshold=args.threshold)
    writers: dict[str, object] = {}
    counts: dict[str, dict] = {}
    total = kept = dupes = 0
    for f in files:
        tier = _tier_of(f)
        w = writers.get(tier)
        if w is None:
            w = writers[tier] = (out / f"{tier}.txt").open("w", encoding="utf-8")
            counts[tier] = {"kept": 0, "dupes": 0, "chars": 0}
        for doc in iter_text_documents(f):
            total += 1
            if len(doc) < 120:
                continue
            key = f"{f.name}:{total}"
            if deduper.is_duplicate(doc)[0]:
                dupes += 1
                counts[tier]["dupes"] += 1
                continue
            deduper.add(key, doc)
            w.write(doc + "\n\n")
            kept += 1
            counts[tier]["kept"] += 1
            counts[tier]["chars"] += len(doc)
        print(f"  {f.relative_to(REPO)}  -> {tier}")

    for w in writers.values():
        w.close()
    report = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": len(files), "docs_in": total, "kept": kept, "duplicates_dropped": dupes,
        "tiers": {t: {**c, "est_tokens": c["chars"] // 4} for t, c in counts.items()},
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{kept:,} kept / {dupes:,} dup  ->  {out}")
    for t, c in report["tiers"].items():
        print(f"  {t:<13} {c['kept']:>6} docs  ~{c['est_tokens']:,} tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
