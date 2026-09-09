"""Turn the tiered recipe JSONLs into deduped prose text for the pretraining corpus
(150M dataset plan, Level 1). No GPU, no LLM.

    python scripts/build_recipe_corpus.py

Reads data/interim/recipes_150m/{core_si,pan_indian,general}.jsonl, renders each row
with recipe_transform.to_prose, drops exact + near-duplicates, scans for held-out
forbidden entities, and writes data/interim/domain_150m/recipes/<tier>.txt + a manifest
with per-tier token counts.

Tier upsampling for the training mix is applied later at pack time, not here.
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
from food_slm.data.recipe_transform import to_prose  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _dup_key(row: dict) -> str:
    t = " ".join(row.get("title", "").lower().split())
    ings = " ".join(sorted(i.lower().strip() for i in row.get("ingredients", [])))[:400]
    return hashlib.sha1(f"{t}|{ings}".encode()).hexdigest()  # noqa: S324


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=REPO / "data" / "interim" / "recipes_150m")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "recipes")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    forbidden = load_forbidden(REPO / "data" / "eval")

    try:
        from food_slm.tokenizer import load_tokenizer

        tok = load_tokenizer(REPO / "artifacts" / "tokenizers" / "corpus-16k")
    except Exception:  # noqa: BLE001 - token count is informational
        tok = None

    seen: set[str] = set()
    report: dict[str, dict] = {}
    for tier in ("core_si", "pan_indian", "general"):
        src = args.in_dir / f"{tier}.jsonl"
        if not src.exists():
            continue
        kept = dupes = forbid = 0
        chars = toks = 0
        with (args.out / f"{tier}.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"# --- recipe corpus: {tier} (scraped; weights only) ---\n")
            for line in src.open(encoding="utf-8"):
                row = json.loads(line)
                k = _dup_key(row)
                if k in seen:
                    dupes += 1
                    continue
                seen.add(k)
                text = to_prose(row)
                if forbidden.scan_text(text):
                    forbid += 1
                    continue
                fh.write(text)
                fh.write("\n\n")
                kept += 1
                chars += len(text) + 2
                if tok is not None:
                    toks += len(tok.encode(text).ids) + 1
        report[tier] = {"kept": kept, "duplicates": dupes, "forbidden_hits": forbid,
                        "chars": chars, "tokens": toks or None}
        print(f"  {tier:<11} kept {kept:>6}  dup {dupes:>5}  forbid {forbid:>3}  "
              f"~{(toks or chars // 4):,} tok")

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "recipe datasets (food_com/archanas/epicurious) via recipe_transform.to_prose",
        "license": "research-only scrapes — weights only, not redistributed",
        "tiers": report,
        "total_tokens": sum((r["tokens"] or r["chars"] // 4) for r in report.values()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"\ntotal ~{manifest['total_tokens']:,} tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
