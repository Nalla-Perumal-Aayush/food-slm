"""Pack the 150M corpus: general English (FineWeb-Edu ~5B) + weighted domain tiers
(150M dataset plan). Mirrors pack_40m.py with a multi-tier source list.

    python scripts/pack_150m.py --tokenizer artifacts/tokenizers/corpus-16k --seq-len 4096

Domain tiers are upsampled so a specialist actually specialises without over-repeating:
  general English   x1   (substrate)
  general food      x1   (cooking register)
  pan_indian        x2
  core_si recipes   x4
  core_si wiki      x4
  canonical store   x6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train.shards import ShardSource, pack_sharded  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INTERIM = REPO / "data" / "interim"

# (name, glob under data/interim, repeat). Domain reads the DEDUPED tiers only —
# dedup_domain.py has already merged recipes/ + wiki/ + the 40M slice into these.
TIERS = [
    ("fineweb-edu", "corpus_150m/fineweb-edu/*.txt", 1),
    ("general_food", "domain_150m/deduped/general_food.txt", 1),
    ("pan_indian", "domain_150m/deduped/pan_indian.txt", 2),
    ("core_si", "domain_150m/deduped/core_si.txt", 6),
    ("synth", "domain_150m/synth/phrasings.txt", 8),
    ("canonical", "domain_150m/deduped/canonical.txt", 8),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO / "artifacts" / "tokenizers" / "corpus-16k")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "corpus" / "150m")
    ap.add_argument("--val-docs-per-source", type=int, default=3000)
    ap.add_argument("--shard-windows", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    sources: list[ShardSource] = []
    for name, pat, repeat in TIERS:
        paths = sorted(INTERIM.glob(pat))
        if not paths:
            print(f"  (skip {name}: no files at {pat})")
            continue
        sources.append(ShardSource(name, paths, repeat))
        print(f"  {name:<18} x{repeat}  {len(paths)} file(s)")
    if not sources:
        print("ERROR: no sources found — run the fetch/dedup steps first")
        return 1

    man = pack_sharded(sources, args.tokenizer, args.seq_len, args.out,
                       val_docs_per_source=args.val_docs_per_source,
                       shard_windows=args.shard_windows, seed=args.seed)

    tot = man.total_windows * man.seq_len
    print(f"\npacked -> {args.out}")
    print(f"  {man.total_windows:,} train windows x {man.seq_len} = {tot / 1e9:.2f}B tokens")
    print(f"  val {man.val_windows:,} windows  ({', '.join(man.val_files)})")
    mix = man.source_mix
    dom = sum(v["train_tokens"] for k, v in mix.items() if k != "fineweb-edu")
    gen = mix.get("fineweb-edu", {}).get("train_tokens", 1)
    print(f"  source mix (post-upsample tokens): {json.dumps({k: v['train_tokens'] for k, v in mix.items()})}")
    print(f"  domain share: {dom / (dom + gen):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
