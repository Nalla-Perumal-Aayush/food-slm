"""Pack the 40M corpus: FineWeb-Edu shards + upsampled domain slice (plan_40m §E).

    python scripts/pack_40m.py \
        --general data/interim/corpus_40m/fineweb-edu \
        --domain  data/interim/domain_40m/corpus.txt --domain-repeat 6 \
        --tokenizer artifacts/tokenizers/corpus-16k \
        --seq-len 4096 --out data/corpus/40m

Writes data/corpus/40m/{train/shard_*.bin, val.bin, val/<source>.bin, manifest.json}.
The corpus is packed once at --seq-len; the training curriculum feeds shorter windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train.shards import ShardSource, pack_sharded  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", type=Path, required=True, help="general-English shard dir")
    ap.add_argument("--domain", type=Path, default=None, help="domain corpus file or dir")
    ap.add_argument("--domain-repeat", type=int, default=6)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "corpus" / "40m")
    ap.add_argument("--val-docs-per-source", type=int, default=2000)
    ap.add_argument("--shard-windows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    gen_files = sorted(args.general.glob("*.txt")) if args.general.is_dir() else [args.general]
    if not gen_files:
        print(f"ERROR: no .txt under {args.general}")
        return 1
    sources = [ShardSource("fineweb-edu", gen_files, 1)]
    if args.domain:
        dom_files = sorted(args.domain.glob("*.txt")) if args.domain.is_dir() else [args.domain]
        sources.append(ShardSource("domain", dom_files, args.domain_repeat))

    man = pack_sharded(sources, args.tokenizer, args.seq_len, args.out,
                       val_docs_per_source=args.val_docs_per_source,
                       shard_windows=args.shard_windows, seed=args.seed)

    total_tok = man.total_windows * man.seq_len
    print(f"packed -> {args.out}")
    print(f"  {man.total_windows:,} train windows  x {man.seq_len}  = {total_tok / 1e6:.0f}M tokens")
    print(f"  {man.val_windows:,} val windows  ({', '.join(man.val_files)})")
    print(f"  source mix: {json.dumps(man.source_mix)}")
    dom = man.source_mix.get("domain", {}).get("train_tokens", 0)
    gen = man.source_mix.get("fineweb-edu", {}).get("train_tokens", 1)
    print(f"  domain token share (post-upsample): {dom / (dom + gen):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
