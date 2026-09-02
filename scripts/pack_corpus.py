"""Pack a corpus text file into fixed-length token windows for training.

    python scripts/pack_corpus.py --seq-len 1024 --val-fraction 0.02

Reads `data/interim/sample/corpus.txt` + `artifacts/tokenizers/poc-16k` by default,
writes `data/corpus/poc/{tokens.bin,val.bin,manifest.json}`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train.packing import pack_corpus_file  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=REPO / "data" / "interim" / "sample" / "corpus.txt")
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO / "artifacts" / "tokenizers" / "poc-16k")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "corpus" / "poc")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--add-bos-per-window", action="store_true")
    args = ap.parse_args()

    for p in (args.corpus, args.tokenizer / "tokenizer.json"):
        if not p.exists():
            print(f"ERROR: {p} not found")
            return 1

    man = pack_corpus_file(
        args.corpus, args.tokenizer, args.seq_len, args.out,
        add_bos_per_window=args.add_bos_per_window, val_fraction=args.val_fraction,
    )
    print(f"packed -> {args.out}")
    print(json.dumps(man.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
