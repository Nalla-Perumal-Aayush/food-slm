"""data/instruction/sft_v1.jsonl -> data/corpus/sft_v1/{train,val}.npz
(food_slm.data.sft_pack.build_sft_pack): tokenized, loss-masked, entity-disjoint
split, with any dish referenced by the golden QA set excluded from training
entirely (see sft_pack.py's module docstring for why).

    python scripts/build_sft_pack.py
    python scripts/build_sft_pack.py --max-seq-len 768 --val-frac 0.08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.sft_pack import build_sft_pack  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", type=Path, default=REPO / "data" / "instruction" / "sft_v1.jsonl")
    ap.add_argument("--canonical", type=Path, default=REPO / "data" / "canonical")
    ap.add_argument("--golden", type=Path, default=REPO / "data" / "eval" / "golden_qa_set.jsonl")
    ap.add_argument("--tokenizer", type=Path, default=REPO / "artifacts" / "tokenizers" / "corpus-16k")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "corpus" / "sft_v1")
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    stats = build_sft_pack(args.instruction, args.canonical, args.golden, args.tokenizer,
                           args.out, max_seq_len=args.max_seq_len, val_frac=args.val_frac,
                           seed=args.seed)
    print(f"train: {stats.train}  val: {stats.val}")
    print(f"excluded (golden QA dish overlap): {stats.excluded_golden_overlap} examples "
         f"across {stats.excluded_golden_dish_count} dishes")
    print(f"skipped (too long for max_seq_len={args.max_seq_len}): {stats.skipped_too_long}")
    print(f"wrote {args.out}/{{train,val}}.npz + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
