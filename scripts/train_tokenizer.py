"""Train the PoC tokenizer and (optionally) print its analysis report.

    python scripts/fetch_sample.py --mb 300          # once, to build the sample
    python scripts/train_tokenizer.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.schema import load_store  # noqa: E402
from food_slm.tokenizer.analyze import analyze_tokenizer  # noqa: E402
from food_slm.tokenizer.train import TokenizerSpec, load_tokenizer, train_tokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _canonical_names(canonical_dir: Path) -> tuple[list[str], list[str]]:
    store = load_store(canonical_dir, strict=True)
    dish, native = [], []
    for rec in store.records.values():
        if rec.type in ("dish", "ingredient", "technique"):
            dish.append(rec.labels[0])
            for n in getattr(rec.structure, "names", []) or []:
                native.append(n.native)
    return dish, native


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tokenizer.yaml")
    ap.add_argument("--corpus", type=Path,
                    default=REPO / "data" / "interim" / "sample" / "corpus.txt")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "tokenizers" / "poc-16k")
    ap.add_argument("--vocab", type=int, default=None, help="override vocab_size")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: {args.corpus} not found — run scripts/fetch_sample.py first")
        return 1

    spec = TokenizerSpec.from_yaml(args.config)
    if args.vocab:
        spec.vocab_size = args.vocab

    out = train_tokenizer([args.corpus], spec, args.out, show_progress=args.progress)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    print(f"trained -> {out}")
    print(f"  target vocab {spec.vocab_size}  actual {meta['actual_vocab_size']}")
    print(f"  special ids: {meta['special_token_ids']}")

    if args.analyze:
        tok = load_tokenizer(out)
        texts = [b for b in args.corpus.read_text(encoding="utf-8").split("\n\n")
                 if b.strip() and not b.startswith("#")][:2000]
        dish, native = _canonical_names(REPO / "data" / "canonical")
        report = analyze_tokenizer(tok, texts, dish_names=dish, native_names=native, ctx=1024)
        print("\n--- analysis report ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        (out / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
