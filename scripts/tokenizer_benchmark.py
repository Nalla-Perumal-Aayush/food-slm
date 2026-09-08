"""16k vs 24k vocab benchmark on the real corpus (plan_40m §D, baseline §6).

Trains a byte-level BPE tokenizer at each candidate vocab size on a sample of the
assembled corpus, then scores each on held-out general + domain text:
  - token fertility (tokens/word) and compression (bytes/token)
  - culinary term / native-name fragmentation
  - native (Dravidian) script vocab share vs the 8% guardrail
  - structural-token integrity (each must round-trip as one id)

    python scripts/tokenizer_benchmark.py \
        --corpus data/interim/corpus_40m/fineweb-edu \
        --domain data/interim/domain_40m/corpus.txt \
        --vocabs 16384 24576 --sample-mb 300

Writes artifacts/reports/tokenizer_benchmark_<date>.json and prints a comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.schema import load_store  # noqa: E402
from food_slm.tokenizer.analyze import analyze_tokenizer  # noqa: E402
from food_slm.tokenizer.train import (  # noqa: E402
    TokenizerSpec,
    load_tokenizer,
    train_tokenizer,
)
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STRUCTURAL = ["<DISH>", "<INGREDIENT>", "<QTY>", "<UNIT>", "<STEP>", "<TECHNIQUE>",
              "<TIME>", "<TEMP>", "<REGION>", "<OUTCOME>", "<CONSTRAINT>"]


def _iter_shard_text(path: Path, budget_bytes: int) -> str:
    """Concatenate .txt shards (or a single file) up to a byte budget."""
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    buf, got = [], 0
    for f in files:
        t = f.read_text(encoding="utf-8")
        buf.append(t)
        got += len(t.encode("utf-8"))
        if got >= budget_bytes:
            break
    return "\n\n".join(buf)


def _canonical_names(canonical_dir: Path):
    store = load_store(canonical_dir, strict=True)
    dish, native = [], []
    for rec in store.records.values():
        if rec.type in ("dish", "ingredient", "technique", "equipment"):
            dish.append(rec.labels[0])
            for n in getattr(rec.structure, "names", []) or []:
                native.append(n.native)
    return dish, native


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="general-English shard dir or file")
    ap.add_argument("--domain", type=Path, default=None, help="domain corpus file/dir")
    ap.add_argument("--vocabs", type=int, nargs="+", default=[16384, 24576])
    ap.add_argument("--sample-mb", type=int, default=300)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tokenizer.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "tokenizers")
    args = ap.parse_args()

    work = args.out / "_benchmark"
    work.mkdir(parents=True, exist_ok=True)
    train_txt = work / "train_sample.txt"
    gen_sample = _iter_shard_text(args.corpus, args.sample_mb * 1024 * 1024)
    dom_sample = _iter_shard_text(args.domain, 50 * 1024 * 1024) if args.domain else ""
    train_txt.write_text(gen_sample + ("\n\n" + dom_sample if dom_sample else ""),
                         encoding="utf-8")
    print(f"train sample: {train_txt.stat().st_size / 1e6:.0f} MB "
          f"(general + {'domain' if dom_sample else 'no domain'})")

    gen_eval = [b for b in gen_sample.split("\n\n") if b.strip()][-1500:]
    dom_eval = [b for b in dom_sample.split("\n\n") if b.strip()][-500:] if dom_sample else []
    dish, native = _canonical_names(REPO / "data" / "canonical")
    max_native_share = load_yaml(args.config).get("dravidian_script", {}).get(
        "max_vocab_share", 0.08)

    results = {}
    for v in args.vocabs:
        spec = TokenizerSpec.from_yaml(args.config)
        spec.vocab_size = v
        outdir = args.out / f"bench-{v}"
        train_tokenizer([train_txt], spec, outdir, min_frequency=2)
        tok = load_tokenizer(outdir)

        gen_rep = analyze_tokenizer(tok, gen_eval, dish_names=dish, native_names=native, ctx=4096)
        dom_rep = (analyze_tokenizer(tok, dom_eval, dish_names=dish, native_names=native, ctx=4096)
                   if dom_eval else {})
        struct_ok = {s: len(tok.encode(s).ids) for s in STRUCTURAL}
        results[str(v)] = {
            "actual_vocab": tok.get_vocab_size(),
            "general": {k: gen_rep.get(k) for k in
                        ("token_fertility", "compression_ratio", "vocab_utilization",
                         "culinary_term_fragmentation", "native_script_vocab_share")},
            "domain": {k: dom_rep.get(k) for k in
                       ("token_fertility", "compression_ratio",
                        "culinary_term_fragmentation")} if dom_rep else None,
            "native_script_vocab_share": gen_rep.get("native_script_vocab_share"),
            "native_share_gate": max_native_share,
            "native_share_ok": (gen_rep.get("native_script_vocab_share") or 0) <= max_native_share,
            "structural_tokens_single_id": all(n == 1 for n in struct_ok.values()),
            "structural_token_lengths": struct_ok,
        }

    report = {"created": datetime.now(UTC).isoformat(timespec="seconds"),
              "sample_mb": args.sample_mb, "vocabs": args.vocabs, "results": results}
    rp = REPO / "artifacts" / "reports" / f"tokenizer_benchmark_{datetime.now(UTC):%Y-%m-%d}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'vocab':>8} {'fert':>7} {'bytes/tok':>10} {'dish frag':>10} "
          f"{'native%':>8} {'struct':>7}")
    for v, r in results.items():
        g = r["general"]
        cf = (g.get("culinary_term_fragmentation") or {}).get("mean")
        print(f"{v:>8} {g['token_fertility']:>7} {g['compression_ratio']:>10} "
              f"{cf if cf is not None else '-':>10} "
              f"{r['native_script_vocab_share']:>8} "
              f"{'ok' if r['structural_tokens_single_id'] else 'BAD':>7}"
              + ("" if r["native_share_ok"] else "  <- native share OVER gate"))
    print(f"\nreport -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
