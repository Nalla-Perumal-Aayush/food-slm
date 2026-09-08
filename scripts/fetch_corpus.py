"""Stream a large general-English corpus into sharded text files (baseline §8, plan_40m §B).

    python scripts/fetch_corpus.py --tokens 1_000_000_000 --source fineweb-edu

Streams from Hugging Face (no full-dataset download), applies a min-length filter, a
sampled language-ID gate, and cheap exact-dedup, and writes blank-line-separated docs to
``data/interim/corpus_40m/<source>/shard_NNNNN.txt`` (~200 MB each) plus a manifest with
per-shard SHA-256, token estimates, and provenance.

Token budget is enforced on a chars/4 estimate; the exact count comes from packing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]

SOURCES = {
    "fineweb-edu": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                        split="train", field="text", license="ODC-By-1.0"),
    "fineweb": dict(path="HuggingFaceFW/fineweb", name="sample-10BT",
                    split="train", field="text", license="ODC-By-1.0"),
    "c4": dict(path="allenai/c4", name="en", split="train", field="text",
               license="ODC-By-1.0"),
}
CHARS_PER_TOKEN = 4  # rough English estimate for the budget gate


def _dedup_key(text: str) -> str:
    head = " ".join(text.split())[:500].lower()
    return hashlib.sha1(head.encode("utf-8")).hexdigest()  # noqa: S324 - dedup key, not security


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=lambda s: int(s.replace("_", "")), default=1_000_000_000,
                    help="target token budget (chars/4 estimate)")
    ap.add_argument("--source", choices=list(SOURCES), default="fineweb-edu")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "interim" / "corpus_40m")
    ap.add_argument("--shard-mb", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--lang-check-every", type=int, default=25,
                    help="run langid on 1 in N docs (0 = never)")
    ap.add_argument("--lang-abort-rate", type=float, default=0.02,
                    help="abort if sampled non-English rate exceeds this")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    out_root = args.out if args.out.is_absolute() else (REPO / args.out)
    out_dir = out_root / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    budget_chars = args.tokens * CHARS_PER_TOKEN
    shard_bytes = args.shard_mb * 1024 * 1024

    detect = None
    if args.lang_check_every:
        from food_slm.data.langid import detect as _detect
        from food_slm.data.langid import ensure_model

        ensure_model()
        detect = _detect

    from datasets import load_dataset

    ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=True)

    shards: list[dict] = []
    seen: set[str] = set()
    n_docs = n_chars = n_short = n_dup = n_nonen = n_lang_checked = 0
    shard_idx = 0
    fh = (out_dir / f"shard_{shard_idx:05d}.txt").open("w", encoding="utf-8")
    shard_chars = 0
    t0 = time.time()

    def _close_shard(path: Path, chars: int) -> None:
        shards.append({
            "file": str(path.relative_to(REPO)),
            "chars": chars,
            "est_tokens": chars // CHARS_PER_TOKEN,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    try:
        for row in ds:
            text = (row.get(spec["field"]) or "").strip()
            if len(text) < args.min_chars:
                n_short += 1
                continue
            key = _dedup_key(text)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            if detect and n_docs % args.lang_check_every == 0:
                n_lang_checked += 1
                lang, conf = detect(text)
                if lang != "en" and conf >= 0.5:
                    n_nonen += 1
                    continue

            fh.write(text)
            fh.write("\n\n")
            clen = len(text) + 2
            n_docs += 1
            n_chars += clen
            shard_chars += clen

            if shard_chars >= shard_bytes:
                fh.close()
                _close_shard(out_dir / f"shard_{shard_idx:05d}.txt", shard_chars)
                shard_idx += 1
                fh = (out_dir / f"shard_{shard_idx:05d}.txt").open("w", encoding="utf-8")
                shard_chars = 0
                rate = n_chars / max(time.time() - t0, 1) / 1e6
                print(f"  shard {shard_idx:>3}  {n_docs:>9,} docs  "
                      f"~{n_chars // CHARS_PER_TOKEN / 1e6:.0f}M tok  {rate:.1f} MB/s")

            if n_lang_checked >= 200 and n_nonen / n_lang_checked > args.lang_abort_rate:
                raise SystemExit(
                    f"ABORT: sampled non-English rate {n_nonen}/{n_lang_checked} "
                    f"> {args.lang_abort_rate}")

            if n_chars >= budget_chars:
                break
    finally:
        fh.close()
        last = out_dir / f"shard_{shard_idx:05d}.txt"
        if shard_chars > 0:
            _close_shard(last, shard_chars)
        elif last.exists():
            last.unlink()

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": spec["path"], "subset": spec["name"], "license": spec["license"],
        "purpose": "40M ablation general-English corpus (baseline §8 / plan_40m §B)",
        "target_tokens": args.tokens,
        "docs": n_docs, "chars": n_chars, "est_tokens": n_chars // CHARS_PER_TOKEN,
        "shards": shards,
        "filtered": {"too_short": n_short, "exact_dup": n_dup,
                     "non_english": n_nonen, "lang_checked": n_lang_checked},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"\ndone: {n_docs:,} docs / ~{n_chars // CHARS_PER_TOKEN / 1e6:.0f}M tok est "
          f"in {len(shards)} shards -> {out_dir}")
    print(f"  filtered: short={n_short:,} dup={n_dup:,} non-en={n_nonen} "
          f"(of {n_lang_checked} checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
