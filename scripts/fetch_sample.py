"""Build a small development corpus for the 15M PoC (baseline §8.0).

NOT the training corpus — just enough permissive English + the serialized canonical
store to train a tokenizer, pack shards, and exercise the training loop. Judged for
pipeline correctness, not model quality.

    python scripts/fetch_sample.py --mb 300 --source wikitext

Sources (all permissive / openly licensed):
  wikitext     Salesforce/wikitext  wikitext-103-raw-v1   (CC-BY-SA)   [default, ~180 MB max]
  wikipedia    wikimedia/wikipedia  20231101.en           (CC-BY-SA)   [streamed]
  fineweb-edu  HuggingFaceFW/fineweb-edu  sample-10BT     (ODC-By)     [streamed]

If the download fails (no network / HF unreachable), writes the canonical-store text
alone (repeated to a floor size) so downstream code is still testable, and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.serialize import store_to_text  # noqa: E402
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

SOURCES = {
    "wikitext": dict(path="Salesforce/wikitext", name="wikitext-103-raw-v1",
                     split="train", field="text", streaming=False, license="CC-BY-SA-3.0"),
    "wikipedia": dict(path="wikimedia/wikipedia", name="20231101.en",
                      split="train", field="text", streaming=True, license="CC-BY-SA-4.0"),
    "fineweb-edu": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                        split="train", field="text", streaming=True, license="ODC-By-1.0"),
}


def _english_docs(source: str, budget_bytes: int):
    from datasets import load_dataset

    spec = SOURCES[source]
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"],
                      streaming=spec["streaming"])
    got = 0
    for row in ds:
        text = (row.get(spec["field"]) or "").strip()
        if len(text) < 200:
            continue
        yield text
        got += len(text.encode("utf-8")) + 2
        if got >= budget_bytes:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=300.0, help="target corpus size in MB")
    ap.add_argument("--source", choices=list(SOURCES), default="wikitext")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "interim" / "sample")
    ap.add_argument("--canonical", type=Path, default=REPO / "data" / "canonical")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / "corpus.txt"
    budget = int(args.mb * 1024 * 1024)

    store = load_store(args.canonical, strict=True)
    canon_text = store_to_text(store)

    english_ok, english_bytes, note = True, 0, ""
    with corpus.open("w", encoding="utf-8") as fh:
        fh.write("# --- canonical store (serialized) ---\n")
        fh.write(canon_text)
        fh.write("\n")
        try:
            fh.write(f"# --- {args.source} sample ---\n")
            for doc in _english_docs(args.source, budget - len(canon_text.encode())):
                fh.write(doc)
                fh.write("\n\n")
                english_bytes += len(doc.encode("utf-8")) + 2
        except Exception as exc:  # noqa: BLE001 - fall back, keep pipeline testable
            english_ok = False
            note = f"english download failed ({type(exc).__name__}: {exc}); wrote canonical only"
            # pad with repeats of the canonical text so packing/tokenizer still have volume
            reps = max(1, (2 * 1024 * 1024) // max(1, len(canon_text.encode())))
            for _ in range(reps):
                fh.write(canon_text)
                fh.write("\n")

    size = corpus.stat().st_size
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": "15M PoC development sample — NOT the training corpus (baseline §8.0)",
        "bytes": size,
        "mb": round(size / 1024 / 1024, 1),
        "canonical_records": len(store),
        "english_source": args.source if english_ok else None,
        "english_license": SOURCES[args.source]["license"] if english_ok else None,
        "english_bytes": english_bytes,
        "note": note or "ok",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")

    print(f"wrote {corpus}  ({manifest['mb']} MB)")
    print(f"  canonical records: {len(store)}   english: "
          f"{args.source if english_ok else 'FAILED — canonical-only fallback'}")
    if not english_ok:
        print(f"  WARNING: {note}")
        print("  (rerun with network access for a representative sample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
