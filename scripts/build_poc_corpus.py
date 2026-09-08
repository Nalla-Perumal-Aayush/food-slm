"""Refresh the 15M PoC development corpus in place (baseline §8.0).

The English substrate in ``data/interim/sample/corpus.txt`` is kept byte-for-byte;
only the serialized canonical-store block at the top is regenerated from the current
``data/canonical`` store and (optionally) repeated so the structural tokens
(``<DISH>``/``<STEP>``/…) get real gradient signal in a short PoC run.

    python scripts/build_poc_corpus.py --canon-repeats 8

This is NOT the training corpus — it is a pipeline-exercising sample. Use
``fetch_sample.py`` to build the corpus from scratch (re-downloads the English
sample); use this when you only want to fold newly-promoted canonical records into
an existing sample without touching the English half.
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
ENGLISH_HEADER_PREFIXES = ("# --- wikitext sample ---", "# --- wikipedia sample ---",
                           "# --- fineweb-edu sample ---")
CANON_HEADER = "# --- canonical store (serialized) ---"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=REPO / "data" / "interim" / "sample")
    ap.add_argument("--canonical", type=Path, default=REPO / "data" / "canonical")
    ap.add_argument("--canon-repeats", type=int, default=8,
                    help="times to repeat the serialized store (structural-token signal)")
    args = ap.parse_args()

    corpus = args.sample_dir / "corpus.txt"
    if not corpus.exists():
        print(f"ERROR: {corpus} not found — run scripts/fetch_sample.py first")
        return 1

    lines = corpus.read_text(encoding="utf-8").splitlines(keepends=True)
    cut = next((i for i, ln in enumerate(lines)
                if ln.rstrip("\n") in ENGLISH_HEADER_PREFIXES), None)
    if cut is None:
        print("ERROR: no English-sample header found; corpus.txt layout unexpected")
        return 1
    english_part = "".join(lines[cut:])

    store = load_store(args.canonical, strict=True)
    canon_text = store_to_text(store)
    reps = max(1, args.canon_repeats)

    with corpus.open("w", encoding="utf-8") as fh:
        fh.write(CANON_HEADER + "\n")
        for _ in range(reps):
            fh.write(canon_text)
            fh.write("\n")
        fh.write(english_part)

    size = corpus.stat().st_size
    man_path = args.sample_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    man.update({
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": "15M PoC development sample — NOT the training corpus (baseline §8.0)",
        "bytes": size,
        "mb": round(size / 1024 / 1024, 1),
        "canonical_records": len(store),
        "canonical_repeats": reps,
        "canonical_bytes": len(canon_text.encode("utf-8")) * reps,
        "note": f"english half unchanged; canonical block refreshed to {len(store)} "
                f"records x{reps}",
    })
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {corpus}  ({man['mb']} MB)")
    print(f"  canonical: {len(store)} records x{reps}  "
          f"({man['canonical_bytes'] / 1024:.0f} KB)")
    print(f"  english half: {len(english_part.encode()) / 1024 / 1024:.1f} MB (unchanged)")
    print("\nnext: python scripts/pack_corpus.py --seq-len 1024 --val-fraction 0.02")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
