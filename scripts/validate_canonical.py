"""Validate the canonical knowledge store against schema v1.

Runs per-record (Pydantic) + cross-record (referential-integrity, §8) checks and
prints a summary. Exit code 1 if anything failed — wire this into the Phase 11 gate.

    python scripts/validate_canonical.py [--dir data/canonical]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import (  # noqa: E402
    check_canonical_store,
    load_forbidden,
    verify_lock,
)
from food_slm.schema import ValidationError, load_store  # noqa: E402
from food_slm.schema.models import RECORD_FILES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/canonical", type=Path)
    ap.add_argument("--skip-forbidden", action="store_true")
    args = ap.parse_args()

    root = REPO / args.dir if not args.dir.is_absolute() else args.dir

    try:
        store = load_store(root, strict=True)
    except ValidationError as exc:
        print(f"FAIL — {len(exc.issues)} issue(s):")
        for i in exc.issues:
            print(f"  - {i}")
        return 1

    if not args.skip_forbidden:
        try:
            verify_lock(REPO / "data" / "eval")
            forbidden = load_forbidden(REPO / "data" / "eval")
            leaks = check_canonical_store(store, forbidden)
        except RuntimeError as exc:
            print(f"FAIL — forbidden-list lock: {exc}")
            return 1
        if leaks:
            print(f"FAIL — {len(leaks)} forbidden-entity leak(s) in the canonical store:")
            for line in leaks:
                print(f"  - {line}")
            return 1
        print(f"forbidden-entity list: {len(forbidden)} entries, lock verified, no leaks")

    counts = Counter(r.type for r in store.records.values())
    print(f"OK — {len(store)} records from {root}")
    for rtype, fname in RECORD_FILES.items():
        print(f"  {counts.get(rtype, 0):>4}  {rtype:<11} ({fname})")

    synth = sum(1 for r in store.records.values() if r.provenance.synthetic)
    unval = sum(1 for r in store.records.values()
                if r.quality.validation_status in ("unverified", "auto_validated"))
    print(f"\n  synthetic: {synth}   awaiting human validation: {unval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
