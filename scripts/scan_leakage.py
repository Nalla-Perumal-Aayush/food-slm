"""Forbidden-entity leakage scan over the assembled corpus (plan_40m §I2, baseline §9.2).

Held-out probe entities (``data/eval/forbidden_entities.jsonl``) must never appear in any
training data. This is a hard gate — run it before packing and before any training run.

    python scripts/scan_leakage.py \
        data/interim/corpus_40m/fineweb-edu data/interim/domain_40m/corpus.txt

Also re-checks the canonical store and verifies the forbidden list is still frozen.
Exits non-zero on any hit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import (  # noqa: E402
    check_canonical_store,
    load_forbidden,
    scan_corpus_files,
    verify_lock,
)
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "data" / "eval"


def _expand(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        elif p.exists():
            out.append(p)
        else:
            print(f"  (skip missing: {p})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", type=Path, nargs="+", help="corpus files or shard dirs")
    ap.add_argument("--canonical", type=Path, default=REPO / "data" / "canonical")
    args = ap.parse_args()

    try:
        verify_lock(EVAL)
        print("forbidden list: frozen OK")
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 2

    forbidden = load_forbidden(EVAL)
    print(f"forbidden entities: {len(forbidden)}")

    store_issues = check_canonical_store(load_store(args.canonical, strict=False), forbidden)
    if store_issues:
        print("\nCANONICAL STORE LEAKS:")
        for i in store_issues:
            print(f"  {i}")

    files = _expand(args.paths)
    print(f"\nscanning {len(files)} corpus files...")
    result = scan_corpus_files(files, forbidden)
    print(f"  {result['chars_scanned'] / 1e6:.0f}M chars scanned")

    if result["hits"]:
        print("\nCORPUS LEAKS (held-out probe entities found in training data):")
        for eid, where in result["hits"].items():
            ent = next((e for e in forbidden.entities if e.id == eid), None)
            sf = ent.surface_forms[0] if ent else "?"
            print(f"  {eid} ({sf!r}) in {where}")

    clean = result["clean"] and not store_issues
    out = {"corpus": result, "store_issues": store_issues, "clean": clean}
    rp = REPO / "artifacts" / "reports" / "leakage_scan.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'CLEAN' if clean else 'LEAKS FOUND'} — report -> {rp}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
