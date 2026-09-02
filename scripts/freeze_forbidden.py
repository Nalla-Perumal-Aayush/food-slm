"""Freeze (or amend) the forbidden-entity list (baseline §9.2).

Writes ``data/eval/forbidden_entities.lock.json`` pinning the SHA-256 of
``forbidden_entities.jsonl`` and appends an entry to
``forbidden_entities.amendments.log``.

    python scripts/freeze_forbidden.py --note "initial freeze before data collection"

Changing an already-frozen list REQUIRES --note explaining why. Data-collection and
training tooling call ``food_slm.data.forbidden.verify_lock`` and refuse to run if
the file no longer matches the lock.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import LIST_NAME, LOCK_NAME, compute_lock  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
LOG_NAME = "forbidden_entities.amendments.log"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="why the list is being (re)frozen")
    args = ap.parse_args()

    list_path = EVAL_DIR / LIST_NAME
    lock_path = EVAL_DIR / LOCK_NAME
    if not list_path.exists():
        print(f"ERROR: {list_path} not found")
        return 1

    new = compute_lock(EVAL_DIR)
    prev = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    now = datetime.now(UTC).isoformat(timespec="seconds")

    if prev and prev.get("sha256") == new["sha256"]:
        print("No change — list already matches the lock.")
        return 0

    amendments = (prev or {}).get("amendments", [])
    amendments.append({
        "at": now,
        "from_sha256": (prev or {}).get("sha256"),
        "to_sha256": new["sha256"],
        "n_entries": new["n_entries"],
        "note": args.note,
    })
    lock = {**new, "frozen_on": (prev or {}).get("frozen_on", now[:10]),
            "last_amended": now, "amendments": amendments}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    with (EVAL_DIR / LOG_NAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{now}  {(prev or {}).get('sha256', '-')[:12]} -> {new['sha256'][:12]}  "
                 f"n={new['n_entries']}  {args.note}\n")

    verb = "amended" if prev else "frozen"
    print(f"{verb}: {new['n_entries']} entries, sha256 {new['sha256'][:12]}")
    print(f"  {new['kinds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
