"""One-off: bulk-approves every currently-pending golden-QA candidate, per explicit
owner instruction (2026-10-09) overriding this project's normal one-at-a-time human
review requirement (evaluation_strategy.md SS5 / golden_qa_review.py's own caption:
"no sampling, no batch-approve").

The owner was shown, before confirming, that all pending items at the time (668) were
unsourced deepseek-r1-drafted content with zero citations -- i.e. this bulk-approval
makes the golden set's majority content AI-drafted with no independent per-item human
check, which may limit its usefulness later for calibrating the judge panel (deepseek-r1
is one of its members). The owner explicitly chose "auto-accept everything anyway"
after seeing that tradeoff stated plainly.

Reuses golden_qa_review.py's own load/save/append functions (not a raw file edit) so
the resulting data is identical in shape to what clicking Approve 668 times would have
produced, and stamps an honest reviewer_notes entry on every affected item recording
that this was a bulk action, not an individual review -- so the data's own provenance
stays truthful for anyone reading it later (matching the existing "drafted_by" field's
honesty, not just the newly-approved status).

    python scripts/bulk_approve_pending_golden.py
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_qa_review as gqr  # noqa: E402

NOTE = (f"Bulk-approved {datetime.now(UTC).date().isoformat()} per explicit owner "
       "instruction, overriding the normal one-at-a-time review requirement -- NOT "
       "individually reviewed. Owner was shown that all pending items at the time "
       "were unsourced deepseek-r1-drafted content with no citation before confirming.")


def main() -> int:
    rows = gqr.load_candidates()
    pending_idx = [i for i, r in enumerate(rows) if r["status"] == "pending"]
    print(f"{len(rows)} total candidates, {len(pending_idx)} pending -> approving all")

    for i in pending_idx:
        rows[i]["status"] = "approved"
        existing_notes = rows[i].get("reviewer_notes") or ""
        rows[i]["reviewer_notes"] = (existing_notes + " " + NOTE).strip()

    gqr.save_candidates(rows)
    for i in pending_idx:
        gqr.append_to_golden_set(rows[i])

    print(f"approved {len(pending_idx)} items, merged into {gqr.GOLDEN_PATH}")
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("final status counts:", by_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
