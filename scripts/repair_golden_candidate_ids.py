"""One-off repair: golden_qa_candidates.jsonl's ids were hashed from question
text alone, which silently collided two DIFFERENT classes of row:

  1. Genuinely distinct candidates that happen to share question phrasing across
     regions (e.g. "How is mutton biryani prepared?" tagged andhra_pradesh vs
     telangana) -- the id collision meant only one of each such pair could ever
     be merged into golden_qa_set.jsonl (last-written-wins on id), silently
     discarding real, region-specific content.
  2. Genuine full duplicates, where the SAME backlog question (identical
     question/task_type/region/diet) appeared multiple times in
     Documents/golden_qa_700_draft_candidates.jsonl itself (confirmed: 13 such
     questions, several abstention items repeated up to 10x) and got drafted
     and appended once per occurrence.

Fixes both: re-keys every row's id on the full (question, task_type, region,
diet) tuple (case 1 -- now each gets its own correct, distinct id), and
collapses true full-tuple duplicates to a single row (case 2 -- keeping the
first occurrence; the drafted answers differ trivially by sampling temperature
but represent the same intended behavior, confirmed by inspection before
writing this).

Then rebuilds data/eval/golden_qa_set.jsonl from scratch from the repaired,
currently-approved candidates, rather than trying to patch the existing file
in place -- it may hold stale entries under the old, wrong ids.

    python scripts/repair_golden_candidate_ids.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_qa_review as gqr  # noqa: E402


def _key(r: dict) -> tuple:
    return (r["question"], r["task_type"], r.get("region"), r.get("diet"))


def _new_id(r: dict) -> str:
    key_str = f"{r['question']}|{r['task_type']}|{r.get('region')}|{r.get('diet')}"
    digest = hashlib.sha256(key_str.encode()).hexdigest()[:8]
    return f"golden:{digest}"


def main() -> int:
    rows = gqr.load_candidates()
    print(f"loaded {len(rows)} candidate rows")

    seen: dict[tuple, dict] = {}
    dropped_true_dupes = 0
    for r in rows:
        k = _key(r)
        if k in seen:
            dropped_true_dupes += 1
            continue
        seen[k] = r

    repaired = list(seen.values())
    for r in repaired:
        r["id"] = _new_id(r)

    ids = [r["id"] for r in repaired]
    assert len(ids) == len(set(ids)), "repair still left colliding ids -- investigate before writing"

    gqr.save_candidates(repaired)
    print(f"dropped {dropped_true_dupes} true full-duplicate rows, "
         f"{len(repaired)} distinct rows remain, all ids now unique")

    # Rebuild golden_qa_set.jsonl from scratch from the repaired data instead of
    # patching in place -- the old file may hold entries under now-stale ids.
    approved = [r for r in repaired if r["status"] == "approved"]
    gqr.GOLDEN_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted(approved, key=lambda r: r["id"]))
        + "\n", encoding="utf-8")
    print(f"rebuilt {gqr.GOLDEN_PATH} with {len(approved)} approved items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
