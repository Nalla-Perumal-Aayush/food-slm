"""Golden QA review tool -- the human-verification gate for every candidate
drafted by draft_golden_candidates.py (or future sourcing passes).

    streamlit run scripts/golden_qa_review.py

Every candidate needs an explicit decision here before it counts as golden
(evaluation_strategy.md SS5: "never AI generation" for the golden set itself --
no sampling, no batch-approve). Approve (optionally after editing the answer),
or reject with a reason. Approved items are written to data/eval/golden_qa_set.jsonl,
the actual golden set; data/eval/golden_qa_candidates.jsonl keeps every item's
full review history (status/notes), nothing is deleted.

Shows running progress toward the ~500-1,000 target and a region x task-type
coverage table so it's obvious what to source next.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = REPO / "data" / "eval" / "golden_qa_candidates.jsonl"
GOLDEN_PATH = REPO / "data" / "eval" / "golden_qa_set.jsonl"
TARGET = 500  # low end of the ~500-1,000 spec target (evaluation_strategy.md)

TASK_TYPES = ["recipe_request", "scaling", "ingredient_substitution", "troubleshooting",
             "technique_explanation", "regional_comparison", "definition", "abstention"]
REGIONS = ["tamil_nadu", "kerala", "karnataka", "andhra_pradesh", "telangana",
          "puducherry", "pan_south"]

st.set_page_config(page_title="Golden QA Review", layout="wide")


def load_candidates() -> list[dict]:
    if not CANDIDATES_PATH.exists():
        return []
    return [json.loads(line) for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
           if line.strip()]


def save_candidates(rows: list[dict]) -> None:
    CANDIDATES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def append_to_golden_set(row: dict) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if GOLDEN_PATH.exists():
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                existing[d["id"]] = d
    existing[row["id"]] = row
    GOLDEN_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted(existing.values(), key=lambda r: r["id"]))
        + "\n", encoding="utf-8")


def remove_from_golden_set(item_id: str) -> None:
    if not GOLDEN_PATH.exists():
        return
    rows = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    rows = [r for r in rows if r["id"] != item_id]
    GOLDEN_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8")


def render_progress(rows: list[dict]) -> None:
    by_status = Counter(r["status"] for r in rows)
    approved = by_status.get("approved", 0)
    st.subheader("Progress")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved (golden)", f"{approved} / {TARGET}")
    c2.metric("Pending review", by_status.get("pending", 0))
    c3.metric("Rejected", by_status.get("rejected", 0))
    c4.metric("Total candidates drafted", len(rows))
    st.progress(min(1.0, approved / TARGET), text=f"{approved}/{TARGET} toward the golden-set target")

    approved_rows = [r for r in rows if r["status"] == "approved"]
    st.subheader("Coverage: approved items by region x task type")
    if approved_rows:
        matrix = pd.DataFrame(0, index=REGIONS + [None], columns=TASK_TYPES)
        for r in approved_rows:
            if r["region"] in matrix.index and r["task_type"] in matrix.columns:
                matrix.loc[r["region"], r["task_type"]] += 1
        matrix.index = [i if i is not None else "(none / cross-region)" for i in matrix.index]
        st.dataframe(matrix, use_container_width=True)
    else:
        st.info("No approved items yet -- the coverage table fills in as you approve candidates.")


def render_item(row: dict, rows: list[dict], idx: int) -> None:
    with st.container(border=True):
        st.caption(f"{row['id']}  |  task: {row['task_type']}  |  "
                  f"region: {row['region'] or '(none)'}  |  diet: {row['diet']}")
        st.markdown(f"**Q:** {row['question']}")

        edit_key = f"answer_{row['id']}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = row["candidate_answer"]
        st.text_area("Candidate answer (edit before approving if needed)",
                     key=edit_key, height=120)

        src = row.get("source") or {}
        if src.get("url"):
            with st.expander(f"Source: {src['url']}"):
                st.write(src.get("quote", "(no quote recorded)"))
        else:
            st.caption("No source citation (abstention item -- behavior, not a fact claim).")

        notes_key = f"notes_{row['id']}"
        st.text_input("Reviewer notes (optional)", key=notes_key,
                      value=row.get("reviewer_notes", ""))

        c1, c2, c3 = st.columns(3)
        if c1.button("Approve", key=f"approve_{row['id']}", type="primary"):
            rows[idx]["candidate_answer"] = st.session_state[edit_key]
            rows[idx]["status"] = "approved"
            rows[idx]["reviewer_notes"] = st.session_state[notes_key]
            save_candidates(rows)
            append_to_golden_set(rows[idx])
            st.rerun()
        if c2.button("Reject", key=f"reject_{row['id']}"):
            rows[idx]["status"] = "rejected"
            rows[idx]["reviewer_notes"] = st.session_state[notes_key]
            save_candidates(rows)
            remove_from_golden_set(row["id"])
            st.rerun()
        if row["status"] != "pending" and c3.button("Reset to pending", key=f"reset_{row['id']}"):
            rows[idx]["status"] = "pending"
            save_candidates(rows)
            remove_from_golden_set(row["id"])
            st.rerun()


def main() -> None:
    st.title("Golden QA Review")
    st.caption(
        "Every item needs an explicit Approve/Reject -- no sampling, no batch-approve. "
        "This is the human-verification gate the whole eval design depends on "
        "(evaluation_strategy.md SS5: golden sets are never AI-generated without this)."
    )
    rows = load_candidates()
    if not rows:
        st.warning(f"No candidates found at `{CANDIDATES_PATH}`. Run "
                  "scripts/draft_golden_candidates.py first.")
        st.stop()

    render_progress(rows)

    st.subheader("Review queue")
    f1, f2, f3 = st.columns(3)
    status_filter = f1.selectbox("Show", ["pending", "approved", "rejected", "all"], index=0)
    task_filter = f2.selectbox("Task type", ["(all)"] + TASK_TYPES)
    region_filter = f3.selectbox("Region", ["(all)"] + REGIONS + ["(none)"])

    indexed = list(enumerate(rows))
    if status_filter != "all":
        indexed = [(i, r) for i, r in indexed if r["status"] == status_filter]
    if task_filter != "(all)":
        indexed = [(i, r) for i, r in indexed if r["task_type"] == task_filter]
    if region_filter != "(all)":
        want = None if region_filter == "(none)" else region_filter
        indexed = [(i, r) for i, r in indexed if r["region"] == want]

    st.caption(f"{len(indexed)} item(s) match this filter.")
    for idx, row in indexed:
        render_item(row, rows, idx)


if __name__ == "__main__":
    main()
