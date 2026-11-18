"""Standalone Recipe Extraction monitor.

    streamlit run scripts/extraction_dashboard.py

Same content as scripts/dashboard.py's "Recipe Extraction" tab, but as its own
page rather than a tab buried inside the training dashboard -- for when this is
the only thing you want open. Self-contained like inference_ui.py (each
Streamlit app in this project is its own script; small helpers are duplicated
rather than cross-imported between scripts/*.py), so a change made here has to
be mirrored into dashboard.py's tab and vice versa.

Never touches the extraction process or its staged files: read-only against
data/interim/canon_staging/ and artifacts/logs/canon_extract_recipes*.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "data" / "interim" / "canon_staging"
EXTRACT_LOG = REPO / "artifacts" / "logs" / "canon_extract_recipes.log"
EXTRACT_PROGRESS = REPO / "artifacts" / "logs" / "canon_extract_recipes_progress.json"

st.set_page_config(page_title="Food SLM Recipe Extraction Monitor", layout="wide")

# validate.py's RecordVerdict.apply_to() stamps "{provider}:{decision}" reasons
# (in this exact spelling, joined "; ") into quality.notes -- "qwen7b"/"gemma9b"
# from the two-model panel, "tiebreak" only when the panel split AND
# corroboration >= 2 (see validate.validate_records's comment on why it's
# skipped otherwise: a skipped tiebreak can't change the outcome, so there's no
# point burning ~6 tok/s deepseek-r1:32b CPU time on it).
_JUDGE_COLUMNS = [("qwen7b", "Judge: Qwen2.5-7B"), ("gemma9b", "Judge: Gemma2-9B"),
                  ("tiebreak", "Judge: DeepSeek-R1 (heavy)")]
_VERDICT_RE = re.compile(r"\b(qwen7b|gemma9b|tiebreak):(accept|reject|uncertain)\b")


def parse_judge_notes(notes: str) -> dict[str, str]:
    found = dict(_VERDICT_RE.findall(notes or ""))
    return {key: found.get(key, "not needed" if key == "tiebreak" else "n/a")
           for key, _ in _JUDGE_COLUMNS}


def load_extraction_progress() -> dict:
    if not EXTRACT_PROGRESS.exists():
        return {}
    try:
        return json.loads(EXTRACT_PROGRESS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_staged_recipes() -> pd.DataFrame:
    """One row per staged `recipe` record (rejected drafts are never staged, so
    every row here already passed schema + forbidden-entity + panel-vote)."""
    d = STAGING / "recipe"
    rows = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        struct = rec.get("structure", {})
        quality = rec.get("quality", {})
        notes = quality.get("notes", "") or ""
        judges = parse_judge_notes(notes)
        diet = struct.get("diet", {})
        rows.append({
            "id": rec.get("id"), "dish_id": struct.get("dish_id"),
            "region": struct.get("cuisine_state"),
            "diet": "veg" if diet.get("vegetarian") else "non-veg",
            "ingredients": len(struct.get("ingredients", [])),
            "steps": len(struct.get("steps", [])),
            "status": quality.get("validation_status", "unverified"),
            "corroboration": quality.get("corroboration_count", 0),
            "confidence": quality.get("confidence"),
            "judge_qwen7b": judges["qwen7b"], "judge_gemma9b": judges["gemma9b"],
            "judge_tiebreak": judges["tiebreak"],
            "forbidden_clean": "forbidden entity" not in notes,
            "notes": notes,
            "source_ids": ", ".join(rec.get("provenance", {}).get("source_ids", [])),
            "staged_file": p.name,
        })
    return pd.DataFrame(rows)


def render_extraction_page() -> None:
    prog = load_extraction_progress()

    st.subheader("Run progress")
    if not prog:
        st.info("No progress snapshot yet -- `canon_extract_recipes_progress.json` "
                "is written every 10 docs; check back shortly after the run starts.")
    else:
        done = prog.get("docs_done", 0)
        total = prog.get("docs_total_this_run", 0)
        rate = prog.get("docs_per_min", 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Docs processed", f"{done:,} / {total:,}" if total else f"{done:,}")
        c2.metric("Docs / min", f"{rate:.2f}")
        c3.metric("Records staged", f"{prog.get('staged', 0):,}")
        c4.metric("No recipe found", f"{prog.get('docs_no_recipe', 0):,}")
        c5.metric("Errors", f"{prog.get('docs_error', 0):,}")

        if total:
            frac = min(1.0, done / total) if total else 0.0
            eta_txt = f"{done:,}/{total:,} docs"
            if rate and rate > 0:
                eta_h = (total - done) / rate / 60
                eta_txt += f" -- ETA ~{eta_h:.1f} h at current pace"
            st.progress(frac, text=eta_txt)

        by_decision = prog.get("by_decision") or {}
        if by_decision:
            st.caption("Final decisions across all docs processed this run "
                      "(includes rejected drafts, which are never staged to disk):")
            st.bar_chart(pd.Series(by_decision, name="count"))
        st.caption(f"Last updated: {prog.get('updated', 'n/a')} "
                  f"({'still running' if not prog.get('final') else 'run finished / stopped'})")

    st.subheader("Staged recipe records -- per-generation checklist")
    df = load_staged_recipes()
    if df.empty:
        st.info(f"No recipe records staged yet under `{STAGING / 'recipe'}`.")
    else:
        f1, f2, f3 = st.columns(3)
        statuses = sorted(df["status"].dropna().unique())
        status_filter = f1.multiselect("Status", statuses, default=statuses)
        region_filter = f2.multiselect("Region", sorted(df["region"].dropna().unique()))
        diet_filter = f3.multiselect("Diet", sorted(df["diet"].dropna().unique()))

        view = df[df["status"].isin(status_filter)]
        if region_filter:
            view = view[view["region"].isin(region_filter)]
        if diet_filter:
            view = view[view["diet"].isin(diet_filter)]

        st.dataframe(
            view[["id", "region", "diet", "ingredients", "steps", "status", "corroboration",
                 "judge_qwen7b", "judge_gemma9b", "judge_tiebreak"]]
            .rename(columns={"judge_qwen7b": "Judge: Qwen2.5-7B",
                             "judge_gemma9b": "Judge: Gemma2-9B",
                             "judge_tiebreak": "Judge: DeepSeek-R1 (heavy)"})
            .sort_values("id"),
            use_container_width=True, height=420,
        )
        st.caption(
            f"{len(view)} / {len(df)} staged records shown. DeepSeek-R1 (the heavyweight "
            "tiebreaker) is only consulted when the Qwen/Gemma panel splits **and** "
            "corroboration >= 2 -- every other row reads **'not needed'** in that column "
            "by design (a skipped tiebreak can't change the outcome, so it isn't run)."
        )

        st.subheader("Record detail + checklist")
        options = view["id"].tolist()
        sel = st.selectbox("Pick a record", options) if options else None
        if sel:
            row = df[df["id"] == sel].iloc[0]

            def _mark(ok: bool) -> str:
                return "✅" if ok else "❌"

            st.markdown(f"""
- {_mark(True)} Schema validated (only schema-valid drafts ever reach staging)
- {_mark(row["forbidden_clean"])} Forbidden-entity clean
- **Judge 1 (Qwen2.5-7B):** {row['judge_qwen7b']}
- **Judge 2 (Gemma2-9B):** {row['judge_gemma9b']}
- **Judge 3 (DeepSeek-R1, heavy tiebreak):** {row['judge_tiebreak']}
- **Corroboration count:** {row['corroboration']} """
                       f"({'meets' if row['corroboration'] >= 2 else 'below'} the "
                       "auto-validation bar of >= 2)\n"
                       f"- **Final decision:** `{row['status']}`")
            st.caption(f"Raw verdict notes: {row['notes'] or '(none)'}")
            with st.expander("Full record JSON"):
                full = json.loads((STAGING / "recipe" / row["staged_file"]).read_text(encoding="utf-8"))
                st.json(full)

    st.subheader("Live log tail")
    if EXTRACT_LOG.exists():
        lines = EXTRACT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        st.code("\n".join(lines[-40:]) or "(empty)", language=None)
    else:
        st.info(f"No log file yet at `{EXTRACT_LOG}`.")


def main() -> None:
    st.title("Food SLM -- Recipe Extraction Monitor")
    with st.sidebar:
        auto = st.checkbox("Auto-refresh", value=True)
        interval = st.slider("Refresh every (s)", 5, 60, 15) if auto else None
        if st.button("Refresh now"):
            st.rerun()

    @st.fragment(run_every=interval if auto else None)
    def _live_page() -> None:
        render_extraction_page()

    _live_page()


if __name__ == "__main__":
    main()
