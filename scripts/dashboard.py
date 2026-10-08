"""Single unified Streamlit dashboard: training monitor + model try-out +
recipe-extraction monitor + golden QA review, consolidated from what used to
be four separate app instances (two redundant copies of this file, plus the
now-retired standalone extraction_dashboard.py and golden_qa_review.py direct
launches) into one app on one port.

    streamlit run scripts/dashboard.py

Tab 1, "Training Progress", is the source of truth: it tails a run's
metrics.jsonl (written by food_slm.train.loop), mirrors it to metrics.csv /
eval.csv next to the checkpoints (the artifact this was asked to produce),
and plots loss / grad-norm / LR / throughput. Tab 2, "Try the Model", loads
a checkpoint and generates text for a prompt -- a qualitative sanity check
only. The numbers in Tab 1 are the ones that matter; Tab 2 is for human
satisfaction, not a metric. Tab 3, "Recipe Extraction", watches
scripts/canon_extract_recipes.py's progress snapshot + log + staged output.
Tab 4, "Golden QA Review", embeds scripts/golden_qa_review.py's
render_review_page() -- the human-approval gate for the golden QA set; unlike
tabs 1 and 3 it's interactive (buttons/text edits) so it is NOT wrapped in the
auto-refresh fragment, same reasoning as "Try the Model".

golden_qa_review.py can still be run standalone (`streamlit run
scripts/golden_qa_review.py`) for a dedicated single-purpose instance; this
file just also embeds the same page so one dashboard can cover everything.

Never touches the training process or its checkpoints, and never touches the
extraction process or its staged files: everything here is read-only against
artifacts/checkpoints/<run>/ and data/interim/canon_staging/ (Golden QA
Review is the one interactive exception, writing reviewer decisions to
data/eval/golden_qa_candidates.jsonl / golden_qa_set.jsonl, same as it does
standalone).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import golden_qa_review  # noqa: E402  (sibling script import, scripts/ dir is on sys.path)

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "artifacts" / "checkpoints"
STAGING = REPO / "data" / "interim" / "canon_staging"
EXTRACT_LOG = REPO / "artifacts" / "logs" / "canon_extract_recipes.log"
EXTRACT_PROGRESS = REPO / "artifacts" / "logs" / "canon_extract_recipes_progress.json"

st.set_page_config(page_title="Food SLM Training Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# metrics.jsonl -> DataFrames + CSV mirror (Tab 1 data layer)
# ---------------------------------------------------------------------------

def discover_runs() -> list[str]:
    if not CKPT_ROOT.exists():
        return []
    return sorted(
        p.name for p in CKPT_ROOT.iterdir()
        if p.is_dir() and (p / "metrics.jsonl").exists()
    )


def load_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse metrics.jsonl into (step_df, eval_df). Tolerant of a partially
    -flushed last line, since a training process may be writing this file
    right now."""
    steps, evals = [], []
    p = run_dir / "metrics.jsonl"
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line mid-write; pick it up next refresh
            if "eval" in row:
                evals.append(row["eval"])
            else:
                steps.append(row)
    step_df = pd.DataFrame(steps)
    eval_df = pd.DataFrame(evals)
    return step_df, eval_df


def mirror_csv(run_dir: Path, step_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[Path, Path]:
    """Write the standing CSV artifacts the dashboard is asked to maintain."""
    steps_csv = run_dir / "metrics.csv"
    eval_csv = run_dir / "eval.csv"
    if not step_df.empty:
        step_df.to_csv(steps_csv, index=False)
    if not eval_df.empty:
        eval_df.to_csv(eval_csv, index=False)
    return steps_csv, eval_csv


def load_train_config(run_name: str) -> dict:
    """Best-effort: configs/train_<run>.yaml, else the first configs/train_*.yaml
    that references this run_name, else {}."""
    from food_slm.utils import load_yaml

    candidates = [
        REPO / "configs" / f"train_{run_name}.yaml",
        *[p for p in (REPO / "configs").glob("train_*.yaml")],
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            cfg = load_yaml(p)
        except Exception:  # noqa: BLE001
            continue
        if cfg.get("run_name") == run_name or p.stem == f"train_{run_name}":
            return cfg
    return {}


# ---------------------------------------------------------------------------
# Tab 1: Training Progress
# ---------------------------------------------------------------------------

def render_training_tab(run_name: str) -> None:
    run_dir = CKPT_ROOT / run_name
    step_df, eval_df = load_metrics(run_dir)
    if step_df.empty:
        st.info("No step records yet -- metrics.jsonl is empty or hasn't been written.")
        return

    steps_csv, eval_csv = mirror_csv(run_dir, step_df, eval_df)
    cfg = load_train_config(run_name)
    max_steps = cfg.get("max_steps")

    last = step_df.iloc[-1]
    n = len(step_df)
    window = step_df.iloc[max(0, n - 200):]
    if len(window) >= 2 and "t" in window and window["t"].iloc[-1] > window["t"].iloc[0]:
        tok_s = ((window["tokens_seen"].iloc[-1] - window["tokens_seen"].iloc[0])
                 / (window["t"].iloc[-1] - window["t"].iloc[0]))
    else:
        tok_s = float("nan")
    elapsed_s = (last["t"] - step_df["t"].iloc[0]) if "t" in step_df else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Step", f"{int(last['step']):,}" + (f" / {max_steps:,}" if max_steps else ""))
    c2.metric("Tokens seen", f"{int(last['tokens_seen']):,}")
    c3.metric("Loss (last)", f"{last['loss']:.4f}")
    c4.metric("Throughput (last 200 steps)", f"{tok_s:,.0f} tok/s" if tok_s == tok_s else "n/a")
    if elapsed_s == elapsed_s:
        h = elapsed_s / 3600
        eta_txt = "n/a"
        if max_steps and tok_s == tok_s and tok_s > 0 and "seq_len" in last:
            steps_left = max_steps - int(last["step"])
            tok_per_step = last["tokens_seen"] / max(1, int(last["step"]) + 1)
            eta_h = (steps_left * tok_per_step) / tok_s / 3600
            eta_txt = f"~{eta_h:.1f} h left"
        c5.metric("Elapsed", f"{h:.1f} h", eta_txt)
    else:
        c5.metric("Elapsed", "n/a")

    if not eval_df.empty:
        best = eval_df.loc[eval_df["val_loss"].idxmin()]
        st.caption(
            f"Best val_loss so far: **{best['val_loss']:.4f}** at step {int(best['step']):,} "
            f"-> saved as `{run_dir / 'best.pt'}`"
        )

    st.subheader("Loss")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=step_df["step"], y=step_df["loss"], mode="lines",
                               name="train loss", line={"width": 1, "color": "rgba(31,119,180,0.35)"}))
    if n >= 20:
        roll = step_df["loss"].rolling(max(5, n // 200), min_periods=1).mean()
        fig.add_trace(go.Scattergl(x=step_df["step"], y=roll, mode="lines",
                                   name="train loss (smoothed)", line={"width": 2, "color": "#1f77b4"}))
    if not eval_df.empty:
        fig.add_trace(go.Scatter(x=eval_df["step"], y=eval_df["val_loss"], mode="lines+markers",
                                 name="val loss", line={"width": 2, "color": "#d62728"}))
        for col in eval_df.columns:
            if col.startswith("val_") and col != "val_loss":
                fig.add_trace(go.Scatter(x=eval_df["step"], y=eval_df[col], mode="lines+markers",
                                         name=col, line={"dash": "dot"}))
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis_title="step", yaxis_title="loss", legend={"orientation": "h"})
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Gradient norm")
        f2 = go.Figure(go.Scattergl(x=step_df["step"], y=step_df["grad_norm"], mode="lines",
                                    line={"width": 1}))
        f2.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(f2, use_container_width=True)
    with col_b:
        st.subheader("Learning rate")
        f3 = go.Figure(go.Scattergl(x=step_df["step"], y=step_df["lr"], mode="lines",
                                    line={"width": 1, "color": "#2ca02c"}))
        f3.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(f3, use_container_width=True)

    if "seq_len" in step_df:
        with st.expander("Context-length curriculum"):
            f4 = go.Figure(go.Scattergl(x=step_df["step"], y=step_df["seq_len"], mode="lines",
                                        line_shape="hv"))
            f4.update_layout(height=200, margin={"l": 10, "r": 10, "t": 10, "b": 10})
            st.plotly_chart(f4, use_container_width=True)

    st.subheader("Step records")
    st.dataframe(step_df.tail(200).iloc[::-1], use_container_width=True, height=280)
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download metrics.csv (all steps)", steps_csv.read_bytes(),
                        file_name=f"{run_name}_metrics.csv")
    if eval_csv.exists():
        dl2.download_button("Download eval.csv", eval_csv.read_bytes(),
                            file_name=f"{run_name}_eval.csv")
    st.caption(f"CSV mirrors kept in sync at `{steps_csv}` and `{eval_csv}`.")


# ---------------------------------------------------------------------------
# Tab 2: Try the Model (qualitative only)
# ---------------------------------------------------------------------------

def list_checkpoints(run_dir: Path) -> list[str]:
    return sorted((p.name for p in run_dir.glob("*.pt")),
                 key=lambda n: (n != "best.pt", n))


@st.cache_resource(show_spinner="Loading checkpoint...")
def _load_model(run_name: str, ckpt_name: str, device: str):
    from food_slm.eval.generate import load_model_for_inference
    from food_slm.model import ModelConfig

    run_dir = CKPT_ROOT / run_name
    cfg = load_train_config(run_name)
    model_cfg_path = REPO / cfg["model_config"] if cfg.get("model_config") else None
    if model_cfg_path is None or not model_cfg_path.exists():
        raise FileNotFoundError(
            "no model_config found in configs/train_*.yaml for this run -- "
            "check load_train_config()'s match against the run_name")
    tok_path = REPO / cfg["tokenizer"] if cfg.get("tokenizer") \
        else REPO / "artifacts" / "tokenizers" / "corpus-16k"
    ModelConfig.from_yaml(model_cfg_path)  # fail fast with a clear error if the yaml is bad
    return load_model_for_inference(model_cfg_path, tok_path, run_dir / ckpt_name, device)


def render_try_tab(run_name: str) -> None:
    st.caption(
        "Qualitative spot-check only, for human satisfaction -- **the Training "
        "Progress tab's numbers are the ground truth**, not what the model says here."
    )
    run_dir = CKPT_ROOT / run_name
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        st.info("No .pt checkpoints in this run yet.")
        return

    import torch

    col1, col2, col3 = st.columns([2, 1, 1])
    ckpt_name = col1.selectbox("Checkpoint", ckpts)
    device_opts = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = col2.selectbox("Device", device_opts,
                            help="CPU is the safe default while training is running on the GPU.")
    if device == "cuda":
        st.warning(
            "Training is likely using the GPU right now. Loading the model here too will "
            "compete for the same VRAM / power budget and can slow the training run."
        )
    max_new = col3.slider("Max new tokens", 16, 300, 100)

    c4, c5, c6 = st.columns(3)
    temperature = c4.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    top_k = c5.slider("Top-k", 0, 200, 50)
    top_p = c6.slider("Top-p", 0.0, 1.0, 0.95, 0.05)

    prompt = st.text_area("Prompt", "Idli is a", height=100)

    if st.button("Generate", type="primary"):
        with st.spinner(f"Loading {ckpt_name} on {device} and generating..."):
            model, tokenizer, ckpt_step = _load_model(run_name, ckpt_name, device)
            from food_slm.eval.generate import generate_text
            t0 = time.time()
            out = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_new, temperature=temperature,
                top_k=top_k or None, top_p=top_p or None,
            )
            dt = time.time() - t0
        st.text_area("Output", out, height=200)
        st.caption(f"Checkpoint at step {ckpt_step:,} -- generated in {dt:.1f}s on {device}.")


# ---------------------------------------------------------------------------
# Tab 3: Recipe Extraction (canon_extract_recipes.py progress + judge checklist)
# ---------------------------------------------------------------------------

# validate.py's RecordVerdict.apply_to() stamps "{provider}:{decision}" reasons
# (in this exact spelling, joined "; ") into quality.notes -- "qwen7b"/"gemma9b"
# from the two-model panel, "tiebreak" only when the panel split AND
# corroboration >= 2 (see validate.validate_records's comment on why it's
# skipped otherwise: a skipped tiebreak can't change the outcome, so there's
# no point burning ~6 tok/s deepseek-r1:32b CPU time on it).
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


def render_extraction_tab() -> None:
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


# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Food SLM -- Unified Dashboard")
    runs = discover_runs()
    # Missing training runs no longer hard-stops the whole app: this dashboard
    # also hosts Recipe Extraction and Golden QA Review, which don't depend on
    # a training run existing, so they should stay usable either way.
    run_name = None

    with st.sidebar:
        if runs:
            run_name = st.selectbox("Run", runs, index=len(runs) - 1)
        else:
            st.caption(f"No runs found under `{CKPT_ROOT}`.")
        auto = st.checkbox("Auto-refresh", value=True)
        interval = st.slider("Refresh every (s)", 5, 60, 15) if auto else None
        if st.button("Refresh now"):
            st.rerun()

    # A blocking time.sleep() + st.rerun() loop in the main script body races
    # Streamlit's own session teardown (AttributeError: 'AppSession' object has
    # no attribute '_scriptrunner') -- st.fragment(run_every=...) is the native,
    # race-free way to auto-rerun just this tab on a timer. Only the read-only
    # monitoring tabs auto-refresh; interactive tabs (Try the Model, Golden QA
    # Review) would otherwise reset mid-interaction.
    @st.fragment(run_every=interval if auto else None)
    def _live_training_tab() -> None:
        if run_name:
            render_training_tab(run_name)
        else:
            st.info(f"No runs found under `{CKPT_ROOT}` (looking for a `metrics.jsonl` in each).")

    @st.fragment(run_every=interval if auto else None)
    def _live_extraction_tab() -> None:
        render_extraction_tab()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Training Progress", "Try the Model", "Recipe Extraction", "Golden QA Review"])
    with tab1:
        _live_training_tab()
    with tab2:
        if run_name:
            render_try_tab(run_name)
        else:
            st.info("No checkpoints available to try.")
    with tab3:
        _live_extraction_tab()
    with tab4:
        golden_qa_review.render_review_page()


if __name__ == "__main__":
    main()
