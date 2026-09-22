"""Query-testing interface with live, per-token confidence visualization.

    streamlit run scripts/inference_ui.py

Separate from scripts/dashboard.py's "Try the Model" tab: this one is built around
*watching a response form*, not just reading the final text. Each token is shown as
soon as it's generated, colour-coded by how confident the model actually was when it
picked it (green = confident, red = the model was essentially guessing), with a
token-by-token breakdown of the alternatives it considered underneath.

Qualitative / human-satisfaction tool only, same as the dashboard's try-it tab --
it says nothing about model quality that the numerical eval loss doesn't say better.
Defaults to CPU so it doesn't compete with an active training run for GPU/VRAM.
"""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "artifacts" / "checkpoints"

st.set_page_config(page_title="Food SLM Query Tester", layout="wide")

EXAMPLE_PROMPTS = [
    "Idli is a",
    "To make Sambar, you need",
    "Dosa is traditionally served with",
    "The main ingredients in Rasam are",
    "Pongal is a dish from",
]


def discover_runs() -> list[str]:
    if not CKPT_ROOT.exists():
        return []
    return sorted(p.name for p in CKPT_ROOT.iterdir()
                 if p.is_dir() and (p / "metrics.jsonl").exists())


def list_checkpoints(run_dir: Path) -> list[str]:
    return sorted((p.name for p in run_dir.glob("*.pt")),
                 key=lambda n: (n != "best.pt", n))


def load_train_config(run_name: str) -> dict:
    from food_slm.utils import load_yaml

    for p in (REPO / "configs").glob("train_*.yaml"):
        try:
            cfg = load_yaml(p)
        except Exception:  # noqa: BLE001
            continue
        if cfg.get("run_name") == run_name or p.stem == f"train_{run_name}":
            return cfg
    return {}


@st.cache_resource(show_spinner="Loading checkpoint...")
def _load_model(run_name: str, ckpt_name: str, device: str):
    from food_slm.eval.generate import load_model_for_inference

    cfg = load_train_config(run_name)
    if not cfg.get("model_config"):
        raise FileNotFoundError(f"no configs/train_*.yaml matches run_name={run_name!r}")
    model_cfg_path = REPO / cfg["model_config"]
    tok_path = REPO / cfg["tokenizer"] if cfg.get("tokenizer") \
        else REPO / "artifacts" / "tokenizers" / "corpus-16k"
    return load_model_for_inference(model_cfg_path, tok_path,
                                    CKPT_ROOT / run_name / ckpt_name, device)


def _confidence_color(prob: float) -> str:
    """Green (confident) -> amber -> red (the model was essentially guessing)."""
    if prob >= 0.7:
        return "#c8f0c8"
    if prob >= 0.4:
        return "#fdeeb0"
    if prob >= 0.15:
        return "#fbd9b0"
    return "#f7b8b8"


def _token_span(text: str, prob: float) -> str:
    safe = html.escape(text).replace("\n", "<br>")
    color = _confidence_color(prob)
    return (f'<span title="p={prob:.3f}" style="background-color:{color}; '
           f'border-radius:3px; padding:1px 0;">{safe}</span>')


def main() -> None:
    st.title("Food SLM -- Query Tester")
    st.caption(
        "Qualitative check only, for human satisfaction -- the training dashboard's "
        "numerical loss is the real signal, not what the model says here. Each token "
        "below is colour-coded by the model's own confidence when it picked it: "
        "green = confident, red = essentially guessing."
    )

    runs = discover_runs()
    if not runs:
        st.warning(f"No runs found under `{CKPT_ROOT}`.")
        st.stop()

    with st.sidebar:
        run_name = st.selectbox("Run", runs, index=len(runs) - 1)
        run_dir = CKPT_ROOT / run_name
        ckpts = list_checkpoints(run_dir)
        if not ckpts:
            st.warning("No checkpoints in this run yet.")
            st.stop()
        ckpt_name = st.selectbox("Checkpoint", ckpts)

        import torch
        device_opts = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        device = st.selectbox("Device", device_opts,
                              help="CPU is the safe default while training uses the GPU.")
        if device == "cuda":
            st.warning(
                "Training likely has the GPU right now -- loading here too competes "
                "for the same VRAM/power budget and can slow the training run."
            )

        st.divider()
        max_new = st.slider("Max new tokens", 10, 200, 60)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
        top_k = st.slider("Top-k", 0, 200, 50)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.05)

    st.subheader("Try a prompt")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for c, ex in zip(cols, EXAMPLE_PROMPTS, strict=True):
        if c.button(ex, use_container_width=True):
            st.session_state["prompt_box"] = ex

    prompt = st.text_area("Prompt", key="prompt_box", height=80)
    go = st.button("Generate", type="primary")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if go and prompt.strip():
        from food_slm.eval.generate import generate_stream

        model, tokenizer, ckpt_step = _load_model(run_name, ckpt_name, device)
        placeholder = st.empty()
        detail_rows = []
        rendered = html.escape(prompt)
        t0 = time.time()

        for i, ts in enumerate(generate_stream(
            model, tokenizer, prompt,
            max_new_tokens=max_new, temperature=temperature,
            top_k=top_k or None, top_p=top_p or None,
        )):
            rendered += _token_span(ts.text, ts.prob)
            placeholder.markdown(
                f'<div style="font-size:1.1rem; line-height:1.9;">{rendered}</div>',
                unsafe_allow_html=True,
            )
            detail_rows.append({
                "#": i, "token": ts.text, "prob": round(ts.prob, 4),
                "entropy": round(ts.entropy, 3),
                "top alternatives": ", ".join(f"{t!r} ({p:.2f})" for t, p in ts.top_alternatives),
            })
        dt = time.time() - t0

        if detail_rows:
            avg_prob = sum(r["prob"] for r in detail_rows) / len(detail_rows)
            avg_ent = sum(r["entropy"] for r in detail_rows) / len(detail_rows)
            least_confident = min(detail_rows, key=lambda r: r["prob"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tokens generated", len(detail_rows))
            c2.metric("Avg. confidence", f"{avg_prob:.1%}")
            c3.metric("Avg. entropy (nats)", f"{avg_ent:.2f}")
            c4.metric("Least confident token",
                     f"{least_confident['token']!r} ({least_confident['prob']:.1%})")
            st.caption(f"Checkpoint step {ckpt_step:,} -- {dt:.1f}s on {device} "
                      f"({len(detail_rows) / max(dt, 1e-6):.1f} tok/s)")

            with st.expander("Per-token breakdown"):
                st.dataframe(detail_rows, use_container_width=True, height=300)

            st.session_state["history"].insert(0, {
                "run": run_name, "checkpoint": ckpt_name, "step": ckpt_step,
                "prompt": prompt, "response": "".join(r["token"] for r in detail_rows),
                "avg_confidence": round(avg_prob, 4), "avg_entropy": round(avg_ent, 3),
                "tokens": len(detail_rows), "seconds": round(dt, 2),
            })

    if st.session_state["history"]:
        st.divider()
        st.subheader("Session history")
        st.dataframe(st.session_state["history"], use_container_width=True, height=250)
        import json
        st.download_button(
            "Download session history (JSON)",
            json.dumps(st.session_state["history"], indent=2, ensure_ascii=False),
            file_name="query_test_session.json",
        )


if __name__ == "__main__":
    main()
