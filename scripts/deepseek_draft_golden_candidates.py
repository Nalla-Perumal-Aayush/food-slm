"""Drafts golden-QA answers for the 700-question backlog using the local
deepseek-r1:32b model (via Ollama), per explicit owner instruction: skip source
sourcing/citation for this batch, draft directly from the model's own knowledge,
and queue every single one as "pending" for individual human approval in
scripts/golden_qa_review.py.

    python scripts/deepseek_draft_golden_candidates.py              # all 700
    python scripts/deepseek_draft_golden_candidates.py --limit 10    # test a few first

IMPORTANT, and intentionally different from draft_golden_candidates.py: these
items carry NO source citation. That is a deliberate, owner-approved exception
to this project's normal golden-QA sourcing rule (evaluation_strategy.md SS5 /
Documents/Golden_QA_Contributor_Guide.pdf), made because deepseek-r1 is one of
this project's own AI judge-panel models (configs/agents.yaml's
data_validation_heavy role) -- using it to independently verify/approve golden
content would be circular, so it is used ONLY to draft a first-pass answer, and
every item still requires full individual human review (no sampling, no
auto-approval) before it can count as golden. golden_qa_review.py visibly warns
on any non-abstention item with no source, specifically so that review step
isn't skipped or treated as a formality.

Input: Documents/golden_qa_700_draft_candidates.jsonl (the pre-existing
backlog's question/task_type/region/diet/dish_ids -- its own placeholder
candidate_answer and source fields are discarded, not used as truth).
Output: appended to data/eval/golden_qa_candidates.jsonl, same schema as
draft_golden_candidates.py's output, dedup by question text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# See canon_extract_recipes.py's identical fix: stdout redirected to a log file on
# Windows defaults to cp1252, and a dish name with an unencodable character crashes
# every print() and kills the whole process, not just that one item.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.llm import ProviderSpec, build_provider  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BACKLOG_PATH = REPO / "Documents" / "golden_qa_700_draft_candidates.jsonl"
CANDIDATES_PATH = REPO / "data" / "eval" / "golden_qa_candidates.jsonl"

MODEL_LABEL = "deepseek-r1:32b (local Ollama) -- no source citation, drafted directly " \
             "from model knowledge per owner instruction 2026-10-08; requires careful " \
             "human fact-check on approval, not just a style/tone read"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_STYLE_BY_TASK = {
    "recipe_request": "List the main ingredients, then the method as a short numbered or "
                      "flowing sequence. 3-6 sentences total.",
    "scaling": "Explain how to scale the dish's ingredients for a different serving count, "
              "and explicitly note that time/salt/spice don't scale linearly. 2-4 sentences.",
    "ingredient_substitution": "Name the substitute, the rough ratio if there is one, where "
                               "it works best, and any caveat. 2-3 sentences.",
    "troubleshooting": "State the likely cause, the fix, and (if relevant) how to prevent it "
                       "next time. 2-4 sentences.",
    "technique_explanation": "Explain what the technique is, how it's done, and how you know "
                             "it's done right (a sensory cue). 2-4 sentences.",
    "regional_comparison": "Compare the two things dimension by dimension (ingredients, "
                           "method, texture, or name) rather than just describing each "
                           "separately. 2-4 sentences.",
    "definition": "A concise, direct factual answer. 1-3 sentences.",
    "abstention": "Politely decline -- state that this is outside South Indian cuisine "
                  "(or outside what you can help with) without guessing. 1-2 sentences.",
}

_SYSTEM = """You are drafting a candidate answer for a South Indian cuisine knowledge base.
Answer factually and concisely in the style described for the task type. Do not hedge with
phrases like "I believe" or "it's possible" -- state it plainly, as you understand it to be
true, since a human will fact-check this before it's used. Never invent an oddly specific
number (an exact minute count, an exact gram weight) unless you're confident of it -- prefer
a general description over a fabricated precise figure. Reply with ONLY the answer text, no
preamble, no "Answer:" prefix, no markdown formatting."""


def _build_prompt(row: dict) -> str:
    style = _STYLE_BY_TASK.get(row["task_type"], "Answer concisely and factually.")
    region = (row.get("region") or "").replace("_", " ").title() or "South India generally"
    diet = row.get("diet") or "na"
    return (f"TASK TYPE: {row['task_type']}\nSTYLE: {style}\nREGION: {region}\nDIET: {diet}\n\n"
           f"QUESTION: {row['question']}\n\nWrite the answer.")


def _clean(text: str) -> str:
    return _THINK.sub("", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max items this run (0 = all)")
    args = ap.parse_args()

    backlog = [json.loads(line) for line in BACKLOG_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    existing_questions = set()
    if CANDIDATES_PATH.exists():
        for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_questions.add(json.loads(line)["question"])

    todo = [r for r in backlog if r["question"] not in existing_questions]
    print(f"{len(backlog)} backlog items, {len(backlog) - len(todo)} already drafted, "
         f"{len(todo)} to process")
    if args.limit:
        todo = todo[:args.limit]
        print(f"--limit {args.limit}: processing {len(todo)} this run")

    provider = build_provider(ProviderSpec(
        name="deepseek-r1-local", kind="ollama", model="deepseek-r1:32b",
        family="deepseek", local=True, max_tokens=800))

    import hashlib

    done = 0
    errors = 0
    t0 = time.time()
    with CANDIDATES_PATH.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo):
            print(f"[{i + 1}/{len(todo)}] {row['task_type']:24s} {row['question'][:60]}")
            try:
                raw = provider.complete(_SYSTEM, _build_prompt(row), temperature=0.3)
                answer = _clean(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR {type(exc).__name__}: {str(exc)[:200]}")
                errors += 1
                continue
            if not answer:
                print("    empty response, skipping")
                errors += 1
                continue

            out = {
                "task_type": row["task_type"], "region": row.get("region"),
                "diet": row.get("diet", "na"), "question": row["question"],
                "candidate_answer": answer,
                "source": {"url": "", "quote": ""},
                "dish_ids": row.get("dish_ids", []),
                "status": "pending", "reviewer_notes": "",
                "drafted_by": MODEL_LABEL,
            }
            digest = hashlib.sha256(out["question"].encode()).hexdigest()[:8]
            out["id"] = f"golden:{digest}"
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            elapsed = time.time() - t0
            print(f"    done ({elapsed / done:.1f}s/item avg, {done} done, {errors} errors)")

    print(f"\n=== {done} drafted, {errors} errors, appended to {CANDIDATES_PATH} ===")
    print("Every item is status=pending -- review and approve individually at "
         "streamlit run scripts/golden_qa_review.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
