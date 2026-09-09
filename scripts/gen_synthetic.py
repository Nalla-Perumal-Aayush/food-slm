"""Generate template synthetic domain text from the canonical store (150M plan, D-series).

    python scripts/gen_synthetic.py

Writes:
  data/interim/domain_150m/synth/phrasings.txt        (pretraining prose, D1)
  data/interim/domain_150m/synth/instruction.jsonl    (SFT: Q&A + abstention, D3/D6)

Template-only (no LLM) — runs alongside training. Yield scales with the store size;
LLM expansion is a later step. Provenance (source record id) is on every SFT row.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.synth import abstention_items, phrasings, qa_pairs  # noqa: E402
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, default=REPO / "data" / "canonical")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "synth")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    store = load_store(args.canonical, strict=False)
    forbidden = load_forbidden(REPO / "data" / "eval")

    ph = list(phrasings(store))
    with (args.out / "phrasings.txt").open("w", encoding="utf-8") as fh:
        fh.write("# --- synthetic phrasings from the canonical store (D1) ---\n")
        for x in ph:
            fh.write(x["text"].rstrip() + "\n\n")
    ph_tok = sum(len(x["text"]) for x in ph) // 4

    sft = [*({"messages": [{"role": "user", "content": x["question"]},
                           {"role": "assistant", "content": x["answer"]}],
              "provenance": x["provenance"], "kind": x["kind"]}
             for x in (*qa_pairs(store), *abstention_items(forbidden)))]
    with (args.out / "instruction.jsonl").open("w", encoding="utf-8") as fh:
        for row in sft:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "records_in": len(store.records),
        "phrasings": len(ph), "phrasing_est_tokens": ph_tok,
        "sft_rows": len(sft),
        "sft_by_kind": {k: sum(1 for r in sft if r["kind"] == k)
                        for k in {r["kind"] for r in sft}},
        "note": "template synthetic (no LLM); scales with store size",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"phrasings: {len(ph):,}  (~{ph_tok:,} tok)")
    print(f"sft rows:  {len(sft):,}  {manifest['sft_by_kind']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
