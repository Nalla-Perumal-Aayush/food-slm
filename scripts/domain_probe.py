"""Run the domain probe set against a trained checkpoint (plan_40m §I4).

NOT a pass/fail gate — a structured eyeball to see whether domain specialization is
starting. A 40-60M model on ~1B tokens with a ~5% domain slice will be weak on recall;
what matters is coherent food-register text, vocabulary use, and structural-token
emission.

    python scripts/domain_probe.py --checkpoint artifacts/checkpoints/ablation-40m-001/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.eval.generate import generate_text  # noqa: E402
from food_slm.model import FoodSLM, ModelConfig  # noqa: E402
from food_slm.tokenizer import load_tokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, default=REPO / "configs" / "model_40m.yaml")
    ap.add_argument("--tokenizer", type=Path, default=REPO / "artifacts" / "tokenizers" / "corpus-16k")
    ap.add_argument("--probes", type=Path, default=REPO / "data" / "eval" / "domain_probes.jsonl")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig.from_yaml(args.model_config)
    model = FoodSLM(cfg).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    tok = load_tokenizer(args.tokenizer)
    print(f"loaded {args.checkpoint.name}  step={ck.get('step')}  "
          f"val={ck.get('best_val')}  params={model.num_params():,}\n")

    probes = [json.loads(x) for x in args.probes.read_text(encoding="utf-8").splitlines()
              if x.strip()]
    by_kind: dict[str, list] = {}
    for p in probes:
        out = generate_text(model, tok, p["prompt"], max_new_tokens=args.max_new,
                            temperature=args.temperature, top_k=40, top_p=0.95)
        cont = out[len(p["prompt"]):].strip()
        by_kind.setdefault(p["kind"], []).append((p, cont))

    for kind, items in by_kind.items():
        print(f"=== {kind} ({len(items)}) ===")
        for p, cont in items:
            print(f"  {p['prompt']!r}")
            print(f"     -> {cont!r}")
            print(f"     (want: {p['note']})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
