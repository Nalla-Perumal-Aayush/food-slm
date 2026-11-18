"""Minimal standalone inference for the South Indian Cuisine SLM on CPU
(Raspberry Pi 5 or any machine) -- no project dependencies beyond torch,
tokenizers, and pyyaml (see requirements.txt).

    python infer.py "How do I make idli?"
    python infer.py --repl                 # interactive loop

Scope: South Indian cuisine only (Tamil Nadu, Kerala, Karnataka, Andhra
Pradesh, Telangana, Puducherry; vegetarian + non-vegetarian). It's trained to
decline clearly out-of-scope questions, but currently sometimes incorrectly
declines legitimate in-domain ingredient-substitution/troubleshooting
questions too (a known, open gap, not yet fixed) -- see README.md's "Known
failure modes" section before wiring this into anything user-facing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import load_model

HERE = Path(__file__).resolve().parent

# Validated empirically against this specific checkpoint (not generic defaults) --
# see README.md's "Recommended sampling settings" for how these were chosen.
DEFAULT_GEN_KWARGS = dict(max_new_tokens=150, temperature=0.4, top_k=50, top_p=0.95,
                         repetition_penalty=1.3)


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, *, max_new_tokens: int, temperature: float,
            top_k: int | None, top_p: float | None, eos_id: int, repetition_penalty: float) -> torch.Tensor:
    import torch.nn.functional as F
    ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    max_len = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_len:])[:, -1, :].float()
        if repetition_penalty != 1.0:
            seen = torch.gather(logits, 1, ids)
            seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
            logits = logits.scatter(1, ids, seen)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                mask = cum - F.softmax(s, dim=-1) > top_p
                s = s.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, si, s)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        if (nxt == eos_id).all():
            break
    return ids


def ask(model, tokenizer: Tokenizer, question: str, **gen_kwargs) -> str:
    eos_id = tokenizer.token_to_id("<|eos|>")
    prompt = f"<|bos|><|user|>{question}<|assistant|>"
    ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long)
    out = generate(model, ids, eos_id=eos_id, **{**DEFAULT_GEN_KWARGS, **gen_kwargs})
    text = tokenizer.decode(out[0].tolist())
    return text.split("<|assistant|>", 1)[-1].split("<|eos|>")[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=None)
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--config", default=str(HERE / "model_config.yaml"))
    ap.add_argument("--weights", default=str(HERE / "model_fp16.pt"))
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer.json"))
    args = ap.parse_args()

    print("Loading model (CPU)...")
    model, cfg = load_model(args.config, args.weights, device="cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    print(f"Ready. {sum(p.numel() for p in model.parameters()):,} parameters.\n")

    if args.repl:
        print("Type a question (South Indian cuisine only), or 'quit':")
        while True:
            q = input("> ").strip()
            if q.lower() in ("quit", "exit"):
                break
            if q:
                print(ask(model, tok, q), "\n")
    elif args.question:
        print(ask(model, tok, args.question))
    else:
        for q in ["How do I make idli?", "What is tempering (tadka)?",
                 "How do I fix a leaking pipe?"]:
            print(f"Q: {q}")
            print(f"A: {ask(model, tok, q)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
