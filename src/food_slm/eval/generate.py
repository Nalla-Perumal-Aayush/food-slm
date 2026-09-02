"""Autoregressive sampling for qualitative checks (baseline §9).

No KV cache (PoC scale — a few hundred tokens is instant). Greedy when
``temperature == 0``, else temperature + top-k / top-p.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    eos_id: int | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    max_len = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_len:]).logits[:, -1, :].float()
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
        if eos_id is not None and (nxt == eos_id).all():
            break

    if was_training:
        model.train()
    return ids


def generate_text(model, tokenizer, prompt: str, **kw) -> str:
    ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long)
    out = generate(model, ids, **kw)[0].tolist()
    return tokenizer.decode(out)
