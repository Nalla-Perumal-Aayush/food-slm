"""Autoregressive sampling for qualitative checks (baseline §9).

No KV cache (PoC scale — a few hundred tokens is instant). Greedy when
``temperature == 0``, else temperature + top-k / top-p.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """`repetition_penalty` > 1.0 discourages re-emitting any token already in
    the sequence (prompt + generated so far) -- the standard HF-style penalty:
    a positive logit is divided by the penalty, a negative one multiplied by it,
    so already-seen tokens become less likely regardless of their original
    sign. 1.0 (default) is a no-op. Applied before temperature/top_k/top_p, to
    both the greedy and sampling paths -- this model has no KV cache, so
    degenerate repetition loops (the same phrase emitted over and over) are a
    real, observed failure mode worth having a direct lever for, independent
    of sampling settings."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    max_len = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_len:]).logits[:, -1, :].float()
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
        if eos_id is not None and (nxt == eos_id).all():
            break

    if was_training:
        model.train()
    return ids


def generate_text(model, tokenizer, prompt: str, **kw) -> str:
    ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long)
    out = generate(model, ids, **kw)[0].tolist()
    return tokenizer.decode(out)


@dataclass
class TokenStep:
    """One generated token plus what the model's own distribution looked like when it
    picked it — for a UI that wants to show *how* a response was produced, not just
    the final text."""

    token_id: int
    text: str                                  # decoded piece for just this token
    prob: float                                # probability of the chosen token (post
                                                # temperature/top-k/top-p — the distribution
                                                # actually sampled from, not the raw logits)
    entropy: float                             # nats, over that same (filtered) distribution;
                                                # ~0 = the model was certain, higher = it was
                                                # choosing among several live options
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)  # (text, prob),
                                                # highest first, chosen token included


@torch.no_grad()
def generate_stream(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    eos_id: int | None = None,
    top_n_alt: int = 5,
):
    """Same sampling as `generate`, but a generator yielding one `TokenStep` at a time
    instead of returning the full sequence at the end — lets a UI show the response
    forming live, with a confidence readout per token, rather than only the finished
    text (qualitative checks only, baseline §9 — not a substitute for the numerical
    eval loss)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device).unsqueeze(0)
    max_len = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_len:]).logits[:, -1, :].float()
        if temperature <= 0:
            probs = F.softmax(logits, dim=-1)
            nxt = logits.argmax(-1, keepdim=True)
        else:
            filt = logits / temperature
            if top_k:
                kth = torch.topk(filt, min(top_k, filt.size(-1)))[0][..., -1, None]
                filt = filt.masked_fill(filt < kth, float("-inf"))
            if top_p and top_p < 1.0:
                s, si = torch.sort(filt, descending=True)
                cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                mask = cum - F.softmax(s, dim=-1) > top_p
                s = s.masked_fill(mask, float("-inf"))
                filt = torch.full_like(filt, float("-inf")).scatter(-1, si, s)
            probs = F.softmax(filt, dim=-1)
            nxt = torch.multinomial(probs, 1)

        chosen_id = nxt.item()
        chosen_prob = probs[0, chosen_id].item()
        ent = -(probs[0] * (probs[0] + 1e-12).log()).sum().item()
        k = min(top_n_alt, probs.size(-1))
        top_vals, top_idx = probs[0].topk(k)
        alts = [(tokenizer.decode([i]), v)
               for v, i in zip(top_vals.tolist(), top_idx.tolist(), strict=True)]

        yield TokenStep(token_id=chosen_id, text=tokenizer.decode([chosen_id]),
                        prob=chosen_prob, entropy=ent, top_alternatives=alts)

        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and chosen_id == eos_id:
            break

    if was_training:
        model.train()


def load_model_for_inference(
    model_config_path: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
):
    """Load a `(model, tokenizer, step)` triple for generation only — no optimizer, no
    RNG restore. Shared by every UI that wants to run a checkpoint interactively
    (scripts/dashboard.py, scripts/inference_ui.py) so the loading logic lives in one
    place instead of being copy-pasted per app."""
    from food_slm.model import FoodSLM, ModelConfig
    from food_slm.tokenizer.train import load_tokenizer
    from food_slm.train.checkpoint import load_checkpoint

    model_cfg = ModelConfig.from_yaml(model_config_path)
    tok_path = Path(tokenizer_path)
    tokenizer = load_tokenizer(tok_path / "tokenizer.json" if (tok_path / "tokenizer.json").exists()
                               else tok_path)
    model = FoodSLM(model_cfg).to(device)
    info = load_checkpoint(checkpoint_path, model=model, optimizer=None,
                           map_location=device, restore_rng=False)
    model.eval()
    return model, tokenizer, info["step"]
