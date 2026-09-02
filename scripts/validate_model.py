"""15M-rung model validation — the model half of the Proof-of-Concept (baseline §12).

Checks, for every rung config:
  1. analytic param count == actual param count (exact)
  2. forward pass: logits shape + all-finite
  3. loss + backward: finite loss, finite grads for every parameter
  4. gradient-checkpointing produces the same loss (dropout disabled)
  5. (15M only) the model can overfit a single random batch — proves the
     forward/backward/optimizer path actually learns

Runs on CUDA (bf16 autocast) when available, else CPU (fp32).

    python scripts/validate_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.model import FoodSLM, ModelConfig  # noqa: E402
from food_slm.utils import set_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/model_15m.yaml",
    "configs/model_40m.yaml",
    "configs/model_89m.yaml",
    "configs/model_150m.yaml",  # Config B — the locked release target (Rev 6)
]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_config(path: str, device: torch.device) -> bool:
    cfg = ModelConfig.from_yaml(REPO / path)
    set_seed(1337, deterministic=False)
    model = FoodSLM(cfg).to(device)

    analytic = cfg.param_count()["total"]
    actual = model.num_params()
    ok_params = analytic == actual
    print(f"  params  analytic={analytic:,}  actual={actual:,}  "
          f"non_embed={model.num_params(non_embedding=True):,}  {'OK' if ok_params else 'MISMATCH'}")

    b, s = 2, min(128, cfg.max_seq_len)
    ids = torch.randint(0, cfg.vocab_size, (b, s), device=device)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model(ids, labels=ids)
    ok_shape = tuple(out.logits.shape) == (b, s, cfg.vocab_size)
    ok_finite = torch.isfinite(out.logits).all().item() and torch.isfinite(out.loss).item()
    print(f"  forward logits={tuple(out.logits.shape)}  loss={out.loss.item():.4f}  "
          f"ce={out.ce_loss.item():.4f}  z={out.z_loss.item():.2e}  "
          f"{'OK' if ok_shape and ok_finite else 'BAD'}")

    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    bad_grad = [n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]
    ok_grad = not missing and not bad_grad
    print(f"  backward grads-present={not missing}  grads-finite={not bad_grad}  "
          f"{'OK' if ok_grad else 'BAD'}")
    if missing:
        print(f"    no grad: {missing[:5]}")

    # gradient checkpointing equivalence (dropout off)
    cfg0 = cfg.model_copy(update={"dropout": 0.0, "attention_dropout": 0.0})
    set_seed(7, deterministic=False)
    m2 = FoodSLM(cfg0).to(device).train()
    ids2 = torch.randint(0, cfg0.vocab_size, (2, 64), device=device)
    l_plain = m2(ids2, labels=ids2).loss.item()
    m2.grad_checkpoint = True
    l_ckpt = m2(ids2, labels=ids2).loss.item()
    ok_ckpt = abs(l_plain - l_ckpt) < 1e-4
    print(f"  grad-ckpt  plain={l_plain:.6f}  ckpt={l_ckpt:.6f}  "
          f"{'OK' if ok_ckpt else 'DIFF'}")

    return ok_params and ok_shape and ok_finite and ok_grad and ok_ckpt


def check_overfit(device: torch.device) -> bool:
    cfg = ModelConfig.from_yaml(REPO / "configs/model_15m.yaml").model_copy(
        update={"dropout": 0.0, "attention_dropout": 0.0})
    set_seed(1337, deterministic=False)
    model = FoodSLM(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)

    b, s = 4, 64
    ids = torch.randint(0, cfg.vocab_size, (b, s), device=device)
    use_amp = device.type == "cuda"

    first = last = None
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model(ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 30 == 0:
            print(f"    step {step:>3}  loss {loss.item():.4f}")
    # a 15M model must drive one random batch far down — proves the learning path works.
    # (bf16 autocast has a precision floor, so we check "learned a lot", not "reached 0".)
    ok = last < 1.5 and last < 0.2 * first
    print(f"  overfit-1-batch  {first:.4f} -> {last:.4f}  {'OK' if ok else 'DID NOT LEARN'}")
    return ok


def main() -> int:
    device = _device()
    print(f"device: {device}  torch {torch.__version__}\n")
    all_ok = True
    for path in CONFIGS:
        print(f"[{path}]")
        all_ok &= check_config(path, device)
        print()
    print("[overfit sanity — 15M]")
    all_ok &= check_overfit(device)
    print(f"\n{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
