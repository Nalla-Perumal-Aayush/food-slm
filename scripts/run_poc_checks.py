"""15M Proof-of-Concept acceptance checks (baseline §12, configs/train_poc.yaml).

  1. runs_end_to_end   — a short train run completes and the loss decreases
  2. resume_bit_exact  — interrupt at step N/2, resume, deterministic-mode loss tail
                         is bit-identical to an uninterrupted run
  3. samples_are_english — generate a continuation from a prompt (manual eyeball)

    python scripts/run_poc_checks.py --steps 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.eval.generate import generate_text  # noqa: E402
from food_slm.model import FoodSLM, ModelConfig  # noqa: E402
from food_slm.tokenizer import load_tokenizer  # noqa: E402
from food_slm.train import TrainConfig, load_checkpoint, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "train_poc.yaml")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    ok = True

    # 1 + 2: deterministic straight vs interrupted+resumed
    common = dict(repo_root=REPO, deterministic=True, device="cpu", max_steps=args.steps)
    print(f"[1] end-to-end straight run ({args.steps} steps, deterministic CPU)...")
    straight = train(args.config, **{**common, "resume": None})

    half = args.steps // 2
    print(f"[2] interrupt at step {half}, resume...")
    train(args.config, **{**common, "resume": None}, stop_after=half)
    resumed = train(args.config, **{**common, "resume": "auto"})

    runs_ok = straight.final_step == args.steps and straight.history[-1]["loss"] < \
        straight.history[0]["loss"]
    tail_s = [h["loss"] for h in straight.history if h["step"] >= half]
    tail_r = [h["loss"] for h in resumed.history]
    resume_ok = tail_s == tail_r and len(tail_r) == args.steps - half
    print(f"    runs_end_to_end : {'PASS' if runs_ok else 'FAIL'} "
          f"(loss {straight.history[0]['loss']:.3f} -> {straight.history[-1]['loss']:.3f})")
    print(f"    resume_bit_exact: {'PASS' if resume_ok else 'FAIL'}")
    ok &= runs_ok and resume_ok

    # 3: sample generation
    print("[3] sample generation...")
    model_cfg = ModelConfig.from_yaml(REPO / cfg.model_config)
    model = FoodSLM(model_cfg)
    ck = sorted((REPO / cfg.checkpoint_dir).glob("step_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]))[-1]
    load_checkpoint(ck, model=model, restore_rng=False)
    tok = load_tokenizer(REPO / cfg.tokenizer)
    for prompt in ("The recipe for", "In South India, "):
        txt = generate_text(model, tok, prompt, max_new_tokens=40, temperature=0.8,
                            eos_id=tok.token_to_id("<|eos|>"))
        print(f"    {prompt!r} -> {txt!r}")
    print("    samples_are_english: MANUAL EYEBALL "
          "(a 15M model on a tiny sample will be barely coherent — that is expected)")

    print(f"\n{'ALL AUTOMATED CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    _ = torch  # torch used indirectly via train / model
    raise SystemExit(main())
