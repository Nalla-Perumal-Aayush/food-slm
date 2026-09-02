"""15M Proof-of-Concept acceptance checks (baseline §12, configs/train_poc.yaml).

  1. runs_end_to_end   — a short train run completes and the loss decreases
  2. resume_bit_exact  — interrupt at step N/2, resume, deterministic-mode loss tail
                         is bit-identical to an uninterrupted run
  3. samples_are_english — generate a continuation from a prompt (manual eyeball;
                           a 15M model on the tiny PoC sample is barely coherent — expected)

    python scripts/run_poc_checks.py --steps 40

Runs in deterministic CPU mode (slow, ~500 tok/s) so check 2 is exact. For a real
throughput read, run `scripts/train.py` on the GPU.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
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
    scratch = Path(tempfile.mkdtemp(prefix="poc_checks_"))
    ck_straight = scratch / "straight"
    ck_resume = scratch / "resume"
    # deterministic CPU: use a tiny batch so the check finishes in minutes, not hours
    common = dict(repo_root=REPO, deterministic=True, device="cpu", max_steps=args.steps,
                  overrides={"micro_batch_size": 2, "grad_accum_steps": 1, "eval_batches": 4})
    ok = True
    try:
        print(f"[1] end-to-end straight run ({args.steps} steps, deterministic CPU)...")
        straight = train(args.config, checkpoint_dir=ck_straight, **common)

        half = args.steps // 2
        print(f"[2] interrupt at step {half}, resume to {args.steps}...")
        train(args.config, checkpoint_dir=ck_resume, stop_after=half, **common)
        resumed = train(args.config, checkpoint_dir=ck_resume, resume="auto", **common)

        runs_ok = (straight.final_step == args.steps
                   and straight.history[-1]["loss"] < straight.history[0]["loss"])
        tail_s = [h["loss"] for h in straight.history if h["step"] >= half]
        tail_r = [h["loss"] for h in resumed.history]
        resume_ok = tail_s == tail_r and len(tail_r) == args.steps - half
        print(f"    runs_end_to_end : {'PASS' if runs_ok else 'FAIL'} "
              f"(loss {straight.history[0]['loss']:.3f} -> {straight.history[-1]['loss']:.3f})")
        print(f"    resume_bit_exact: {'PASS' if resume_ok else 'FAIL'} "
              f"(straight tail == resumed tail: {tail_s == tail_r})")
        ok &= runs_ok and resume_ok

        print("[3] sample generation (expect barely-coherent English at 15M / tiny sample)...")
        model_cfg = ModelConfig.from_yaml(REPO / cfg.model_config)
        model = FoodSLM(model_cfg)
        ck = sorted((ck_straight).glob("step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))[-1]
        load_checkpoint(ck, model=model, restore_rng=False)
        tok = load_tokenizer(REPO / cfg.tokenizer)
        for prompt in ("The recipe for", "In South India, "):
            txt = generate_text(model, tok, prompt, max_new_tokens=40, temperature=0.8,
                                eos_id=tok.token_to_id("<|eos|>"))
            print(f"    {prompt!r} -> {txt!r}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{'ALL AUTOMATED CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    _ = torch  # torch used indirectly via train / model
    raise SystemExit(main())
