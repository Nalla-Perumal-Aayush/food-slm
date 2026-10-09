"""Small SFT LR/epoch ablation -- picks a real peak LR and epoch count for
configs/sft_150m.yaml, same discipline pretraining's own LR went through before
being locked in (not asserted from instinct). See food_slm.train.sft_loop.

For each candidate LR, runs a FRESH 12-epoch run from the same pretrained
checkpoint (never carrying state between candidates), with frequent eval so the
val_loss curve is well-resolved -- the goal is the curve's shape: does it
decrease stably, where does it bottom out (best epoch for that LR), and does it
clearly start rising again after (overfitting) within the budget. Only `best.pt`
is kept per candidate (keep_last_n=1) -- these are throwaway ablation runs, not
candidates for the real release checkpoint, and the full step_*.pt series isn't
needed once the eval_history curve is extracted.

    python scripts/sft_lr_ablation.py
    python scripts/sft_lr_ablation.py --lrs 1e-5 3e-5                 # subset
    python scripts/sft_lr_ablation.py --epochs 8                      # shorter budget

Writes a result summary to artifacts/reports/sft_lr_ablation.json; deletes each
candidate's checkpoint directory after extracting its metrics (keeps only the
summary, not the weights -- the real run gets launched fresh with the winning
hyperparameters, not resumed from an ablation candidate).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Same fix as canon_extract_recipes.py etc.: stdout redirected to a log file on
# Windows defaults to cp1252, and this runs for hours unattended.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train.sft_loop import train_sft  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "reports" / "sft_lr_ablation.json"
STEPS_PER_EPOCH = 86  # from the real pack at micro_batch=8, grad_accum=4 (2,778 train examples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5, 6e-5, 1.2e-4])
    ap.add_argument("--epochs", type=float, default=12.0)
    ap.add_argument("--eval-every-steps", type=int, default=15)
    args = ap.parse_args()

    results = []
    for lr in args.lrs:
        ckpt_dir = f"artifacts/checkpoints/_sft_ablation_lr{lr:.0e}"
        print(f"\n=== LR {lr:.1e} -- {args.epochs} epochs ({int(args.epochs * STEPS_PER_EPOCH)} steps) ===")
        t0 = time.time()
        res = train_sft("configs/sft_150m.yaml", repo_root=REPO, overrides={
            "lr": lr, "epochs": args.epochs, "checkpoint_dir": ckpt_dir,
            "eval_every_steps": args.eval_every_steps,
            "checkpoint_every_steps": int(args.epochs * STEPS_PER_EPOCH),  # ~once, at the end
            "keep_last_n": 1, "log_every_steps": 20,
        })
        dt = time.time() - t0

        best_entry = min(res.eval_history, key=lambda e: e["val_loss"]) if res.eval_history else None
        result = {
            "lr": lr, "epochs_run": args.epochs, "final_step": res.final_step,
            "final_train_loss": res.final_loss, "best_val_loss": res.best_val,
            "best_val_step": best_entry["step"] if best_entry else None,
            "best_val_epoch": round(best_entry["step"] / STEPS_PER_EPOCH, 2) if best_entry else None,
            "eval_history": res.eval_history, "wall_time_s": round(dt, 1),
            "diverged": any(e["step"] > 0 and not (e["val_loss"] == e["val_loss"])  # NaN check
                           for e in res.eval_history),
        }
        results.append(result)
        print(f"  best_val_loss={result['best_val_loss']:.4f} at step {result['best_val_step']} "
             f"(epoch {result['best_val_epoch']}), wall_time={dt:.0f}s")

        shutil.rmtree(REPO / ckpt_dir, ignore_errors=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for r in results:
        print(f"  lr={r['lr']:.1e}: best_val_loss={r['best_val_loss']:.4f} "
             f"at epoch {r['best_val_epoch']}  (final train_loss={r['final_train_loss']:.4f})")
    winner = min(results, key=lambda r: r["best_val_loss"])
    print(f"\nbest: lr={winner['lr']:.1e}, epoch={winner['best_val_epoch']}, "
         f"val_loss={winner['best_val_loss']:.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
