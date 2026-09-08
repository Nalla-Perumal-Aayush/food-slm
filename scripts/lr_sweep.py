"""Learning-rate sweep for the 40M rung (plan_40m §G1).

The 15M LR (6e-4) will not transfer (the config is not muP-parameterised), so pick the
40M peak LR empirically: run a short slice of the real schedule at each candidate and
compare the loss reached.

    python scripts/lr_sweep.py --config configs/train_40m.yaml \
        --lrs 3e-4 6e-4 1e-3 2e-3 --steps 1500

Each run uses the real warmup_ratio and cosine shape (total = --steps), an isolated
checkpoint dir, and the first curriculum phase only (short seq_len -> fast). Writes
artifacts/reports/lr_sweep_<date>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train import train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "train_40m.yaml")
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 6e-4, 1e-3, 2e-3])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", type=Path,
                    default=REPO / "artifacts" / "checkpoints" / "_lr_sweep")
    args = ap.parse_args()

    results = []
    for lr in args.lrs:
        ck = args.out / f"lr_{lr:.0e}".replace("-", "m")
        print(f"\n=== lr {lr:.2e}  ({args.steps} steps) ===")
        res = train(args.config, max_steps=args.steps, repo_root=REPO,
                    checkpoint_dir=ck, overrides={"lr": lr})
        losses = [h["loss"] for h in res.history]
        tail = losses[-50:] if len(losses) >= 50 else losses
        diverged = any(x != x or x > 20 for x in losses)  # noqa: PLR0124 - x!=x is NaN test
        spikes = sum(1 for a, b in zip(tail, tail[1:], strict=False) if b > 1.5 * a)
        results.append({
            "lr": lr,
            "final_loss": round(res.final_loss, 5),
            "mean_tail_loss": round(sum(tail) / len(tail), 5),
            "best_val": round(res.best_val, 5) if res.best_val else None,
            "diverged": diverged,
            "tail_spikes": spikes,
        })
        print(f"  final {res.final_loss:.4f}  tail-mean {results[-1]['mean_tail_loss']:.4f}  "
              f"best_val {results[-1]['best_val']}  diverged={diverged}  spikes={spikes}")

    ok = [r for r in results if not r["diverged"]]
    best = min(ok, key=lambda r: r["mean_tail_loss"]) if ok else None
    report = {"created": datetime.now(UTC).isoformat(timespec="seconds"),
              "config": str(args.config), "steps": args.steps,
              "results": results, "recommended_lr": best["lr"] if best else None}
    rp = REPO / "artifacts" / "reports" / f"lr_sweep_{datetime.now(UTC):%Y-%m-%d}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'lr':>10} {'tail-mean':>11} {'best_val':>10} {'diverged':>9} {'spikes':>7}")
    for r in results:
        print(f"{r['lr']:>10.2e} {r['mean_tail_loss']:>11} {str(r['best_val']):>10} "
              f"{str(r['diverged']):>9} {r['tail_spikes']:>7}")
    print(f"\nrecommended peak LR: {report['recommended_lr']}")
    print(f"report -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
