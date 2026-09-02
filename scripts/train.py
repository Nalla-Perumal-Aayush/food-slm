"""Run a training loop from a train config.

    python scripts/train.py --config configs/train_poc.yaml
    python scripts/train.py --config configs/train_poc.yaml --max-steps 200
    python scripts/train.py --config configs/train_poc.yaml --resume auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train import train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "train_poc.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for the latest")
    args = ap.parse_args()

    res = train(args.config, max_steps=args.max_steps, resume=args.resume, repo_root=REPO)
    print("\n--- done ---")
    print(json.dumps({
        "final_step": res.final_step,
        "final_loss": round(res.final_loss, 5),
        "best_val": res.best_val,
        "eval_history": res.eval_history[-5:],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
