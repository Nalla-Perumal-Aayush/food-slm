"""Run the SFT training loop from an SFT config.

    python scripts/train_sft.py --config configs/sft_150m.yaml
    python scripts/train_sft.py --config configs/sft_150m.yaml --resume auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.train.sft_loop import train_sft  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "sft_150m.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for the latest")
    args = ap.parse_args()

    res = train_sft(args.config, resume=args.resume, repo_root=REPO)
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
