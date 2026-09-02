"""Print parameter accounting for each model config.

    python scripts/show_model_stats.py
"""
from __future__ import annotations

from pathlib import Path

from food_slm.model.config import ModelConfig

CONFIGS = [
    "configs/model_15m.yaml",
    "configs/model_40m.yaml",
    "configs/model_89m.yaml",
    "configs/model_150m.yaml",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in CONFIGS:
        cfg = ModelConfig.from_yaml(root / rel)
        pc = cfg.param_count()
        print(f"\n{cfg.name}  ({rel})")
        print(f"  d={cfg.hidden_dim}  L={cfg.num_layers}  heads={cfg.num_heads} "
              f"(head_dim {cfg.head_dim})  ffn={cfg.ffn_inner_dim}  ctx={cfg.max_seq_len}")
        print(f"  non-embedding : {pc['non_embedding']:>12,}")
        print(f"  embedding     : {pc['embedding']:>12,}")
        print(f"  TOTAL         : {pc['total']:>12,}  (~{pc['total']/1e6:.1f}M)")


if __name__ == "__main__":
    main()
