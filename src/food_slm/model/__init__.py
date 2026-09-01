"""Config A decoder-only Transformer (plain PyTorch).

One implementation, parameterized by configs/model_*.yaml. The 15M PoC and the
89M release model run the exact same code path.
"""
from food_slm.model.config import ModelConfig

__all__ = ["ModelConfig"]
