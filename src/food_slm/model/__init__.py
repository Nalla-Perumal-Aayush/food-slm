"""Config A decoder-only Transformer (plain PyTorch).

One implementation, parameterized by configs/model_*.yaml. The 15M PoC and the
89M release model run the exact same code path.
"""
from food_slm.model.config import ModelConfig
from food_slm.model.transformer import FoodSLM, ModelOutput, build_model

__all__ = ["ModelConfig", "FoodSLM", "ModelOutput", "build_model"]
