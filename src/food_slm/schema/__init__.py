"""Canonical knowledge-store schema (baseline §3, Phase 5).

The canonical store is separate linked JSONL record types under data/canonical/.
Every record separates: CONTENT + SEMANTIC STRUCTURE + PROVENANCE + QUALITY + VERSION.
It is the sole grounding source for agent-generated synthetic data (baseline §8.2).

STATUS: schema v0 stub. The full 20-field configuration (Phase 5) is the next
design artifact after the repo scaffold.
"""
from food_slm.schema.models import (
    Dish,
    Ingredient,
    Provenance,
    Quality,
    Recipe,
    Source,
)

__all__ = ["Dish", "Ingredient", "Recipe", "Source", "Provenance", "Quality"]
