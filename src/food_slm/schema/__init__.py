"""Canonical knowledge-store schema (baseline §3, Phase 5).

The canonical store is separate linked JSONL record types under ``data/canonical/``.
Every record shares the envelope in :class:`~food_slm.schema.models.Record`:
identity + CONTENT + SEMANTIC STRUCTURE + PROVENANCE + QUALITY + VERSION.
It is the sole grounding source for agent-generated synthetic data (baseline §8.2).

STATUS: schema **v2** (South Indian domain) — authoritative. Human spec:
``Documents/phase5_canonical_schema.md`` + ``Documents/master_baseline.md`` §3.1.
"""
from food_slm.schema.models import (
    DISALLOWED_COMMUNITY_TERMS,
    RECORD_FILES,
    RECORD_MODELS,
    SCHEMA_VERSION,
    Constraint,
    CuisineState,
    DietProfile,
    Dish,
    Equipment,
    Ingredient,
    LocalName,
    Measurement,
    Outcome,
    Problem,
    Provenance,
    Quality,
    Recipe,
    Record,
    Region,
    Relationship,
    Source,
    State,
    Technique,
    Term,
    Version,
    find_disallowed_term,
    parse_record,
)
from food_slm.schema.store import (
    CanonicalStore,
    ValidationError,
    load_store,
)

__all__ = [
    "SCHEMA_VERSION",
    "RECORD_MODELS",
    "RECORD_FILES",
    "DISALLOWED_COMMUNITY_TERMS",
    "Record",
    "Source",
    "Ingredient",
    "Dish",
    "Recipe",
    "Technique",
    "Equipment",
    "Region",
    "State",
    "Outcome",
    "Constraint",
    "Problem",
    "Term",
    "Relationship",
    "Measurement",
    "LocalName",
    "DietProfile",
    "CuisineState",
    "Provenance",
    "Quality",
    "Version",
    "parse_record",
    "find_disallowed_term",
    "CanonicalStore",
    "load_store",
    "ValidationError",
]
