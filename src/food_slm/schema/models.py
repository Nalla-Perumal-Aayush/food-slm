"""Pydantic models for the canonical knowledge store — SCHEMA v0 (stub).

This is a deliberately minimal starting point for the 15M PoC. The authoritative
schema (entity IDs, relationship format, process/state model, measurement
normalization, constraint/outcome/troubleshooting/terminology schemas, full
provenance and versioning) is decided in Phase 5 and will replace this.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# --- cross-cutting blocks (attached to every entity record) --------------------


class Provenance(BaseModel):
    source_id: str                       # -> Source.id
    license: str                         # SPDX id or explicit statement
    acquired_on: date
    url: str | None = None
    transformations: list[str] = Field(default_factory=list)
    synthetic: bool = False
    generator: str | None = None         # set when synthetic=True
    seed_record_id: str | None = None    # canonical record the synthesis was grounded on


class Quality(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: Literal["unverified", "auto_validated", "human_validated", "rejected"]
    notes: str | None = None


class _Record(BaseModel):
    id: str                              # e.g. "dish:ven-pongal", "ing:urad-dal"
    schema_version: str = "0"
    provenance: Provenance
    quality: Quality


# --- entities -----------------------------------------------------------------


class Source(BaseModel):
    id: str
    title: str
    kind: Literal[
        "recipe_site", "cookbook", "gov_doc", "cultural_doc", "agri_doc",
        "wikipedia", "transcript", "hand_authored", "other",
    ]
    license: str
    url: str | None = None
    accessed_on: date | None = None
    notes: str | None = None


class Ingredient(_Record):
    name_en: str
    name_ta: str | None = None
    transliteration: str | None = None
    aliases: list[str] = Field(default_factory=list)
    category: str | None = None          # dal, spice, vegetable, ...
    culinary_role: list[str] = Field(default_factory=list)  # thickener, tempering, souring, ...


class Dish(_Record):
    name_en: str
    name_ta: str | None = None
    transliteration: str | None = None
    aliases: list[str] = Field(default_factory=list)
    category: str | None = None          # tiffin, kuzhambu, kootu, sweet, ...
    region: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None


class RecipeIngredient(BaseModel):
    ingredient_id: str                   # -> Ingredient.id
    quantity: float | None = None
    unit: str | None = None              # g, kg, ml, tsp, tbsp, cup, count, "to taste"
    note: str | None = None


class RecipeStep(BaseModel):
    order: int
    text: str
    technique: str | None = None
    duration_min: float | None = None


class Recipe(_Record):
    dish_id: str                         # -> Dish.id
    servings: int | None = None
    ingredients: list[RecipeIngredient]
    steps: list[RecipeStep]
    equipment: list[str] = Field(default_factory=list)
    total_time_min: float | None = None
    variations: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
