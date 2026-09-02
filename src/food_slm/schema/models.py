"""Pydantic models for the canonical knowledge store — SCHEMA v2.

Executable form of ``Documents/phase5_canonical_schema.md`` + the v2 deltas in
``Documents/master_baseline.md`` §3.1. That document is the human-readable spec;
this module is the validator. They move together.

Every record is one line of a JSONL file under ``data/canonical/`` (one file per
``type``). Every record shares the envelope defined by :class:`Record`. Type-specific
fields live in ``content`` (human-readable prose) / ``structure`` (typed, enumerated,
normalized) blocks.

v2 changes (South Indian domain):
  - ``names[]`` (multi-language) replaces ``name_ta`` / ``transliteration`` / ``ipa``
  - ``cuisine_state`` on dish / recipe (required) and ingredient / technique (optional)
  - ``region`` gains ``level`` (state | sub_region | style)
  - ``diet`` (:class:`DietProfile`) on dish / recipe; non-veg ``ingredient.category`` values
  - ``constraint.kind`` drops ``ritual_religious`` / ``seasonal_ayurvedic``, adds ``cooking_style``
  - community / religion terms are rejected in labels and structure
  - process-step params gain ``internal_temp`` / ``doneness_cue`` (meat & seafood)
"""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "2"

# ---------------------------------------------------------------------------
# Controlled vocabularies. Mirror of configs/canonical_enums.yaml.
# ---------------------------------------------------------------------------

RecordType = Literal[
    "source", "ingredient", "dish", "recipe", "technique", "equipment", "region",
    "state", "outcome", "constraint", "problem", "term", "rel",
]

DravidianLang = Literal["ta", "ml", "kn", "te"]
LANG_SCRIPT = {"ta": "tamil", "ml": "malayalam", "kn": "kannada", "te": "telugu"}
LANG_NAME = {"ta": "Tamil", "ml": "Malayalam", "kn": "Kannada", "te": "Telugu"}

CuisineState = Literal[
    "tamil_nadu", "kerala", "karnataka", "andhra_pradesh", "telangana", "puducherry", "pan_south",
]
RegionLevel = Literal["state", "sub_region", "style"]

Dimension = Literal["mass", "volume", "count", "time", "temperature", "length", "qualitative"]
Qualitative = Literal["to_taste", "a_pinch", "a_few", "handful", "as_needed", "garnish"]
Unit = Literal[
    "mg", "g", "kg", "ml", "l", "tsp", "tbsp", "cup",
    "count", "clove", "sprig", "piece", "sec", "min", "hr", "C",
]

HeatLevel = Literal["low", "medium-low", "medium", "medium-high", "high"]
Phase = Literal[
    "prep", "soak", "grind", "ferment", "temper", "cook", "assemble", "finish", "rest", "serve",
]
ActionVerb = Literal[
    "soak", "drain", "rinse", "grind", "blend", "mix", "knead", "ferment", "rest", "heat",
    "temper", "saute", "fry", "deep_fry", "roast", "dry_roast", "boil", "simmer",
    "pressure_cook", "steam", "add", "combine", "mash", "season", "garnish", "cool", "serve",
    "reduce", "strain", "cover", "flip", "marinate", "sear", "braise", "grill",
]
Predicate = Literal[
    "used_in", "substitutes_for", "precedes", "transforms_state", "variant_of",
    "belongs_to_region", "requires_technique", "requires_equipment", "pairs_with", "part_of",
    "causes_problem", "fixes_problem", "violates_constraint", "satisfies_constraint",
    "produces_outcome", "has_alias", "derived_from", "seasonal_in",
]

# predicate -> (allowed subject prefixes, allowed object prefixes); "*" = any entity
_ANY = ("*",)
PREDICATE_TYPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "used_in": (("ing",), ("dish", "recipe")),
    "substitutes_for": (("ing",), ("ing",)),
    "precedes": (("tech",), ("tech",)),
    "transforms_state": (("tech",), ("state",)),
    "variant_of": (("dish",), ("dish",)),
    "belongs_to_region": (("dish", "ing", "tech", "region"), ("region",)),
    "requires_technique": (("dish", "recipe"), ("tech",)),
    "requires_equipment": (("dish", "recipe", "tech"), ("equip",)),
    "pairs_with": (("dish", "ing"), ("dish", "ing")),
    "part_of": (_ANY, _ANY),
    "causes_problem": (("ing", "tech", "state"), ("problem",)),
    "fixes_problem": (("tech", "ing"), ("problem",)),
    "violates_constraint": (("ing", "dish", "recipe"), ("constraint",)),
    "satisfies_constraint": (("dish", "recipe"), ("constraint",)),
    "produces_outcome": (("recipe", "tech"), ("outcome",)),
    "has_alias": (_ANY, ("term",)),
    "derived_from": (_ANY, _ANY),
    "seasonal_in": (("ing", "dish"), ("region",)),
}

_PREFIX_FOR_TYPE = {
    "source": "src", "ingredient": "ing", "dish": "dish", "recipe": "recipe",
    "technique": "tech", "equipment": "equip", "region": "region", "state": "state",
    "outcome": "outcome", "constraint": "constraint", "problem": "problem", "term": "term",
    "rel": "rel",
}

# --- community / religion terms disallowed anywhere in labels or structure (Rev 5) ----------
# Dishes are tagged by region and cooking style, never by community, caste or religion.
# This is the always-on core list; store.check_community_tags() extends the scan to `content`.
DISALLOWED_COMMUNITY_TERMS: tuple[str, ...] = (
    "brahmin", "iyer", "iyengar", "smartha", "madhwa",
    "muslim", "christian", "hindu", "islamic", "nasrani",
    "mappila", "moplah", "syrian christian",
)
_DISALLOWED_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in DISALLOWED_COMMUNITY_TERMS) + r")\b",
    re.IGNORECASE,
)


def find_disallowed_term(text: str, extra: tuple[str, ...] = ()) -> str | None:
    """Return the first community/religion term found in *text* (word-boundary), else None."""
    m = _DISALLOWED_RE.search(text)
    if m:
        return m.group(0)
    for t in extra:
        if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE):
            return t
    return None


def iter_strings(obj) -> list[str]:
    """Flatten every string value in a nested dict/list structure."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(iter_strings(v))
    elif isinstance(obj, list | tuple):
        for v in obj:
            out.extend(iter_strings(v))
    return out


# ---------------------------------------------------------------------------
# Multi-language names (v2)
# ---------------------------------------------------------------------------


class LocalName(BaseModel):
    """A name for an entity in one South Indian language.

    ``transliteration`` (popular romanization, NOT ISO 15919) is required — it is the
    join key between English text and the entity (baseline §6 / tamil_script_policy §3).
    """

    lang: DravidianLang
    native: str                       # native-script text
    transliteration: str              # popular romanization — REQUIRED
    primary: bool = False

    @model_validator(mode="after")
    def _check(self) -> LocalName:
        if not self.native.strip():
            raise ValueError("LocalName.native must be non-empty")
        if not self.transliteration.strip():
            raise ValueError("LocalName.transliteration is required (transliteration bridge)")
        return self


class _LocalNamed(BaseModel):
    """Mixin for structures that carry ``names[]``."""

    names: list[LocalName] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_names(self):
        if sum(1 for n in self.names if n.primary) > 1:
            raise ValueError("at most one name may be marked primary")
        langs = [n.lang for n in self.names]
        if len(langs) != len(set(langs)):
            raise ValueError("duplicate language in names[]")
        return self


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

CANONICAL_BASE_UNIT = {"mass": "g", "volume": "ml", "count": "count", "time": "min",
                       "temperature": "C", "length": "cm"}


class Measurement(BaseModel):
    original_text: str
    value: float | None = None
    unit: Unit | None = None
    value_canonical: float | None = None
    unit_canonical: str | None = None
    dimension: Dimension
    approximate: bool = False
    range: tuple[float, float] | None = None
    qualitative: Qualitative | None = None

    @model_validator(mode="after")
    def _check(self) -> Measurement:
        if self.dimension == "qualitative":
            if self.qualitative is None:
                raise ValueError("qualitative measurement needs a `qualitative` value")
            if self.value is not None:
                raise ValueError("qualitative measurement must not carry a numeric `value`")
        else:
            if self.value is None:
                raise ValueError(f"{self.dimension} measurement needs a numeric `value`")
            if self.qualitative is not None:
                raise ValueError("non-qualitative measurement must not set `qualitative`")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError("range lo > hi")
        return self


# ---------------------------------------------------------------------------
# Diet profile (v2)
# ---------------------------------------------------------------------------


class DietProfile(BaseModel):
    """Practical, neutral dietary classification of a dish or recipe (baseline §2.1).

    ``vegetarian`` here means strict lacto-vegetarian (no meat, fish, or egg; dairy allowed).
    """

    vegetarian: bool
    vegan: bool = False
    contains_dairy: bool = False
    contains_egg: bool = False
    contains_seafood: bool = False
    contains_poultry: bool = False
    contains_red_meat: bool = False

    @property
    def has_meat_or_fish(self) -> bool:
        return self.contains_seafood or self.contains_poultry or self.contains_red_meat

    @model_validator(mode="after")
    def _check(self) -> DietProfile:
        if self.has_meat_or_fish and self.vegetarian:
            raise ValueError("vegetarian cannot be true with a seafood/poultry/red_meat flag")
        if self.contains_egg and self.vegetarian:
            raise ValueError("set vegetarian=false when contains_egg=true (South Indian convention)")
        if self.vegan:
            if not self.vegetarian:
                raise ValueError("vegan implies vegetarian=true")
            if self.contains_dairy or self.contains_egg or self.has_meat_or_fish:
                raise ValueError("vegan cannot be true with dairy / egg / meat / seafood flags")
        return self


# ---------------------------------------------------------------------------
# Envelope blocks
# ---------------------------------------------------------------------------

ValidationStatus = Literal["unverified", "auto_validated", "human_validated", "rejected"]
RecordStatus = Literal["active", "deprecated", "merged"]


class Provenance(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    acquired_on: date | None = None
    transformations: list[str] = Field(default_factory=list)
    synthetic: bool = False
    generator: str | None = None
    seed_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Provenance:
        if self.synthetic:
            if not self.generator:
                raise ValueError("synthetic record needs `generator`")
            if not self.seed_record_ids:
                raise ValueError("synthetic record needs `seed_record_ids`")
        return self


class Quality(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    validation_status: ValidationStatus
    corroboration_count: int = Field(default=0, ge=0)
    notes: str | None = None


class Version(BaseModel):
    revision: int = Field(default=1, ge=1)
    created_on: date
    updated_on: date
    status: RecordStatus = "active"


class Record(BaseModel):
    """Common envelope shared by every canonical record."""

    id: str
    type: RecordType
    schema_version: str = SCHEMA_VERSION
    labels: list[str] = Field(min_length=1)
    slug_aliases: list[str] = Field(default_factory=list)
    content: BaseModel
    structure: BaseModel
    provenance: Provenance
    quality: Quality
    version: Version
    merged_into: str | None = None

    @model_validator(mode="after")
    def _check_envelope(self) -> Record:
        want_prefix = _PREFIX_FOR_TYPE[self.type]
        if not self.id.startswith(f"{want_prefix}:"):
            raise ValueError(f"id {self.id!r} must start with {want_prefix!r} for type {self.type!r}")
        slug = self.id.split(":", 1)[1]
        if slug != slug.lower() or not all(c.isascii() for c in slug):
            raise ValueError(f"id slug {slug!r} must be lowercase ASCII")
        if (self.version.status == "merged") != (self.merged_into is not None):
            raise ValueError("status=='merged' iff merged_into is set")
        # Rev 5: no community / religion tags in labels or structure.
        for text in [*self.labels, *self.slug_aliases, *iter_strings(self.structure.model_dump())]:
            term = find_disallowed_term(text)
            if term:
                raise ValueError(
                    f"community/religion term {term!r} not allowed in labels/structure "
                    "(Rev 5 scope) — use a regional or cooking-style tag instead"
                )
        return self


# ---------------------------------------------------------------------------
# content / structure blocks, per type
# ---------------------------------------------------------------------------

# -- source ------------------------------------------------------------------
SourceKind = Literal[
    "recipe_blog", "cookbook", "gov_doc", "agri_extension", "cultural_doc", "academic",
    "wikipedia", "video_transcript", "community_forum", "hand_authored",
]
LicenseClass = Literal[
    "public_domain", "permissive", "share_alike", "noncommercial", "unknown", "all_rights_reserved",
]
UsableFor = Literal["training", "grounding", "reference_only"]
Authority = Literal["primary", "secondary", "tertiary"]


class SourceContent(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    description: str | None = None
    notes: str | None = None


class SourceStructure(BaseModel):
    kind: SourceKind
    license: str
    license_class: LicenseClass
    usable_for: list[UsableFor] = Field(min_length=1)
    authority: Authority
    region_focus: list[str] = Field(default_factory=list)
    accessed_on: date | None = None


# -- ingredient ------------------------------------------------------------
IngredientCategory = Literal[
    "dal_legume", "rice_grain", "flour", "vegetable", "leafy_green", "fruit", "spice_whole",
    "spice_ground", "herb", "tempering_agent", "souring_agent", "sweetener", "fat_oil", "dairy",
    "nut_seed", "flavouring", "leavening", "salt", "water",
    "seafood", "poultry", "red_meat", "egg", "other",
]
CulinaryRole = Literal[
    "base", "thickener", "souring", "tempering", "aromatic", "sweetening", "binding", "garnish",
    "fat", "leavening", "colouring", "bulking", "protein", "marinade",
]
IngredientForm = Literal[
    "whole", "split", "skinned", "ground", "paste", "roasted", "soaked", "sprouted", "fresh",
    "dried", "powder", "flattened", "minced", "cubed", "sliced", "cleaned",
]
FlavourNote = Literal[
    "sweet", "sour", "salty", "bitter", "umami", "pungent", "astringent", "nutty", "earthy",
    "neutral",
]
AllergenClass = Literal["nuts", "dairy", "gluten", "legume", "sesame", "mustard", "fish",
                        "shellfish", "egg", "soy", "none"]


class IngredientContent(BaseModel):
    summary: str
    description: str | None = None
    selection_notes: str | None = None
    storage_notes: str | None = None
    preparation_notes: str | None = None


class IngredientFormEntry(BaseModel):
    form: IngredientForm
    notes: str | None = None


class IngredientStructure(_LocalNamed):
    category: IngredientCategory
    cuisine_state: CuisineState | None = None
    culinary_roles: list[CulinaryRole] = Field(default_factory=list)
    forms: list[IngredientFormEntry] = Field(default_factory=list)
    default_unit: Unit
    flavour_profile: list[FlavourNote] = Field(default_factory=list)
    common_substitutes: list[str] = Field(default_factory=list)
    allergen_class: list[AllergenClass] = Field(default_factory=list)
    seasonality: list[str] = Field(default_factory=list)


# -- dish ----------------------------------------------------------------
DishCategory = Literal[
    "tiffin", "rice_dish", "kuzhambu", "sambar", "rasam", "kootu", "poriyal", "masiyal",
    "thogayal", "chutney", "podi", "pachadi", "appalam_vadagam", "sweet", "snack", "beverage",
    "pickle", "bread", "festival_special",
    "biryani", "pulao", "kurma", "non_veg_curry", "seafood_dish", "egg_dish",
    "thoran", "avial", "olan", "mezhukkupuratti", "palya", "gojju", "saaru", "pulusu", "pappu",
    "vepudu", "dry_curry", "gravy_curry", "stew", "porridge", "other",
]
Course = Literal[
    "breakfast", "lunch", "dinner", "tiffin", "side", "main", "accompaniment", "dessert", "festival",
]


class DishContent(BaseModel):
    summary: str
    description: str | None = None
    serving_context: str | None = None
    cultural_notes: str | None = None
    history: str | None = None


class DishStructure(_LocalNamed):
    category: DishCategory
    cuisine_state: CuisineState
    diet: DietProfile
    style: str | None = None                       # e.g. "chettinad", "malabar", "udupi"
    course: list[Course] = Field(default_factory=list)
    primary_ingredients: list[str] = Field(default_factory=list)
    typical_techniques: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    expected_outcome_ids: list[str] = Field(default_factory=list)
    constraints_typically_met: list[str] = Field(default_factory=list)
    variant_of: str | None = None
    festival_associations: list[str] = Field(default_factory=list)


# -- recipe (process graph) --------------------------------------------------
Difficulty = Literal["easy", "moderate", "advanced"]


class RecipeIngredient(BaseModel):
    ingredient_id: str
    quantity: Measurement | None = None
    form: IngredientForm | None = None
    preparation: str | None = None
    role_in_recipe: CulinaryRole | None = None
    optional: bool = False
    substitutable: bool = True
    group: str | None = None


class StepParams(BaseModel):
    heat_level: HeatLevel | None = None
    duration: Measurement | None = None
    temperature: Measurement | None = None
    internal_temp: Measurement | None = None       # v2 — meat / seafood doneness
    doneness_cue: str | None = None                # v2 — "flakes easily", "juices run clear"
    until_condition: str | None = None


class ProcessStep(BaseModel):
    step_id: str
    order: int = Field(ge=1)
    phase: Phase
    action: str
    action_verb: ActionVerb
    input_ingredient_ids: list[str] = Field(default_factory=list)
    input_state_ids: list[str] = Field(default_factory=list)
    output_state_ids: list[str] = Field(default_factory=list)
    equipment_id: str | None = None
    technique_id: str | None = None
    params: StepParams = Field(default_factory=StepParams)
    depends_on: list[str] = Field(default_factory=list)
    parallel_ok: bool = False


class RecipeYield(BaseModel):
    servings: int | None = None
    quantity: Measurement | None = None


class RecipeContent(BaseModel):
    summary: str
    intro: str | None = None
    tips: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    serving_suggestions: str | None = None


class RecipeStructure(BaseModel):
    dish_id: str
    cuisine_state: CuisineState
    diet: DietProfile
    variant_label: str | None = None
    style: str | None = None
    yield_: RecipeYield = Field(default_factory=RecipeYield, alias="yield")
    total_time: Measurement | None = None
    active_time: Measurement | None = None
    difficulty: Difficulty | None = None
    ingredients: list[RecipeIngredient] = Field(min_length=1)
    equipment_ids: list[str] = Field(default_factory=list)
    steps: list[ProcessStep] = Field(min_length=1)
    states: list[str] = Field(default_factory=list)
    outcome_ids: list[str] = Field(default_factory=list)
    constraints_met: list[str] = Field(default_factory=list)
    scalable: bool = True
    make_ahead: bool = False

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_dag(self) -> RecipeStructure:
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step_id")
        seen: set[str] = set()
        produced: set[str] = set()
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in seen:
                    raise ValueError(f"step {step.step_id}: depends_on {dep!r} not an earlier step")
            for st in step.input_state_ids:
                if st not in produced and st not in self.states:
                    raise ValueError(
                        f"step {step.step_id}: input state {st!r} not produced upstream "
                        "nor declared in recipe.states"
                    )
            produced.update(step.output_state_ids)
            seen.add(step.step_id)
        return self


# -- technique -------------------------------------------------------------
TechniqueFamily = Literal[
    "heat_dry", "heat_fat", "heat_moist", "mechanical", "fermentation", "soaking_hydration",
    "tempering", "resting", "assembly", "marination",
]


class TechniqueContent(BaseModel):
    summary: str
    description: str | None = None
    how_to: str | None = None
    sensory_cues: str | None = None
    failure_modes: str | None = None


class StateChange(BaseModel):
    from_state: str | None = None
    to_state: str


class TechniqueTypicalParams(BaseModel):
    heat_level: HeatLevel | None = None
    duration: Measurement | None = None
    temperature: Measurement | None = None
    internal_temp: Measurement | None = None       # v2
    doneness_cue: str | None = None                # v2


class TechniqueStructure(_LocalNamed):
    family: TechniqueFamily
    cuisine_state: CuisineState | None = None
    typical_equipment_ids: list[str] = Field(default_factory=list)
    inputs_state_change: list[StateChange] = Field(default_factory=list)
    typical_params: TechniqueTypicalParams = Field(default_factory=TechniqueTypicalParams)
    precedes: list[str] = Field(default_factory=list)
    follows: list[str] = Field(default_factory=list)


# -- equipment ------------------------------------------------------------
EquipmentCategory = Literal[
    "vessel", "stove", "grinder", "steamer", "pan", "press", "utensil", "storage",
]


class EquipmentContent(BaseModel):
    summary: str
    description: str | None = None
    alternatives_notes: str | None = None


class EquipmentStructure(_LocalNamed):
    category: EquipmentCategory
    substitutes: list[str] = Field(default_factory=list)
    required_for_techniques: list[str] = Field(default_factory=list)


# -- region -------------------------------------------------------------
class RegionContent(BaseModel):
    summary: str
    description: str | None = None
    culinary_signature: str | None = None


class RegionStructure(_LocalNamed):
    level: RegionLevel                              # v2 — state | sub_region | style
    cuisine_state: CuisineState | None = None       # the state this region belongs to
    parent_region: str | None = None
    districts: list[str] = Field(default_factory=list)
    signature_dishes: list[str] = Field(default_factory=list)
    signature_ingredients: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> RegionStructure:
        if self.level == "state" and self.parent_region is not None:
            raise ValueError("a state-level region has no parent_region")
        if self.level != "state" and self.parent_region is None:
            raise ValueError(f"a {self.level} region needs a parent_region")
        return self


# -- state --------------------------------------------------------------
StateAppliesTo = Literal["ingredient", "mixture", "dish"]


class StateContent(BaseModel):
    description: str
    sensory_markers: str | None = None


class StateProperties(BaseModel):
    texture: str | None = None
    colour: str | None = None
    temperature_band: str | None = None
    moisture: str | None = None
    structure: str | None = None


class StateStructure(BaseModel):
    applies_to: StateAppliesTo
    properties: StateProperties = Field(default_factory=StateProperties)
    reachable_from: list[str] = Field(default_factory=list)
    via_technique_ids: list[str] = Field(default_factory=list)


# -- outcome ------------------------------------------------------------
OutcomeDimension = Literal[
    "texture", "flavour", "aroma", "appearance", "temperature", "consistency", "doneness",
]


class OutcomeContent(BaseModel):
    description: str
    positive_markers: str | None = None
    negative_markers: str | None = None


class OutcomeStructure(BaseModel):
    dimension: list[OutcomeDimension] = Field(min_length=1)
    measurable_by: list[str] = Field(default_factory=list)


# -- constraint ---------------------------------------------------------
ConstraintKind = Literal["dietary", "allergy", "preference", "medical", "cooking_style"]


class ConstraintContent(BaseModel):
    description: str
    rationale: str | None = None


class ConstraintStructure(BaseModel):
    kind: ConstraintKind
    excludes_ingredients: list[str] = Field(default_factory=list)
    excludes_categories: list[IngredientCategory] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    common_substitution_strategy: str | None = None


# -- problem ----------------------------------------------------------
ProblemScope = Literal[
    "ingredient_prep", "fermentation", "cooking", "texture", "flavour", "storage", "doneness",
]


class ProblemContent(BaseModel):
    description: str
    symptoms: str | None = None
    diagnosis: str | None = None
    fixes: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)


class ProblemStructure(BaseModel):
    scope: ProblemScope
    related_dish_ids: list[str] = Field(default_factory=list)
    related_technique_ids: list[str] = Field(default_factory=list)
    related_ingredient_ids: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    fix_relationship_ids: list[str] = Field(default_factory=list)


# -- term -----------------------------------------------------------
TermCategory = Literal["ingredient", "dish", "technique", "equipment", "measure", "general"]


class TermContent(BaseModel):
    gloss: str
    usage_notes: str | None = None
    example_sentence: str | None = None


class TermStructure(BaseModel):
    lang: DravidianLang
    native: str
    transliteration: str
    part_of_speech: str | None = None
    maps_to_id: str | None = None
    category: TermCategory

    @model_validator(mode="after")
    def _check(self) -> TermStructure:
        if not self.native.strip() or not self.transliteration.strip():
            raise ValueError("term needs both `native` and `transliteration`")
        return self


# -- relationship ---------------------------------------------------------
class RelContent(BaseModel):
    note: str | None = None


class RelStructure(BaseModel):
    predicate: Predicate
    subject_id: str
    object_id: str
    directional: bool = True
    attributes: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_types(self) -> RelStructure:
        subj_ok, obj_ok = PREDICATE_TYPES[self.predicate]
        sp = self.subject_id.split(":", 1)[0]
        op = self.object_id.split(":", 1)[0]
        if subj_ok != ("*",) and sp not in subj_ok:
            raise ValueError(f"predicate {self.predicate!r}: subject prefix {sp!r} not in {subj_ok}")
        if obj_ok != ("*",) and op not in obj_ok:
            raise ValueError(f"predicate {self.predicate!r}: object prefix {op!r} not in {obj_ok}")
        return self


# ---------------------------------------------------------------------------
# Concrete record classes — envelope + typed content/structure
# ---------------------------------------------------------------------------


class Source(Record):
    type: Literal["source"] = "source"
    content: SourceContent
    structure: SourceStructure


class Ingredient(Record):
    type: Literal["ingredient"] = "ingredient"
    content: IngredientContent
    structure: IngredientStructure


class Dish(Record):
    type: Literal["dish"] = "dish"
    content: DishContent
    structure: DishStructure


class Recipe(Record):
    type: Literal["recipe"] = "recipe"
    content: RecipeContent
    structure: RecipeStructure


class Technique(Record):
    type: Literal["technique"] = "technique"
    content: TechniqueContent
    structure: TechniqueStructure


class Equipment(Record):
    type: Literal["equipment"] = "equipment"
    content: EquipmentContent
    structure: EquipmentStructure


class Region(Record):
    type: Literal["region"] = "region"
    content: RegionContent
    structure: RegionStructure


class State(Record):
    type: Literal["state"] = "state"
    content: StateContent
    structure: StateStructure


class Outcome(Record):
    type: Literal["outcome"] = "outcome"
    content: OutcomeContent
    structure: OutcomeStructure


class Constraint(Record):
    type: Literal["constraint"] = "constraint"
    content: ConstraintContent
    structure: ConstraintStructure


class Problem(Record):
    type: Literal["problem"] = "problem"
    content: ProblemContent
    structure: ProblemStructure


class Term(Record):
    type: Literal["term"] = "term"
    content: TermContent
    structure: TermStructure


class Relationship(Record):
    type: Literal["rel"] = "rel"
    labels: list[str] = Field(default_factory=lambda: ["relationship"])
    content: RelContent = Field(default_factory=RelContent)
    structure: RelStructure


# type -> concrete model, for the JSONL loader / validator
RECORD_MODELS: dict[str, type[Record]] = {
    "source": Source,
    "ingredient": Ingredient,
    "dish": Dish,
    "recipe": Recipe,
    "technique": Technique,
    "equipment": Equipment,
    "region": Region,
    "state": State,
    "outcome": Outcome,
    "constraint": Constraint,
    "problem": Problem,
    "term": Term,
    "rel": Relationship,
}

# type -> canonical JSONL filename under data/canonical/
RECORD_FILES: dict[str, str] = {
    "source": "sources.jsonl",
    "ingredient": "ingredients.jsonl",
    "dish": "dishes.jsonl",
    "recipe": "recipes.jsonl",
    "technique": "techniques.jsonl",
    "equipment": "equipment.jsonl",
    "region": "regions.jsonl",
    "state": "states.jsonl",
    "outcome": "outcomes.jsonl",
    "constraint": "constraints.jsonl",
    "problem": "problems.jsonl",
    "term": "terminology.jsonl",
    "rel": "relationships.jsonl",
}

AnyRecord = Annotated[
    Source | Ingredient | Dish | Recipe | Technique | Equipment | Region | State | Outcome
    | Constraint | Problem | Term | Relationship,
    Field(discriminator="type"),
]


def parse_record(obj: dict) -> Record:
    """Validate one raw dict into its concrete record model (dispatch on ``type``)."""
    rtype = obj.get("type")
    if rtype not in RECORD_MODELS:
        raise ValueError(f"unknown record type: {rtype!r}")
    return RECORD_MODELS[rtype].model_validate(obj)
