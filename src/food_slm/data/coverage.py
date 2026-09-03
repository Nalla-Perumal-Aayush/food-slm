"""Canonical-store coverage vs `configs/coverage_targets.yaml` — powers `list_gaps()`.

Reads the current store, compares against the domain target matrix, and returns
ranked gaps following the config's `priority_order`. No LLM.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from food_slm.schema import load_store
from food_slm.schema.store import CanonicalStore
from food_slm.utils import load_yaml


@dataclass
class Gap:
    kind: str                      # e.g. "dish_cell", "ingredient_category", "technique_family"
    key: str                       # e.g. "kerala/seafood_dish", "seafood", "fermentation"
    have: int
    target: int
    priority_rank: int

    @property
    def deficit(self) -> int:
        return max(0, self.target - self.have)

    @property
    def pct(self) -> float:
        return self.have / self.target if self.target else 1.0


@dataclass
class CoverageReport:
    totals: dict[str, int] = field(default_factory=dict)          # type -> count
    total_targets: dict[str, int] = field(default_factory=dict)
    dish_cells: dict[str, int] = field(default_factory=dict)      # "state/category" -> count
    technique_families: dict[str, int] = field(default_factory=dict)
    ingredient_categories: dict[str, int] = field(default_factory=dict)
    problem_scopes: dict[str, int] = field(default_factory=dict)
    styles_present: set[str] = field(default_factory=set)

    def summary(self) -> dict:
        return {
            "records": sum(self.totals.values()),
            "by_type": {k: f"{self.totals.get(k, 0)}/{v}"
                        for k, v in self.total_targets.items()},
        }


def load_targets(path: str | Path) -> dict:
    return load_yaml(path)


def _dish_cell_targets(targets: dict) -> dict[str, int]:
    total = targets["record_targets"]["dish"]
    states = targets["cuisine_states"]
    cats = targets["dish_category_weights"]
    out: dict[str, int] = {}
    for st, meta in states.items():
        st_total = total * meta["dish_share"]
        for cat, w in cats.items():
            out[f"{st}/{cat}"] = round(st_total * w)
    return out


def compute_coverage(store: CanonicalStore, targets: dict) -> CoverageReport:
    r = CoverageReport()
    r.totals = Counter(rec.type for rec in store.records.values())
    r.total_targets = dict(targets["record_targets"])

    for rec in store.records.values():
        s = rec.structure
        if rec.type == "dish":
            cs = getattr(s, "cuisine_state", None)
            cat = getattr(s, "category", None)
            if cs and cat:
                r.dish_cells[f"{cs}/{cat}"] = r.dish_cells.get(f"{cs}/{cat}", 0) + 1
        elif rec.type == "technique":
            fam = getattr(s, "family", None)
            if fam:
                r.technique_families[fam] = r.technique_families.get(fam, 0) + 1
        elif rec.type == "ingredient":
            c = getattr(s, "category", None)
            if c:
                r.ingredient_categories[c] = r.ingredient_categories.get(c, 0) + 1
        elif rec.type == "problem":
            sc = getattr(s, "scope", None)
            if sc:
                r.problem_scopes[sc] = r.problem_scopes.get(sc, 0) + 1
        elif rec.type == "region" and getattr(s, "level", None) == "style":
            r.styles_present.add(rec.id.split(":", 1)[1])
    return r


def rank_gaps(report: CoverageReport, targets: dict, *, limit: int = 40) -> list[Gap]:
    gaps: list[Gap] = []
    prio = {name: i for i, name in enumerate(targets.get("priority_order", []))}
    dish_cell_t = _dish_cell_targets(targets)

    def add(kind: str, key: str, have: int, target: int, prio_key: str) -> None:
        if have < target:
            gaps.append(Gap(kind, key, have, target, prio.get(prio_key, 99)))

    for cat, tgt in targets.get("ingredient_category_min", {}).items():
        add("ingredient_category", cat, report.ingredient_categories.get(cat, 0), tgt,
            "under_min_ingredient_category")
    for fam, tgt in targets.get("technique_family_min", {}).items():
        add("technique_family", fam, report.technique_families.get(fam, 0), tgt,
            "under_min_technique_family")
    for cell, tgt in dish_cell_t.items():
        have = report.dish_cells.get(cell, 0)
        if tgt and have < 0.5 * tgt:
            add("dish_cell", cell, have, tgt, "dish_cell_below_50pct")
        elif tgt and have < 0.9 * tgt:
            add("dish_cell", cell, have, tgt, "dish_cell_below_90pct")
    for style in targets.get("style_targets", []):
        if style not in report.styles_present:
            add("style_tag", style, 0, 1, "missing_style_tag")
    for sc, tgt in targets.get("problem_scope_min", {}).items():
        have = report.problem_scopes.get(sc, 0)
        if have < 0.5 * tgt:
            add("problem_scope", sc, have, tgt, "problem_scope_below_50pct")

    gaps.sort(key=lambda g: (g.priority_rank, -g.deficit))
    return gaps[:limit]


def list_gaps(canonical_dir: str | Path, targets_path: str | Path, *, limit: int = 40):
    store = load_store(canonical_dir, strict=False)
    targets = load_targets(targets_path)
    report = compute_coverage(store, targets)
    return report, rank_gaps(report, targets, limit=limit)
