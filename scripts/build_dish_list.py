"""Merge the curated region dish seeds into the v1 1,000-dish list (150M plan, Phase 1).

    python scripts/build_dish_list.py

Primary source: data/dish_seed/*.yaml  (hand-authored, one file per region).
Signal overlay:  scraped recipe titles (blogs + recipe JSONL + canonical store) are
used ONLY to (a) boost the prominence score of a dish that is already curated and
(b) surface an appendix of candidates not yet in the list — never auto-merged
(automated harvest was shown to be too noisy, see plan_150m dataset notes).

Outputs:
  configs/dish_list_v1.yaml                     — the draft list, grouped by region
  artifacts/reports/dish_list_v1_coverage.md    — coverage vs configs/coverage_targets.yaml
No GPU, no LLM, no network.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.dish_list import (  # noqa: E402
    _NOT_A_DISH,
    _si_signal,
    dedupe_key,
    harvest,
    normalise_name,
)
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEED_DIR = REPO / "data" / "dish_seed"
REGIONS = ["tamil_nadu", "kerala", "karnataka", "andhra_pradesh", "telangana",
           "puducherry", "pan_south"]

# categories that carry more weight when no explicit prominence is given
_BASE_PROM = {
    "tiffin": 6, "rice_dish": 6, "sambar": 6, "rasam": 6, "biryani": 6,
    "non_veg_curry": 5, "seafood_dish": 5, "sweet": 5, "chutney": 5, "snack": 4,
    "kuzhambu": 5, "pulusu": 4, "pappu": 5, "kootu": 4, "poriyal": 4, "thoran": 4,
    "palya": 4, "vepudu": 4, "kura": 4, "pachadi": 4, "podi": 3, "pickle": 3,
    "beverage": 4, "bread": 5, "gojju": 3, "gravy_curry": 4, "kurma": 4,
    "dry_curry": 4, "egg_dish": 4, "festival_special": 4, "porridge": 3,
    "masiyal": 3, "thogayal": 3, "side": 3, "staple": 6, "concept": 5,
    "technique": 4, "saaru": 4, "avial": 4, "olan": 3,
}
_NATIVE_KEYS = {"ta", "ml", "kn", "te"}
_PARENS = re.compile(r"\s*\(([^)]*)\)")
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-")


def _split_name(raw: str) -> tuple[str, list[str], str | None]:
    """'Curd Rice / Thayir Sadam (Andhra style)' -> ('Curd Rice',
    ['Thayir Sadam'], 'Andhra style')."""
    note = None
    m = _PARENS.search(raw)
    if m:
        note = m.group(1).strip()
        raw = _PARENS.sub("", raw).strip()
    parts = [p.strip() for p in raw.split(" / ") if p.strip()]
    canonical = parts[0]
    aliases = parts[1:]
    return canonical, aliases, note


def _diet(spec) -> tuple[dict, bool]:
    d = {"vegetarian": True, "vegan": False, "contains_dairy": False,
         "contains_egg": False, "contains_seafood": False, "contains_poultry": False,
         "contains_red_meat": False, "contains_beef": False}
    if spec == "vegan":
        d["vegan"] = True
        return d, False
    if spec == "veg" or spec is None:
        d["contains_dairy"] = True          # ghee/curd/milk assumed unless marked vegan
        return d, False
    toks = spec if isinstance(spec, list) else [spec]
    d["vegetarian"] = False
    for t in toks:
        t = str(t).strip().lower()
        if t == "egg":
            d["contains_egg"] = True
        elif t == "seafood":
            d["contains_seafood"] = True
        elif t == "poultry":
            d["contains_poultry"] = True
        elif t == "red_meat":
            d["contains_red_meat"] = True
        elif t == "beef":
            d["contains_beef"] = True
            d["contains_red_meat"] = True
    return d, True


def load_curated() -> list[dict]:
    rows: list[dict] = []
    for region in REGIONS:
        f = SEED_DIR / f"{region}.yaml"
        if not f.exists():
            print(f"  ! missing {f.name}")
            continue
        doc = load_yaml(f)
        for e in doc.get("dishes", []):
            canonical, aliases, note = _split_name(e["name"])
            aliases = list(aliases)
            aka = e.get("aka") or []
            aliases.extend(a for a in aka if a not in aliases)
            native = {k: e[k] for k in _NATIVE_KEYS if k in e}
            diet, non_veg = _diet(e.get("diet"))
            cat = e.get("cat", "other")
            rows.append({
                "canonical_name": canonical,
                "aliases": aliases,
                "native": native,
                "region": region,
                "also_in": list(e.get("also_in", [])),
                "category": cat,
                "style": e.get("style"),
                "diet": diet,
                "non_veg": non_veg,
                "note": note,
                "base_prom": e.get("prominence", _BASE_PROM.get(cat, 3)),
                "key": dedupe_key(normalise_name(canonical)),
            })
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Collapse duplicates conservatively: only merge two entries with the same
    dedupe key when they are in the SAME region, or when one of them is pan_south
    (regional variants of a shared dish stay separate on purpose — regional
    comparison is a domain-question category)."""
    order = {r: i for i, r in enumerate(REGIONS)}
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["key"]].append(r)

    out: list[dict] = []
    for _, group in buckets.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        kept: list[dict] = []
        for r in sorted(group, key=lambda x: order[x["region"]]):
            merged = False
            for k in kept:
                same_region = k["region"] == r["region"]
                pan_involved = "pan_south" in (k["region"], r["region"])
                if same_region or pan_involved:
                    winner = k if order[k["region"]] <= order[r["region"]] else r
                    loser = r if winner is k else k
                    if winner is r:                    # r replaces k in place
                        k.update(r)
                    for reg in [loser["region"], *loser["also_in"]]:
                        if reg != winner["region"] and reg != "pan_south" \
                                and reg not in winner["also_in"]:
                            winner["also_in"].append(reg)
                    for a in loser["aliases"]:
                        if a not in winner["aliases"]:
                            winner["aliases"].append(a)
                    winner["base_prom"] = max(winner["base_prom"], loser["base_prom"])
                    merged = True
                    break
            if not merged:
                kept.append(r)
        out.extend(kept)
    return out


def scrape_freq(sources: list[Path]) -> Counter:
    raw = harvest(sources)
    freq: Counter = Counter()
    for title, _ in raw:
        if _NOT_A_DISH.match(title.strip()):
            continue
        n = normalise_name(title)
        if _NOT_A_DISH.match(n) or not (3 <= len(n) and 2 <= len(n.split()) <= 6):
            continue
        freq[dedupe_key(n)] += 1
    return freq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "dish_list_v1.yaml")
    ap.add_argument("--report", type=Path,
                    default=REPO / "artifacts" / "reports" / "dish_list_v1_coverage.md")
    ap.add_argument("--appendix-min", type=int, default=8,
                    help="min scrape frequency for an un-curated candidate to be listed")
    args = ap.parse_args()

    rows = dedupe(load_curated())
    print(f"curated dishes: {len(rows)}  (after cross-file dedupe)")

    blog_dir = REPO / "data" / "interim" / "domain_150m" / "blogs"
    scrape_sources = [
        *sorted(blog_dir.glob("*.txt")),
        REPO / "data" / "interim" / "recipes_150m" / "core_si.jsonl",
        REPO / "data" / "canonical" / "dishes.jsonl",
    ]
    freq = scrape_freq([p for p in scrape_sources if p.exists()])
    print(f"scrape title keys: {len(freq)}  ({sum(freq.values())} titles)")

    matched = 0
    for r in rows:
        f = freq.get(r["key"], 0)
        if f:
            matched += 1
        boost = min(4, round(math.log2(f + 1))) if f else 0
        r["prominence"] = max(1, min(10, r["base_prom"] + boost))
        r["scrape_freq"] = f
        r["id"] = f"dish:{_slug(r['canonical_name'])}"
    print(f"curated dishes with a scrape-title match: {matched}/{len(rows)}")

    # ids unique
    seen: dict[str, int] = {}
    for r in rows:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
        if seen[r["id"]] > 1:
            r["id"] = f"{r['id']}-{seen[r['id']]}"

    _write_yaml(rows, args.out)
    print(f"-> {args.out.relative_to(REPO)}")

    curated_keys = {r["key"] for r in rows}
    uncurated = [(k, c) for k, c in freq.most_common()
                 if c >= args.appendix_min and k not in curated_keys
                 and _si_signal(k, [])]
    _write_report(rows, uncurated, args.report)
    print(f"-> {args.report.relative_to(REPO)}")
    return 0


def _write_yaml(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    by_region: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_region[r["region"]].append(r)
    lines = [
        "# v1 South Indian dish list — DRAFT for review.",
        "# Primary source: data/dish_seed/*.yaml (hand-authored). Prominence = hand base",
        "# score + log2 boost from scraped-title frequency. Regenerate:",
        "#   python scripts/build_dish_list.py",
        f"# generated {datetime.now(UTC).isoformat(timespec='seconds')}  total={len(rows)}",
        "version: 1",
        "dishes:",
    ]
    for region in REGIONS:
        rs = sorted(by_region.get(region, []), key=lambda d: (-d["prominence"],
                                                              d["canonical_name"]))
        if not rs:
            continue
        lines.append(f"  # --- {region}  ({len(rs)}) ---")
        for r in rs:
            lines.append(f"  - id: {r['id']}")
            lines.append(f"    canonical_name: {_q(r['canonical_name'])}")
            if r["aliases"]:
                lines.append(f"    also_known_as: [{', '.join(_q(a) for a in r['aliases'])}]")
            if r["native"]:
                nat = ", ".join(f"{k}: {_q(v)}" for k, v in r["native"].items())
                lines.append(f"    native: {{{nat}}}")
            lines.append(f"    region: {r['region']}")
            if r["also_in"]:
                lines.append(f"    also_in: [{', '.join(r['also_in'])}]")
            lines.append(f"    category: {r['category']}")
            if r["style"]:
                lines.append(f"    style: {r['style']}")
            if r["note"]:
                lines.append(f"    note: {_q(r['note'])}")
            d = r["diet"]
            flags = [k.replace("contains_", "") for k in
                     ("contains_dairy", "contains_egg", "contains_seafood",
                      "contains_poultry", "contains_red_meat", "contains_beef") if d[k]]
            lines.append(f"    diet: {{vegetarian: {str(d['vegetarian']).lower()}, "
                         f"vegan: {str(d['vegan']).lower()}, "
                         f"non_veg: {str(r['non_veg']).lower()}"
                         + (f", contains: [{', '.join(flags)}]" if flags else "") + "}")
            lines.append(f"    prominence: {r['prominence']}    # scrape_freq={r['scrape_freq']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _q(s: str) -> str:
    s = str(s)
    return f'"{s}"' if re.search(r"[:#\[\]{}',]", s) or s != s.strip() else s


def _write_report(rows: list[dict], uncurated: list[tuple[str, int]], out: Path) -> None:
    targets = load_yaml(REPO / "configs" / "coverage_targets.yaml")
    cs = targets["cuisine_states"]
    cw = targets["dish_category_weights"]
    n = len(rows)

    by_region = Counter(r["region"] for r in rows)
    nv_by_region: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        nv_by_region[r["region"]][0] += 1
        nv_by_region[r["region"]][1] += int(r["non_veg"])
    by_cat = Counter(r["category"] for r in rows)
    by_style = Counter(r["style"] for r in rows if r["style"])

    L = [
        f"# Dish list v1 — coverage report ({datetime.now(UTC).date()})",
        "",
        f"**{n} dishes** across {len(by_region)} regions. Draft; awaiting review.",
        "",
        "## Region balance vs target",
        "",
        "| region | dishes | share | target share | at N=1000 | non-veg | nv target |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for region in REGIONS:
        cnt = by_region.get(region, 0)
        share = cnt / n if n else 0
        tgt = cs.get(region, {}).get("dish_share", 0)
        tot, nv = nv_by_region[region]
        nvf = nv / tot if tot else 0
        nvt = cs.get(region, {}).get("non_veg_fraction", 0)
        L.append(f"| {region} | {cnt} | {share:.0%} | {tgt:.0%} | "
                 f"{round(tgt * 1000)} | {nvf:.0%} | {nvt:.0%} |")
    L += [
        "",
        "## Category balance vs target weight",
        "",
        "| category | dishes | share | target weight |",
        "|---|--:|--:|--:|",
    ]
    for cat, cnt in by_cat.most_common():
        L.append(f"| {cat} | {cnt} | {cnt / n:.1%} | "
                 + (f"{cw[cat]:.0%}" if cat in cw else "—") + " |")
    L += ["", "## Style tags present", ""]
    for s, c in by_style.most_common():
        L.append(f"- {s}: {c}")
    L += [
        "",
        "## Un-curated scrape candidates (freq ≥ threshold, NOT in list)",
        "",
        "Review these — add the genuine South Indian ones to the seed files, ignore the rest.",
        "",
        "| dedupe key | scrape freq |",
        "|---|--:|",
    ]
    for k, c in uncurated[:120]:
        L.append(f"| {k} | {c} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
