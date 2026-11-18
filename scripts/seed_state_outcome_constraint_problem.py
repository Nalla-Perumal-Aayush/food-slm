"""Hand-authored seed records for four more thin canonical-store gaps: `state`
(5/400), `outcome` (1/200), `constraint` (1/120), `problem` (2/600).

Same footing as scripts/seed_region_equipment.py: real, well-documented South
Indian cooking knowledge, not derived from an existing canonical record, so
transformations=["hand_authored"], not synthetic. `problem` records matter most
here -- they're the grounding source for the "troubleshooting" SFT task shape
(master_baseline.md §4.2), so getting a real batch of these in is worth more per
record than the other three.

    python scripts/seed_state_outcome_constraint_problem.py
    python scripts/seed_state_outcome_constraint_problem.py --promote --reviewer NAME
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.canon_staging import stage_records  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "data" / "interim" / "canon_staging"

# slug, applies_to, description
STATES = [
    ("raw", "ingredient", "Uncooked, unprocessed form as bought or harvested."),
    ("washed", "ingredient", "Rinsed clean of dirt/residue, not yet cut or cooked."),
    ("peeled", "ingredient", "Outer skin/husk removed."),
    ("chopped", "ingredient", "Cut into small irregular or diced pieces."),
    ("julienned", "ingredient", "Cut into thin matchstick-shaped strips."),
    ("grated", "ingredient", "Shredded into fine strands using a grater."),
    ("ground", "ingredient", "Reduced to a paste or powder by grinding."),
    ("soaked", "ingredient", "Left submerged in water to soften or hydrate before cooking."),
    ("roasted", "ingredient", "Dry-heated until fragrant, often lightly browned."),
    ("toasted", "ingredient", "Briefly dry-heated for aroma, typically spices or lentils."),
    ("marinated", "ingredient", "Coated/steeped in a seasoned mixture before cooking."),
    ("mixed", "mixture", "Combined into a uniform batter, dough, or masala."),
    ("fermented", "mixture", "Left to rise via natural fermentation (e.g. idli/dosa batter)."),
    ("tempered", "mixture", "Had a hot oil/ghee tempering (tadka) of spices stirred in."),
    ("reduced", "mixture", "Simmered down to concentrate flavour and thicken."),
    ("rested", "mixture", "Left to sit undisturbed after mixing, before cooking or serving."),
    ("boiling", "mixture", "Actively at a rolling boil."),
    ("simmering", "mixture", "Held at a gentle, low bubbling heat."),
    ("plated", "dish", "Arranged on a serving plate/banana leaf, ready to serve."),
    ("garnished", "dish", "Finished with a topping (coriander, coconut, curry leaves, etc.)."),
    ("reheated", "dish", "Warmed again after initial cooking, e.g. leftovers."),
]

# slug, dimensions (list), description
OUTCOMES = [
    ("crispy", ["texture"], "Firm and crunchy on the outside when bitten."),
    ("soft", ["texture"], "Yielding and tender, minimal resistance when bitten."),
    ("fluffy", ["texture"], "Light and airy, not dense (idli, rice)."),
    ("mushy", ["texture"], "Overly soft, broken-down, loss of structure -- usually undesired."),
    ("chewy", ["texture"], "Requires sustained chewing; elastic bite."),
    ("crumbly", ["texture"], "Breaks apart easily into small pieces."),
    ("spongy", ["texture"], "Porous and springs back when pressed (idli)."),
    ("golden-brown", ["appearance"], "Even golden-brown surface colour, sign of good frying/roasting."),
    ("glossy", ["appearance"], "Shiny surface sheen, often from oil/ghee."),
    ("al-dente", ["doneness", "texture"], "Cooked through but still firm to the bite."),
    ("tender", ["texture", "doneness"], "Cooked soft enough to cut/chew easily without resistance."),
    ("tangy", ["flavour"], "Pronounced sour note, typically from tamarind, yogurt, or lime."),
    ("balanced-spice", ["flavour"], "Heat, salt, sour, and other notes in proportion, no single note dominating."),
    ("bitter", ["flavour"], "Unpleasant bitter note, often from burnt garlic/spices or over-reduced greens."),
    ("aromatic", ["aroma"], "Strong, pleasant fragrance, typically from fresh tempering or roasted spices."),
    ("burnt-smell", ["aroma"], "Acrid smell from scorched oil/spices/batter -- a sign something overcooked."),
    ("piping-hot", ["temperature"], "Served very hot, just off the stove."),
    ("thick-consistency", ["consistency"], "Dense, coats the back of a spoon (e.g. a well-reduced gravy)."),
    ("runny-consistency", ["consistency"], "Thin and watery, flows easily -- often undesired for a gravy/batter."),
    ("well-cooked", ["doneness"], "Cooked all the way through, no raw centre remaining."),
]

# slug, kind, description
CONSTRAINTS = [
    ("vegetarian", "dietary", "Excludes meat, poultry, and seafood; dairy and egg vary by sub-type."),
    ("vegan", "dietary", "Excludes all animal products, including dairy, egg, and honey."),
    ("eggetarian", "dietary", "Vegetarian diet that additionally permits eggs."),
    ("pescatarian", "dietary", "Vegetarian diet that additionally permits seafood/fish."),
    ("onion-garlic-free", "cooking_style", "Excludes onion and garlic, common in temple-style/saatvik cooking."),
    ("jain-style", "cooking_style", "Excludes onion, garlic, and root vegetables, per Jain dietary practice."),
    ("gluten-free", "dietary", "Excludes wheat and other gluten-containing grains; most South Indian rice/lentil staples qualify naturally."),
    ("nut-allergy", "allergy", "Must avoid tree nuts and peanuts / groundnuts."),
    ("dairy-allergy", "allergy", "Must avoid milk, ghee, curd, and other dairy."),
    ("seafood-allergy", "allergy", "Must avoid fish and shellfish."),
    ("low-spice", "preference", "Reduced chili/pepper heat versus the standard recipe."),
    ("high-spice", "preference", "Increased chili/pepper heat versus the standard recipe."),
    ("high-protein", "preference", "Emphasizes protein-dense ingredients (lentils, egg, meat, dairy)."),
    ("low-oil", "preference", "Reduced oil/ghee versus the standard recipe."),
    ("diabetic-friendly", "medical", "Limits added sugar and high-glycemic ingredients."),
    ("low-sodium", "medical", "Reduced salt versus the standard recipe."),
    ("heart-healthy", "medical", "Limits saturated fat and salt; favours steaming/light tempering over deep-frying."),
    ("one-pot-meal", "cooking_style", "Prepared and often served entirely from a single vessel."),
]

# slug, scope, description, symptoms, diagnosis, fixes, prevention
PROBLEMS = [
    ("idli-batter-not-fermenting", "fermentation",
     "Idli/dosa batter fails to rise after the usual fermentation period.",
     "Batter stays flat and smells the same as when mixed; no bubbles.",
     "Ambient temperature too cold, too little urad dal, or old/dead fermenting culture (no starter batter).",
     ["Keep the batter in a warm spot (oven with light on, or near a warm appliance)",
      "Increase the urad dal proportion slightly",
      "Add a spoon of old fermented batter as a starter"],
     ["Ferment in a warm season/location", "Use fresh urad dal", "Don't over-wash away the natural yeasts"]),
    ("idli-turns-hard", "texture",
     "Steamed idlis come out dense and hard instead of soft and spongy.",
     "Idli is compact, doesn't spring back when pressed.",
     "Under-fermented batter, batter ground too coarse, or over-steaming.",
     ["Steam for the correct time only (usually 8-12 minutes)",
      "Ensure batter is well-fermented and light before steaming"],
     ["Grind batter to the right smooth-but-slightly-grainy consistency", "Don't skip fermentation time"]),
    ("dosa-sticks-to-tawa", "cooking",
     "Dosa tears or sticks when trying to flip or remove it from the tawa.",
     "Batter clings to the pan surface instead of releasing cleanly.",
     "Tawa not seasoned/heated properly, or too little oil.",
     ["Heat the tawa properly and season it with oil before the first dosa",
      "Use enough oil around the edges", "Let the dosa cook fully before attempting to lift it"],
     ["Use a well-seasoned cast-iron or non-stick tawa", "Maintain consistent medium heat"]),
    ("dosa-not-crispy", "texture",
     "Dosa turns out soft/chewy instead of thin and crisp.",
     "Dosa doesn't crisp at the edges even after cooking.",
     "Batter too thick, too much rice flour vs. urad dal, or tawa not hot enough.",
     ["Thin the batter slightly with water", "Cook on a hotter tawa", "Spread the batter thinner"],
     ["Get the rice-to-urad-dal ratio right", "Preheat the tawa well before pouring batter"]),
    ("sambar-too-watery", "texture",
     "Sambar turns out thin and soup-like instead of a thick stew consistency.",
     "Sambar doesn't coat the back of a spoon.",
     "Too much water added, or dal under-cooked/under-mashed.",
     ["Simmer uncovered to reduce and thicken", "Mash the cooked dal well before adding vegetables"],
     ["Measure water against dal quantity carefully"]),
    ("sambar-bland", "flavour",
     "Sambar lacks the expected tangy, spiced depth of flavour.",
     "Tastes flat despite having the right ingredients.",
     "Too little tamarind/sambar powder, or tempering (tadka) skipped or added too early.",
     ["Adjust tamarind and sambar powder to taste", "Add a fresh tempering of mustard seeds, curry leaves, and asafoetida just before serving"],
     ["Always finish with a fresh, hot tempering"]),
    ("rasam-too-sour", "flavour",
     "Rasam tastes overpoweringly sour rather than balanced.",
     "Sourness dominates over the pepper/tomato/spice notes.",
     "Too much tamarind or tomato relative to water and spice.",
     ["Dilute with more water/dal water", "Balance with a touch of jaggery"],
     ["Measure tamarind extract carefully"]),
    ("rasam-no-froth", "texture",
     "Rasam lacks the light frothy top that indicates it's properly finished.",
     "Rasam looks flat/thin on top instead of slightly frothy.",
     "Boiled too vigorously for too long, or rasam powder added too early and cooked out.",
     ["Heat gently and remove from heat as soon as it just starts to froth, don't let it hard-boil"],
     ["Add rasam powder near the end and avoid a rolling boil after"]),
    ("chutney-turns-bitter", "flavour",
     "Coconut/vegetable chutney develops an unexpected bitter edge.",
     "Bitterness noticeable despite normal ingredients.",
     "Garlic or spices burnt during tempering, or the brown part of the coconut ground in.",
     ["Redo the tempering carefully at moderate heat", "Scrape away the brown coconut skin before grinding"],
     ["Temper on medium heat and watch closely", "Use only the white coconut flesh"]),
    ("chutney-separates", "texture",
     "Ground chutney turns watery and separates instead of staying emulsified.",
     "Liquid pools on top after grinding or sitting.",
     "Too much water added while grinding, or chutney sat too long before serving.",
     ["Grind with minimal water and adjust consistency at the end", "Serve fresh"],
     ["Add water gradually while grinding"]),
    ("vada-absorbs-too-much-oil", "cooking",
     "Vadas turn out greasy/oily instead of crisp and light.",
     "Vada feels heavy and oil-soaked when bitten.",
     "Oil not hot enough when frying, or batter too soft/wet.",
     ["Fry at the correct hot oil temperature (test with a small drop of batter)", "Firm up the batter before shaping"],
     ["Always test oil temperature before frying a full batch"]),
    ("vada-no-holes",
     "texture", "Vada turns out dense/flat instead of having the characteristic airy holes.",
     "Vada lacks the light doughnut-like texture.",
     "Batter over- or under-beaten, or not aerated enough before frying.",
     ["Beat the batter well to incorporate air before shaping"],
     ["Beat until the batter feels light and slightly increases in volume"]),
    ("payasam-curdles", "texture",
     "Milk-based payasam curdles/splits instead of staying smooth.",
     "Visible curdled specks in the payasam.",
     "Adding an acidic ingredient (like fruit) to hot milk too quickly, or milk cooked at too high heat for too long.",
     ["Add acidic ingredients off the heat and gradually", "Cook milk-based payasam on low-medium heat"],
     ["Temper acidic additions in slowly at the end"]),
    ("rice-mushy", "doneness",
     "Cooked rice turns out sticky/mushy instead of separate grains.",
     "Grains clump together and lose shape.",
     "Too much water or overcooked.",
     ["Use the correct water-to-rice ratio for the variety", "Reduce cooking time slightly"],
     ["Rinse rice before cooking to remove excess surface starch"]),
    ("rice-undercooked", "doneness",
     "Rice grains remain hard/uncooked in the centre.",
     "Bite resistance is too firm, grains taste starchy-raw.",
     "Too little water or insufficient cooking time.",
     ["Add a little more water and continue cooking covered", "Extend cooking time"],
     ["Measure water correctly for the rice variety used"]),
    ("batter-over-fermented", "fermentation",
     "Idli/dosa batter turns unpleasantly sour from fermenting too long.",
     "Strong sour smell and taste beyond the normal mild tang.",
     "Left to ferment too long or in too warm an environment.",
     ["Use it for dosa (which tolerates more sourness) rather than idli", "Add a pinch of baking soda to balance sourness"],
     ["Refrigerate the batter once it has fermented enough"]),
    ("puttu-dry-crumbly", "texture",
     "Puttu turns out dry and falls apart instead of holding together lightly.",
     "Puttu crumbles excessively when unmoulded.",
     "Rice flour not moistened enough before steaming.",
     ["Sprinkle and mix water into the flour until it just holds together when pressed, before steaming"],
     ["Test the flour's moisture with a pinch-test before layering into the steamer"]),
    ("puttu-too-wet", "texture",
     "Puttu turns out soggy/gummy instead of light and fluffy.",
     "Puttu feels dense and sticky.",
     "Too much water added to the rice flour before steaming.",
     ["Reduce the water slightly next time and steam a touch longer"],
     ["Add water gradually, testing texture as you go"]),
    ("papad-not-puffing", "doneness",
     "Papad doesn't puff up fully when fried or roasted.",
     "Papad stays flat or only partially expands.",
     "Oil not hot enough (if frying) or papad too old/dried out.",
     ["Increase oil temperature and fry briefly", "Use fresher papad"],
     ["Store papad in an airtight container away from moisture"]),
    ("pickle-mould", "storage",
     "Pickle develops mould during storage.",
     "Visible fuzzy growth on the surface or jar sides.",
     "Insufficient oil/salt as a preservative, moisture contamination, or a non-dry serving spoon used.",
     ["Discard affected pickle", "Remake with adequate oil/salt and ensure the jar and spoon are completely dry"],
     ["Always use a dry spoon", "Keep pickle submerged in oil", "Store in a cool, dry place"]),
    ("spice-powder-loses-aroma", "storage",
     "Ground spice powder (sambar/rasam podi etc.) loses its fragrance and potency over time.",
     "Powder smells and tastes flat compared to when freshly made.",
     "Prolonged storage, exposure to air/light/moisture, or storing whole roasted spices ground too far in advance.",
     ["Use within a few weeks of grinding for best flavour"],
     ["Store in an airtight container away from light and heat", "Grind in smaller batches more frequently"]),
    ("dal-not-cooking-soft", "doneness",
     "Lentils/dal remain firm even after extended cooking.",
     "Dal stays grainy/hard rather than turning creamy.",
     "Old dal stock, hard water, or salt/acidic ingredients added too early.",
     ["Continue cooking with more water", "Add a pinch of baking soda to help soften (use sparingly)"],
     ["Add salt and tamarind/tomato only after the dal is fully cooked soft"]),
    ("tempering-burns", "cooking",
     "Tempering (tadka) burns instead of sizzling and releasing aroma.",
     "Mustard seeds/curry leaves turn black and smell acrid rather than fragrant.",
     "Oil too hot or tempering left in the pan too long.",
     ["Discard the burnt tempering and redo it at a slightly lower heat"],
     ["Add ingredients to the tempering in the right order (mustard seeds first, curry leaves/asafoetida last) and keep heat moderate"]),
    ("filter-coffee-bitter", "flavour",
     "South Indian filter coffee tastes bitter/harsh instead of smooth.",
     "Sharp bitterness overwhelms the coffee-chicory flavour.",
     "Over-extraction (too long a drip time or too fine a grind) or too high a coffee-to-water ratio.",
     ["Dilute with more hot milk", "Use a coarser grind or reduce drip time next time"],
     ["Follow the traditional slow-drip method for the right extraction"]),
    ("ghee-turns-rancid", "storage",
     "Stored ghee develops an off, rancid smell/taste.",
     "Ghee smells sour or 'off' rather than nutty and fresh.",
     "Moisture contamination or prolonged exposure to heat/light.",
     ["Discard rancid ghee"],
     ["Always use a completely dry spoon", "Store in an airtight container away from direct heat/sunlight"]),
]


def _envelope(rid: str, rtype: str, label: str, today: str) -> dict:
    return {
        "id": rid, "type": rtype, "schema_version": "2", "labels": [label],
        "provenance": {"source_ids": [], "transformations": ["hand_authored"],
                       "acquired_on": today, "synthetic": False},
        "quality": {"confidence": 0.85, "validation_status": "auto_validated",
                   "corroboration_count": 1},
        "version": {"revision": 1, "created_on": today, "updated_on": today,
                   "status": "active"},
    }


def build_states(today: str) -> list[dict]:
    out = []
    for slug, applies_to, desc in STATES:
        rec = _envelope(f"state:{slug}", "state", slug.replace("-", " ").title(), today)
        rec["content"] = {"description": desc}
        rec["structure"] = {"applies_to": applies_to}
        out.append(rec)
    return out


def build_outcomes(today: str) -> list[dict]:
    out = []
    for slug, dims, desc in OUTCOMES:
        rec = _envelope(f"outcome:{slug}", "outcome", slug.replace("-", " ").title(), today)
        rec["content"] = {"description": desc}
        rec["structure"] = {"dimension": dims}
        out.append(rec)
    return out


def build_constraints(today: str) -> list[dict]:
    out = []
    for slug, kind, desc in CONSTRAINTS:
        rec = _envelope(f"constraint:{slug}", "constraint", slug.replace("-", " ").title(), today)
        rec["content"] = {"description": desc}
        rec["structure"] = {"kind": kind}
        out.append(rec)
    return out


def build_problems(today: str) -> list[dict]:
    out = []
    for slug, scope, desc, symptoms, diagnosis, fixes, prevention in PROBLEMS:
        rec = _envelope(f"problem:{slug}", "problem", slug.replace("-", " ").title(), today)
        rec["content"] = {"description": desc, "symptoms": symptoms, "diagnosis": diagnosis,
                          "fixes": fixes, "prevention": prevention}
        rec["structure"] = {"scope": scope}
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--reviewer", default=None)
    args = ap.parse_args()
    if args.promote and not args.reviewer:
        ap.error("--promote requires --reviewer")

    today = datetime.now(UTC).date().isoformat()
    records = (build_states(today) + build_outcomes(today)
              + build_constraints(today) + build_problems(today))
    res = stage_records(records, STAGING)
    print(f"staged: {len(res.staged)}  rejected: {len(res.rejected)}")
    for rid, reason in res.rejected:
        print(f"  REJECTED {rid}: {reason}")

    if args.promote:
        subprocess.run(  # noqa: S603
            [sys.executable, str(REPO / "scripts" / "canon_run.py"), "promote-ready",
             "--reviewer", args.reviewer], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
