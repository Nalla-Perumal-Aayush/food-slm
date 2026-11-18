"""Hand-authored seed records for `region` and `equipment` -- two of the thinnest
canonical-store gaps (8/60 and 7/80) that don't need fresh LLM extraction: both are
small, closed, well-documented domains (the 6 South Indian states + their
well-known culinary sub-regions/styles; standard South Indian cookware).

Not synthetic (there's no existing canonical record this was derived/transformed
from -- it's original content, same footing as the hand-authored data/dish_seed/
files), so provenance uses `transformations: ["hand_authored"]` with no
source_ids, per food_slm.schema.models.Provenance / check_provenance().

    python scripts/seed_region_equipment.py                     # stage only
    python scripts/seed_region_equipment.py --promote --reviewer NAME
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

# id, level, cuisine_state (None for pan_south's own record), parent_region, summary
REGIONS = [
    ("tamil-nadu", "state", "tamil_nadu", None,
     "Tamil Nadu, southeastern coastal state; rice-and-lentil-based cuisine "
     "(sambar, rasam, dosa, idli), Tamil-speaking."),
    ("kerala", "state", "kerala", None,
     "Kerala, southwestern coastal state; coconut- and seafood-heavy cuisine "
     "with extensive spice use, Malayalam-speaking."),
    ("karnataka", "state", "karnataka", None,
     "Karnataka, southwestern inland/coastal state; diverse sub-regional cuisines "
     "(Udupi, coastal Mangalorean, North Karnataka), Kannada-speaking."),
    ("andhra-pradesh", "state", "andhra_pradesh", None,
     "Andhra Pradesh, southeastern coastal/inland state; known for the spiciest "
     "South Indian cuisine, tamarind- and chili-forward, Telugu-speaking."),
    ("telangana", "state", "telangana", None,
     "Telangana, south-central inland state on the Deccan plateau; millet-based "
     "staples and Hyderabadi-influenced cuisine, Telugu-speaking."),
    ("puducherry", "state", "puducherry", None,
     "Puducherry, small coastal union territory; blends Tamil culinary tradition "
     "with French colonial influence."),
    ("pan-south", "state", "pan_south", None,
     "Cross-cutting category for dishes/ingredients/techniques common across "
     "multiple South Indian states rather than specific to one."),
    ("chettinad", "style", "tamil_nadu", "region:tamil-nadu",
     "Chettinad style, from the Chettinad region of Tamil Nadu; complex spice "
     "blends, strong use of black pepper, star anise, and fried curry leaves."),
    ("kongunadu", "style", "tamil_nadu", "region:tamil-nadu",
     "Kongunadu style, from the Kongu belt (Coimbatore/Erode/Tiruppur) of Tamil "
     "Nadu; hearty, distinctively spiced vegetarian and non-vegetarian dishes."),
    ("malabar", "style", "kerala", "region:kerala",
     "Malabar style, from northern Kerala's Malabar coast; coconut- and "
     "seafood-rich, with historic Arab trade influence on spice use."),
    ("udupi", "style", "karnataka", "region:karnataka",
     "Udupi style, from Udupi in coastal Karnataka; a vegetarian tradition known "
     "for its distinctive sambar, rasam, and temple-influenced tiffin items."),
    ("rayalaseema", "style", "andhra_pradesh", "region:andhra-pradesh",
     "Rayalaseema style, from the Rayalaseema region of Andhra Pradesh; "
     "extremely spicy, dry, chili- and groundnut-forward preparations."),
    ("coorg", "style", "karnataka", "region:karnataka",
     "Coorg (Kodagu) style, from the hill region of Karnataka; pork-based "
     "dishes, kachampuli (a sour vinegar), and bamboo-shoot use."),
    ("hyderabadi-deccan", "style", "telangana", "region:telangana",
     "Hyderabadi-Deccan style, from the Hyderabad/Deccan region of Telangana; "
     "blends Mughlai and Telugu influences, known for biryani and dum-cooking."),
    ("mangalorean", "style", "karnataka", "region:karnataka",
     "Mangalorean style, from coastal Karnataka around Mangalore; coconut- and "
     "seafood-based, with generous use of tamarind and red chilies."),
]

# id, category, summary
EQUIPMENT = [
    ("idli-stand", "steamer",
     "Stackable perforated plates that hold idli batter, placed inside a "
     "steamer or pressure cooker to steam idlis."),
    ("dosa-tawa", "pan",
     "Flat, heavy round griddle, traditionally cast iron, used to spread and "
     "cook dosa batter into thin crepes."),
    ("kadai", "vessel",
     "Wide, deep, round-bottomed pan (similar to a wok) used for frying, "
     "sauteing, and simmering."),
    ("uruli", "vessel",
     "Wide, shallow, round-bottomed bronze or bell-metal vessel from Kerala, "
     "traditionally used for slow-cooking payasam and halwa."),
    ("puttu-kudam", "steamer",
     "Cylindrical steamer with a perforated base used to layer and steam "
     "puttu (rice flour and coconut)."),
    ("appa-chatti", "pan",
     "Small, deep, rounded pan with a lid used to cook appam, producing its "
     "characteristic thick soft center and thin crisp edges."),
    ("kuzhi-paniyaram-pan", "pan",
     "Cast-iron pan with multiple hemispherical indentations used to cook "
     "paniyaram / appe."),
    ("wet-grinder", "grinder",
     "Traditional stone-wheel or motorized grinder used to grind soaked rice "
     "and urad dal into the batter for idli/dosa."),
    ("ural", "grinder",
     "Traditional stone mortar and pestle used for grinding spices, "
     "chutneys, and batters by hand."),
    ("murukku-achu", "press",
     "Hand press with interchangeable perforated discs used to extrude "
     "murukku dough into its characteristic spiral shape."),
    ("sombu", "vessel",
     "Traditional lidded metal vessel, often used for serving or storing "
     "rasam or water."),
    ("manchatti", "vessel",
     "Unglazed clay pot used for slow-cooking curries and rice, valued for "
     "the earthy flavour it imparts."),
    ("idiyappam-press", "press",
     "Hand press used to extrude rice-flour dough into thin strands for "
     "idiyappam (string hoppers)."),
    ("aduppu", "stove",
     "Traditional wood- or charcoal-fired stove, historically used for South "
     "Indian cooking before gas stoves became common."),
    ("kal-chatti", "vessel",
     "Traditional soapstone (granite) cookware, prized for even heat "
     "retention, used for curries and rice."),
    ("sada-karandi", "utensil",
     "Long-handled ladle used for stirring and serving liquid dishes like "
     "sambar and rasam."),
    ("uppu-petti", "storage",
     "Traditional salt-storage box, often kept near the stove for easy "
     "access during cooking."),
]


def _envelope(rid: str, rtype: str, label: str, today: str) -> dict:
    return {
        "id": rid, "type": rtype, "schema_version": "2", "labels": [label],
        "provenance": {"source_ids": [], "transformations": ["hand_authored"],
                       "acquired_on": today, "synthetic": False},
        "quality": {"confidence": 0.9, "validation_status": "auto_validated",
                   "corroboration_count": 1},
        "version": {"revision": 1, "created_on": today, "updated_on": today,
                   "status": "active"},
    }


def build_regions(today: str) -> list[dict]:
    out = []
    for slug, level, cuisine_state, parent, summary in REGIONS:
        rid = f"region:{slug}"
        label = summary.split(",")[0].split(" (")[0]
        rec = _envelope(rid, "region", label, today)
        rec["content"] = {"summary": summary}
        rec["structure"] = {"level": level, "cuisine_state": cuisine_state,
                            "parent_region": parent}
        out.append(rec)
    return out


def build_equipment(today: str) -> list[dict]:
    out = []
    for slug, category, summary in EQUIPMENT:
        rid = f"equip:{slug}"
        label = slug.replace("-", " ").title()
        rec = _envelope(rid, "equipment", label, today)
        rec["content"] = {"summary": summary}
        rec["structure"] = {"category": category}
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
    records = build_regions(today) + build_equipment(today)
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
