"""Hand-authored seed records for two of the thinnest canonical-store gaps that
directly feed the two weakest SFT task types: `rel` (substitutes_for predicate,
3/~unknown target -> ingredient_substitution SFT examples) and `problem`
(27/~600 target -> troubleshooting SFT examples).

Found live: an SFT'd model (sft-150m-002) generalized its (correctly fixed)
abstention training into over-refusing legitimate in-domain substitution and
troubleshooting questions -- root-caused to these two categories having far too
little real "here's an answer" signal (3 and 27 examples respectively) relative
to the rest of the training mix. This is the direct fix: more real canonical
content for the SFT serializer to turn into real-answer training examples.

Not synthetic (hand-authored, original culinary content, same footing as
scripts/seed_region_equipment.py and scripts/seed_state_outcome_constraint_
problem.py) -- transformations=["hand_authored"], no source_ids needed per
food_slm.schema.models.Provenance / check_provenance(). Every ingredient id
referenced was checked against data/canonical/ingredients.jsonl before writing
this so nothing here creates a dangling reference.

    python scripts/seed_substitutes_and_problems.py                     # stage only
    python scripts/seed_substitutes_and_problems.py --promote --reviewer NAME
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

# subject_id, object_id, ratio, context, caveat -- all ids verified to already
# exist in data/canonical/ingredients.jsonl.
SUBSTITUTES = [
    ("ing:kokum", "ing:tamarind", "roughly 1:1 by volume",
     "fish curries, Konkani/coastal dishes, rasam",
     "kokum gives a milder, less acidic sourness and a faint pink tint -- won't "
     "match tamarind's sharper tang exactly"),
    ("ing:raw-mango", "ing:tamarind", "a small handful of grated raw mango per "
     "tablespoon of tamarind pulp called for",
     "sambar, pulihora-style rice, dal",
     "raw mango brings its own fruity tartness rather than tamarind's deep sourness"),
    ("ing:amchur", "ing:tamarind", "about 1/2 teaspoon amchur per tablespoon of "
     "tamarind pulp",
     "a quick sour fix when there's no tamarind or kokum on hand",
     "amchur is drier and sharper -- add gradually and taste, it's easy to overdo"),
    ("ing:lemon", "ing:tamarind", "juice of half a lemon per tablespoon of "
     "tamarind pulp, added at the end of cooking",
     "rasam, dal, quick curries",
     "lemon's sourness is brighter and fades faster with heat -- add it off the "
     "flame rather than simmering it like tamarind"),
    ("ing:tamarind", "ing:kokum", "roughly 1:1, soaked and extracted as usual",
     "when kokum isn't available for a Konkani-style curry",
     "loses kokum's distinctive colour and milder sourness"),
    ("ing:chana-dal", "ing:toor-dal", "1:1, but soak chana dal for at least 30 "
     "minutes first",
     "kootu, some sambar variations",
     "chana dal holds its shape more and gives a slightly nuttier, less creamy "
     "result than toor dal"),
    ("ing:moong-dal", "ing:toor-dal", "1:1",
     "sambar, kootu, when toor dal isn't available",
     "moong dal cooks faster and breaks down softer -- reduce cooking time to "
     "avoid an overly mushy result"),
    ("ing:toor-dal", "ing:moong-dal", "1:1, but expect a longer cook time",
     "payasam, pesarattu-adjacent batters calling for moong dal",
     "toor dal won't cook down as smoothly as moong dal and gives a heavier texture"),
    ("ing:coconut-oil", "ing:ghee", "1:1",
     "tempering, frying, any recipe wanting a dairy-free result",
     "loses ghee's characteristic nutty, buttery aroma -- the dish will taste "
     "noticeably different, not just 'the same minus dairy'"),
    ("ing:sesame-oil", "ing:ghee", "1:1",
     "tempering in Chettinad and Andhra-style dishes, where sesame oil is "
     "traditional anyway",
     "sesame oil has its own strong, nutty flavour that's not neutral like some "
     "other oil substitutes"),
    ("ing:ghee", "ing:coconut-oil", "1:1",
     "when a richer, non-coconut flavour is wanted",
     "changes the dish's flavour profile -- ghee is noticeably more buttery/rich"),
    ("ing:jaggery", "ing:sugar", "slightly more jaggery than sugar by volume "
     "(it's less sweet gram-for-gram) -- start 1:1 and adjust to taste",
     "payasam, sweet pongal, most South Indian sweets",
     "jaggery adds a caramel-like depth and darker colour sugar won't give -- "
     "the result will taste noticeably plainer/sweeter rather than complex"),
    ("ing:sugar", "ing:jaggery", "slightly less sugar than the jaggery called "
     "for -- start 1:1 and adjust to taste",
     "when jaggery isn't available",
     "loses jaggery's caramel depth and characteristic colour"),
    ("ing:curd", "ing:buttermilk", "dilute curd with water (roughly equal parts) "
     "to approximate buttermilk's thinner consistency",
     "mor kuzhambu, kadhi-style dishes, tempering bases",
     "curd alone is thicker and tangier than traditional buttermilk -- dilute "
     "and whisk well before using"),
    ("ing:buttermilk", "ing:curd", "use buttermilk at roughly double the volume "
     "the curd quantity called for, since it's thinner",
     "when only buttermilk is on hand for a recipe calling for curd",
     "buttermilk won't thicken a dish the way curd does -- results will be thinner"),
    ("ing:rice-flour", "ing:sago", "not a direct swap -- use about 2/3 the "
     "quantity of rice flour, since sago gives more chew",
     "murukku and some crispy-snack doughs",
     "texture changes noticeably -- sago gives a distinctive translucent chewiness "
     "rice flour alone won't replicate"),
    ("ing:semolina", "ing:rice-flour", "1:1",
     "rava dosa, upma-adjacent batters",
     "semolina gives a coarser, slightly grainy texture compared to rice flour's "
     "smoother result"),
    ("ing:cardamom", "ing:cinnamon", "use a pinch of cinnamon powder in place of "
     "a cardamom pod -- much less, cinnamon is stronger and very different",
     "payasam, sweets, when cardamom specifically isn't available",
     "this is a flavour swap, not a true substitute -- cinnamon tastes quite "
     "different from cardamom's floral sweetness, use only if cardamom is truly "
     "unavailable"),
    ("ing:black-pepper", "ing:green-chilli", "a generous pinch of ground black "
     "pepper roughly in place of one green chilli's worth of heat",
     "rasam, pepper-forward dishes, when a milder, less raw heat is wanted",
     "pepper's heat is warmer and less sharp than green chilli's -- the dish "
     "will taste different, not just milder"),
    ("ing:garlic", "ing:asafoetida", "a small pinch of asafoetida in place of a "
     "clove or two of garlic",
     "tempering, dal, sambar, in cooking that avoids garlic",
     "asafoetida gives a savoury depth rather than garlic's actual flavour -- "
     "it's an approximation, not a true like-for-like swap"),
    ("ing:asafoetida", "ing:garlic", "a small amount of finely chopped garlic in "
     "the tempering, roughly in place of a pinch of asafoetida",
     "when asafoetida isn't available",
     "changes the flavour noticeably -- garlic is pungent and distinct from "
     "asafoetida's savoury-umami note"),
    ("ing:fenugreek-seeds", "ing:fenugreek-powder", "use about half the volume "
     "of powder, since it's more concentrated",
     "idli/dosa batter, tempering",
     "powder disperses instantly and can taste sharper/more bitter if "
     "overdone -- add less than you think and taste"),
    ("ing:turmeric-powder", "ing:turmeric", "grate or pound fresh turmeric "
     "finely, using roughly double the volume of the powder called for",
     "any curry or dal calling for turmeric powder",
     "fresh turmeric is milder and earthier -- colour may be less intensely "
     "yellow than the powdered version"),
    ("ing:tomato", "ing:tamarind", "a very rough swap -- use extra tomato (about "
     "double) in place of tamarind pulp for tang, knowing it won't be sour in "
     "the same way",
     "a last-resort substitution when no souring agent at all is on hand",
     "tomato brings acidity but not tamarind's sharp sourness -- the dish will "
     "taste meaningfully different, this is a compromise, not a real match"),
    ("ing:coconut", "ing:coconut-milk", "roughly 1 cup grated coconut ground "
     "with 1/2 cup water approximates 1 cup thin coconut milk",
     "kootu, thoran, curries calling for coconut milk when only fresh coconut "
     "is on hand",
     "homemade extraction is thinner and less consistent than packaged coconut "
     "milk -- adjust the water used to match the thickness you need"),
    ("ing:bay-leaf", "ing:cinnamon", "one small piece of cinnamon stick in "
     "place of a bay leaf, used sparingly",
     "biryani and pulao-style rice dishes",
     "cinnamon is sweeter and more assertive than bay leaf's subtle aroma -- "
     "use less than you'd think"),
    ("ing:red-chilli", "ing:red-chilli-powder", "about 1/2 teaspoon powder per "
     "whole dried red chilli called for",
     "tempering, any recipe calling for whole dried red chillies",
     "powder disperses heat evenly and faster -- whole chillies give a more "
     "gradual, tempered heat and a visual/textural element powder lacks"),
    ("ing:urad-dal", "ing:urad-dal-split", "1:1, but soaking time may differ "
     "slightly since split dal hydrates faster",
     "idli/dosa batter and most dishes calling for urad dal",
     "split (skinned) dal gives a smoother grind; whole dal retains a bit more "
     "texture"),
    ("ing:black-pepper", "ing:red-chilli-powder", "use about half the quantity "
     "of pepper, since its heat is more pronounced per unit than mild chilli "
     "powder",
     "rasam, pepper rasam specifically, when a different heat profile is wanted",
     "pepper's heat is sharper and warmer, quite different in character from "
     "red chilli's heat -- this changes the dish's identity, not just its spice "
     "level"),
    ("ing:curd", "ing:coconut-milk", "use coconut milk at a similar volume, "
     "understanding the dish becomes dairy-free rather than tangy",
     "when a dairy-free version of a curd-based curry is wanted",
     "loses the tang curd provides entirely -- this changes the dish's flavour "
     "character, it's a dietary swap, not a flavour-matching one"),
]

# description, symptoms, diagnosis, fixes (list), prevention (list), scope
PROBLEMS = [
    ("biryani-rice-mushy", "cooking",
     "Biryani rice turns out mushy and clumped rather than separate, fluffy grains.",
     "Grains stick together and lose their individual shape; the dish looks "
     "more like a thick pilaf than layered biryani.",
     "Too much water relative to rice, overcooking, or stirring the rice too "
     "much after it's added to the pot.",
     ["Use slightly less water than usual next time and avoid stirring once the "
      "rice goes in", "If already mushy, spread it out on a wide tray to cool "
      "and let excess steam escape rather than leaving it covered in the pot"],
     ["Rinse rice well before cooking to remove excess surface starch",
      "Use the dum (sealed, low-heat steam) method rather than boiling the "
      "rice fully in the pot"]),
    ("biryani-bottom-burnt", "cooking",
     "The bottom layer of dum biryani burns or sticks before the dish finishes cooking.",
     "A scorched, bitter smell and a blackened layer at the very bottom of the pot.",
     "Heat too high for the dum stage, or not enough fat/liquid lining the "
     "bottom of the pot before layering.",
     ["Lower the heat immediately and, if using a stovetop, place a tawa "
      "underneath the pot to diffuse heat more evenly",
      "Scrape off only the burnt layer and serve the rest -- the burnt flavour "
      "doesn't usually spread far"],
     ["Grease the bottom of the pot well before layering",
      "Use a heavy-bottomed pot and keep the dum heat on the lowest setting "
      "possible, or finish in a low oven instead of stovetop"]),
    ("murukku-breaks-while-frying", "cooking",
     "Murukku cracks or breaks apart as soon as it hits the hot oil.",
     "Pieces fall apart into fragments rather than frying as one intact spiral.",
     "Dough too dry/crumbly, or not kneaded enough before pressing.",
     ["Knead the dough a little longer with slightly more water or oil until "
      "it holds together smoothly before pressing"],
     ["Rest the dough briefly, covered, before pressing so it stays pliable",
      "Press directly into the hot oil rather than letting shaped pieces sit "
      "and dry out first"]),
    ("murukku-too-hard", "texture",
     "Murukku turns out hard and tough to bite rather than crisp and light.",
     "Dense, difficult to bite through even though it looks properly fried.",
     "Too much rice flour relative to urad dal flour, or oil not hot enough "
     "so it fried too slowly.",
     ["Adjust the flour ratio next time toward a bit more urad dal flour for a "
      "lighter texture", "Fry in properly hot oil so the murukku cooks through "
      "quickly rather than slowly hardening"],
     ["Test oil temperature with a small piece of dough first -- it should "
      "sizzle and rise promptly"]),
    ("payasam-too-thick", "texture",
     "Payasam thickens into something closer to a pudding than a pourable, "
     "spoonable payasam.",
     "The payasam sets firm or very dense as it cools, losing its characteristic "
     "flowing consistency.",
     "Too much reduction of the milk, or too much of the starchy base "
     "ingredient (rice, vermicelli, etc.) relative to liquid.",
     ["Loosen it by warming through with a little extra milk, stirred in "
      "gradually until it reaches the right consistency"],
     ["Remember that payasam continues to thicken as it cools, so take it off "
      "the heat slightly thinner than the final texture you want"]),
    ("payasam-lumpy", "texture",
     "Payasam has lumps rather than a smooth, even consistency.",
     "Small clumps of rice, jaggery, or coconut milk solids visible or felt "
     "while eating.",
     "Jaggery not fully dissolved/strained, or the base ingredient not cooked "
     "down fully before adding milk.",
     ["Strain the jaggery syrup before adding it to remove any undissolved bits "
      "or impurities", "Mash or blend lightly if lumps are from undercooked "
      "rice/dal"],
     ["Always dissolve jaggery in a little hot water and strain it before "
      "adding to the payasam"]),
    ("idiyappam-sticks-together", "texture",
     "Idiyappam strands clump into a dense mass instead of staying separate "
     "and light.",
     "The steamed noodles stick together in a solid lump rather than falling "
     "apart into individual strands.",
     "Dough too wet, or strands pressed too close together on the steaming plate.",
     ["Spread the pressed strands out more loosely on the plate before steaming",
      "If the dough feels too soft, add a little more rice flour next time "
      "before pressing"],
     ["Use water at the right temperature (usually just-boiled, mixed in "
      "quickly) when making the dough -- too much water makes it stick"]),
    ("pesarattu-sticks-to-pan", "cooking",
     "Pesarattu tears or sticks to the pan when trying to flip or remove it.",
     "The batter won't release cleanly, tearing into pieces rather than lifting "
     "off whole.",
     "Pan not hot enough or not seasoned/oiled enough, or batter spread too thick.",
     ["Make sure the pan is properly hot before pouring, and use a thin, even "
      "layer of oil around the edges", "Spread the batter a bit thinner next time"],
     ["Season a cast-iron or non-stick pan well and let it heat fully before "
      "the first pesarattu"]),
    ("spice-powder-clumps", "storage",
     "Ground spice powder (sambar podi, rasam podi, etc.) clumps into hard "
     "lumps during storage.",
     "The powder forms solid chunks instead of staying loose and free-flowing.",
     "Moisture getting into the container, often from a damp spoon or humid "
     "storage conditions.",
     ["Break up the clumps by hand or with a dry spoon before use",
      "Dry-roast the powder briefly in a pan on low heat to drive off moisture, "
      "then cool completely before re-storing"],
     ["Always use a completely dry spoon, and store in an airtight container "
      "away from steam (not right next to the stove)"]),
    ("vatha-kuzhambu-too-oily", "texture",
     "Vatha kuzhambu turns out swimming in oil rather than a balanced, tangy gravy.",
     "A visible oily layer on top that doesn't incorporate into the gravy.",
     "Too much oil used in tempering/cooking relative to the gravy base.",
     ["Skim off the excess oil from the top before serving",
      "Reduce the oil quantity next time, especially in the initial tempering step"],
     ["Use oil mainly for tempering the spices, not as a cooking medium for "
      "the whole gravy"]),
    ("poriyal-watery", "texture",
     "Poriyal turns out watery and soupy instead of dry and lightly coated "
     "with coconut.",
     "Liquid pools at the bottom of the pan instead of the vegetables staying "
     "dry and separate.",
     "Vegetables released too much water during cooking, often from cooking "
     "covered for too long or adding water unnecessarily.",
     ["Cook uncovered on higher heat for the last few minutes to let excess "
      "moisture evaporate"],
     ["Avoid adding extra water -- most vegetables release enough of their own "
      "moisture to cook through", "Cut vegetables evenly so they cook at the "
      "same rate without needing extra liquid"]),
    ("poriyal-undercooked", "doneness",
     "Vegetables in poriyal remain hard/raw in the centre despite looking cooked "
     "on the outside.",
     "A raw, starchy bite in the middle of the vegetable pieces.",
     "Pieces cut too large, or cooked too briefly/on too high heat so the "
     "outside cooked faster than the inside.",
     ["Cover and cook a little longer on lower heat to let the heat penetrate "
      "through before finishing uncovered to dry it out"],
     ["Cut vegetables into smaller, even pieces so they cook through more "
      "quickly and evenly"]),
    ("appam-not-lacy", "texture",
     "Appam comes out dense and flat rather than with the characteristic lacy, "
     "thin edges.",
     "No visible lace pattern at the edges; the appam looks closer to a thick pancake.",
     "Batter not fermented enough, or not swirled properly in the appachatti "
     "while cooking.",
     ["Let the batter ferment longer next time until it's visibly airy",
      "Swirl the appachatti quickly right after pouring the batter so it coats "
      "the sides thinly"],
     ["Make sure the batter has the right thin, pourable consistency -- too "
      "thick won't swirl into a lacy edge"]),
    ("sambar-podi-burnt-roasting", "cooking",
     "Sambar podi (spice powder) tastes bitter and burnt rather than toasty "
     "and aromatic.",
     "A sharp, acrid smell and bitter aftertaste in the finished powder.",
     "Spices roasted on too high heat or for too long before grinding.",
     ["If only lightly over-roasted, use a smaller quantity in cooking to "
      "dilute the bitterness; if badly burnt, it's best to re-roast a fresh batch"],
     ["Dry-roast each spice on low to medium heat, stirring constantly, and "
      "remove from heat the moment it turns fragrant -- don't wait for visible "
      "colour change"]),
    ("dosa-batter-too-thick", "texture",
     "Dosa batter is too thick to spread into a thin, even crepe.",
     "The batter tears or clumps instead of spreading smoothly across the pan.",
     "Not enough water added when grinding or after fermentation.",
     ["Thin the batter with a little water, mixed in gradually until it "
      "reaches a pourable, spreadable consistency"],
     ["Check consistency before fermenting and adjust water then, since "
      "fermentation itself can also thicken the batter slightly"]),
    ("dosa-batter-too-thin", "texture",
     "Dosa batter is too thin and runs off the pan instead of holding a shape.",
     "The batter spreads unevenly into holes or a very thin, patchy dosa.",
     "Too much water added relative to the rice and urad dal.",
     ["Thicken by grinding in a small additional amount of soaked rice or "
      "urad dal, or simply let some water evaporate by resting it a while "
      "uncovered"],
     ["Add water gradually while grinding and check consistency before adding "
      "more, rather than adding it all at once"]),
    ("kesari-lumpy", "texture",
     "Kesari (rava sheera/halwa) has hard, undissolved lumps of rava instead of "
     "a smooth texture.",
     "Gritty or hard bits scattered through an otherwise soft dish.",
     "Rava added to hot water/liquid too quickly without stirring continuously, "
     "or not roasted first.",
     ["Pass it through a sieve to remove the worst lumps, or continue cooking "
      "with extra liquid while stirring vigorously to break them down"],
     ["Roast the rava in ghee first until fragrant, then add the hot liquid "
      "gradually while stirring continuously, not all at once"]),
    ("biryani-meat-tough", "doneness",
     "Meat in biryani remains tough and chewy rather than tender.",
     "Meat pieces are difficult to bite through even though the rice is cooked.",
     "Meat not marinated long enough, or the dum (steaming) stage too short "
     "for the cut of meat used.",
     ["If already served, a quick extra steam (adding a splash of water, "
      "covering, and cooking on low heat a bit longer) can help soften it "
      "somewhat"],
     ["Marinate meat longer, especially tougher cuts, and use a yogurt-based "
      "marinade which helps tenderize", "Match the dum cooking time to the "
      "specific cut and type of meat being used"]),
    ("fish-curry-fish-breaks-apart", "cooking",
     "Fish pieces disintegrate into the curry instead of holding their shape.",
     "Fish turns mushy and loses its distinct pieces when stirred.",
     "Fish stirred too much while cooking, or simmered too long/too vigorously.",
     ["Stop stirring once the fish is added -- gently shake or swirl the pan "
      "instead of using a spoon"],
     ["Add fish pieces near the end of cooking and simmer gently rather than "
      "at a hard boil", "Choose a firmer fish variety if the curry needs more "
      "simmering time"]),
    ("adai-batter-not-fermenting", "fermentation",
     "Adai batter doesn't ferment/rise even after the usual resting time.",
     "The batter looks the same as when first ground, with no bubbles or rise.",
     "Batter kept somewhere too cold, or too little lentil content relative to "
     "rice to drive fermentation.",
     ["Move the batter somewhere warmer and give it more time",
      "If it still won't ferment, adai can still be made from unfermented "
      "batter -- it will be denser but still cook through fine"],
     ["Use a good ratio of lentils to rice (adai traditionally uses more lentils "
      "than idli/dosa batter) and keep it warm while fermenting"]),
    ("ven-pongal-too-watery", "texture",
     "Ven pongal turns out runny and soupy instead of soft but holding together.",
     "The dish pools liquid and doesn't hold its shape on a plate.",
     "Too much water relative to rice and dal, or not cooked down enough at the end.",
     ["Continue cooking uncovered, stirring occasionally, until the excess "
      "water evaporates and it thickens to the right consistency"],
     ["Use the standard ratio (roughly 1 part dal to 4-5 parts rice, with "
      "water about 4-5 times the combined grain) and adjust gradually rather "
      "than adding all the water upfront"]),
    ("pickle-too-salty", "flavour",
     "Pickle tastes overwhelmingly salty rather than balanced.",
     "Salt dominates over the sour/spicy notes of the pickle.",
     "Too much salt added relative to the amount of main ingredient, often from "
     "over-correcting for preservation.",
     ["Mix in some unsalted, freshly prepared portion of the same vegetable/fruit "
      "to dilute the overall salt level, if practical",
      "Serve smaller portions alongside other dishes to balance the saltiness "
      "in the overall meal"],
     ["Measure salt carefully rather than adding by feel, especially for "
      "pickles meant for long storage where it's tempting to over-salt"]),
    ("pickle-oil-separates", "storage",
     "Oil in a pickle separates and floats visibly rather than staying "
     "incorporated.",
     "A distinct oil layer on top of the jar that doesn't recombine when stirred.",
     "Normal for oil-based pickles over time, but excessive separation usually "
     "means too much oil was used relative to the solid content.",
     ["Simply stir or shake before serving -- this is largely cosmetic unless "
      "the proportions were badly off"],
     ["Use oil in the proportion traditional for that specific pickle recipe "
      "rather than adding extra 'for safety'"]),
    ("coconut-chutney-turns-pink", "storage",
     "Coconut chutney develops a pinkish tinge after sitting for a while.",
     "A faint pink or reddish discolouration appears in white coconut chutney "
     "that wasn't there when freshly made.",
     "A natural enzymatic reaction in fresh coconut when exposed to air and "
     "certain ingredients like tamarind or chilli over time -- not spoilage on "
     "its own, but can indicate it's aging.",
     ["It's generally still fine to eat shortly after this appears, but check "
      "smell and taste before serving", "Make chutney fresh and consume within "
      "a day for best colour and taste"],
     ["Store coconut chutney covered and refrigerated, and consume promptly "
      "rather than keeping it for multiple days"]),
    ("rasam-too-thin-no-body", "texture",
     "Rasam tastes watery and lacks body, even though the flavour seasoning is right.",
     "Thin, almost like flavoured water, missing the usual light-but-rounded "
     "mouthfeel.",
     "Not enough dal water/stock used as the base, or dal cooked separately and "
     "not incorporated.",
     ["Add a little mashed cooked dal or dal cooking water to give it body"],
     ["Always use the water dal was cooked in as part of the rasam base rather "
      "than plain water"]),
    ("biryani-layers-not-separating", "texture",
     "Biryani comes out as one uniform mass rather than distinct layers of "
     "rice and meat/vegetables.",
     "No visible separation when serving -- everything is mixed together "
     "rather than layered.",
     "Not enough care taken while layering, or too much mixing/stirring during "
     "the dum stage.",
     ["Avoid stirring once layers are assembled -- serve by scooping down "
      "through the layers rather than mixing first"],
     ["Layer deliberately (rice, then meat/veg, then rice again) and avoid "
      "disturbing the layers until it's time to serve"]),
    ("dum-biryani-steam-escaping", "cooking",
     "Steam escapes during the dum stage, and the biryani doesn't cook evenly "
     "or stay moist.",
     "Rice near the top stays undercooked while the pot loses visible steam "
     "from the lid.",
     "The pot lid isn't sealed well enough to trap steam.",
     ["Seal the lid with a simple dough strip (atta) around the rim, or use a "
      "tight-fitting heavy lid with a damp cloth underneath it"],
     ["Check that the pot and lid fit tightly before starting the dum stage, "
      "and seal any gaps before placing on low heat"]),
    ("idli-stand-water-runs-out", "cooking",
     "The steaming water runs low or boils dry partway through steaming idlis.",
     "Idlis near the end of steaming look uncooked or a burning smell starts "
     "from the pot.",
     "Not enough water added at the start for the expected steaming duration.",
     ["If caught in time, carefully add more hot water to the pot without "
      "disturbing the idli stand, and continue steaming"],
     ["Fill the steaming vessel with enough water for the full expected "
      "steaming time before starting, checking the level if steaming a large batch"]),
    ("papad-sticking-together-storage", "storage",
     "Stored papads stick together and tear when trying to separate them.",
     "Papads clump into a stack that won't pull apart cleanly.",
     "Humidity getting into the storage container, softening the papads.",
     ["Separate them carefully while still slightly pliable rather than fully "
      "dried and brittle"],
     ["Store in an airtight container with a moisture absorber if the climate "
      "is humid, and keep them fully dried before stacking"]),
    ("tamarind-extract-too-thick", "texture",
     "Tamarind extract comes out too thick and pulpy instead of a smooth, "
     "pourable liquid.",
     "Fibrous bits and a paste-like consistency rather than a strained liquid.",
     "Not strained well enough, or too little water used when soaking/mashing.",
     ["Add a little more water and strain again through a fine sieve, "
      "pressing the pulp to extract remaining liquid"],
     ["Soak tamarind in enough warm water and strain thoroughly, discarding "
      "the fibrous solids rather than including them"]),
    ("gongura-pachadi-too-bitter", "flavour",
     "Gongura pachadi tastes unexpectedly bitter rather than pleasantly tangy.",
     "A sharp bitterness that overwhelms the characteristic sourness of the dish.",
     "Gongura leaves cooked too long at high heat, or older/tougher leaves used.",
     ["Balance with a bit more jaggery or oil-tempered spice to offset the "
      "bitterness in that batch"],
     ["Use young, tender gongura leaves where possible, and avoid prolonged "
      "high-heat cooking which can bring out bitterness"]),
]


def _envelope(rid: str, rtype: str, label: str, today: str) -> dict:
    return {
        "id": rid, "type": rtype, "schema_version": "2", "labels": [label],
        "provenance": {"source_ids": [], "transformations": ["hand_authored"],
                       "acquired_on": today, "synthetic": False},
        "quality": {"confidence": 0.85, "validation_status": "human_validated",
                   "corroboration_count": 1},
        "version": {"revision": 1, "created_on": today, "updated_on": today,
                   "status": "active"},
    }


def _rel_id(subj: str, obj: str) -> str:
    return f"rel:{subj.split(':', 1)[1]}-substitutes-for-{obj.split(':', 1)[1]}"


def build_substitutes(today: str) -> list[dict]:
    out = []
    for subj, obj, ratio, context, caveat in SUBSTITUTES:
        label = f"{subj.split(':', 1)[1].replace('-', ' ')} substitutes_for " \
               f"{obj.split(':', 1)[1].replace('-', ' ')}"
        rec = _envelope(_rel_id(subj, obj), "rel", label, today)
        rec["content"] = {"note": caveat}
        rec["structure"] = {"predicate": "substitutes_for", "subject_id": subj,
                            "object_id": obj, "directional": True,
                            "attributes": {"ratio": ratio, "context": context,
                                          "caveat": caveat}}
        out.append(rec)
    return out


def build_problems(today: str) -> list[dict]:
    out = []
    for slug, scope, description, symptoms, diagnosis, fixes, prevention in PROBLEMS:
        rid = f"problem:{slug}"
        label = slug.replace("-", " ").title()
        rec = _envelope(rid, "problem", label, today)
        rec["content"] = {"description": description, "symptoms": symptoms,
                          "diagnosis": diagnosis, "fixes": fixes, "prevention": prevention}
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
    records = build_substitutes(today) + build_problems(today)
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
