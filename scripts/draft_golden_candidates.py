"""First batch of golden-QA-set candidates, hand-drafted by Claude citing the
primary sources in data/eval/golden_sources/ (fetched via fetch_golden_sources.py).

This is a DRAFT, not a golden set: every item needs human verification before it
counts as golden (evaluation_strategy.md SS5 -- "never AI generation" for the
golden set itself; this script's output is explicitly the pre-verification
candidate pool, reviewed one item at a time in scripts/golden_qa_review.py).

Deliberately thin / absent task types, and why:
  - scaling: needs a source with real quantities (a cookbook/recipe, not an
    encyclopedia entry) -- none of the fetched sources qualify. Needs its own
    sourcing pass.
  - troubleshooting: encyclopedia text doesn't describe failure modes ("why did
    my X go wrong"). Needs a cooking-guide-style source, not Wikipedia.
  - Puducherry has no representation yet -- no good primary source found in this
    batch; flagged as a gap for the next sourcing pass.

Re-run fetch_golden_sources.py with more URLs and extend CANDIDATES below to
grow toward the ~500-1,000 target -- this is meant to be grown over many
sessions, not a one-shot.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "eval" / "golden_qa_candidates.jsonl"


def _c(task_type: str, region: str | None, diet: str, question: str, answer: str,
      source_url: str, source_quote: str, dish_ids: list[str] | None = None) -> dict:
    return {
        "task_type": task_type, "region": region, "diet": diet,
        "question": question, "candidate_answer": answer,
        "source": {"url": source_url, "quote": source_quote},
        "dish_ids": dish_ids or [],
        "status": "pending", "reviewer_notes": "",
        "drafted_by": "claude (candidate only -- requires human verification before use as golden)",
    }


CANDIDATES: list[dict] = [
    # -- definition: ingredients --------------------------------------------
    _c("definition", None, "vegetarian",
      "What is tamarind and what does it add to a dish?",
      "Tamarind is the pod-like fruit of a leguminous tree (Tamarindus indica), native to "
      "tropical Africa and naturalized in Asia. Its pulp is sweet-and-tangy, and in South "
      "Indian cooking it's used to add a sour note to dishes like sambar and rasam.",
      "https://en.wikipedia.org/wiki/Tamarind",
      "The tamarind tree produces brown, pod-like fruits that contain a sweet, tangy pulp, "
      "which is used in cuisines around the world.",
      ["ing:tamarind"]),

    _c("definition", None, "na",
      "What is asafoetida used for in Indian cooking?",
      "Asafoetida is a dried resin (gum oleoresin) from the root of Ferula plants. In Indian "
      "and Middle Eastern cooking, it's used to add a savoury, umami depth, and specifically "
      "as a substitute for or complement to onion and garlic flavour.",
      "https://en.wikipedia.org/wiki/Asafoetida",
      "As a culinary ingredient, asafoetida is primarily used in Indian and Middle Eastern "
      "cuisines, where it is employed to complement or substitute for alliaceous ingredients "
      "such as garlic and onion.",
      ["ing:asafoetida"]),

    _c("definition", None, "na",
      "What is jaggery and how is it made?",
      "Jaggery is a traditional unrefined cane (or palm) sugar. It's made by boiling raw "
      "sugarcane juice or palm sap in large shallow vessels until it concentrates into a "
      "solid block or paste, without separating out the molasses -- which is why it's darker "
      "and less refined than white sugar.",
      "https://en.wikipedia.org/wiki/Jaggery",
      "Traditionally, the syrup is made by boiling raw sugarcane juice or palm sap in large, "
      "shallow, round-bottomed vessels.",
      ["ing:jaggery"]),

    _c("definition", "andhra_pradesh", "vegetarian",
      "What is gongura?",
      "Gongura is a variety of roselle plant (Hibiscus sabdariffa) grown for its tart, edible "
      "leaves. It's especially associated with Andhra cuisine, most famously in gongura "
      "pachadi, a tangy chutney/pickle made from the leaves.",
      "https://en.wikipedia.org/wiki/Gongura",
      "These leaves are used in south-central Indian cuisine to impart a tart flavour. "
      "Gongura'pacchadi, a form of chutney or relish, is a quintessential part of Andhra "
      "cuisine.",
      ["ing:gongura"]),

    # -- definition: dishes ---------------------------------------------------
    _c("definition", "pan_south", "vegetarian",
      "What is rasam?",
      "Rasam is a thin, spiced, tangy South Indian soup, usually made from a tamarind or "
      "kokum base with tomato and lentil, and served as part of a meal alongside rice. It's "
      "known by different names across the South -- saaru in Kannada, chaaru in Telugu.",
      "https://en.wikipedia.org/wiki/Rasam_(soup)",
      "Rasam...is a spicy South Indian soup-like dish. It is usually served as a side dish "
      "with rice...Rasam has a distinct taste in comparison to sambar due to its own "
      "seasoning ingredients and is watery in consistency.",
      ["dish:rasam"]),

    _c("definition", "tamil_nadu", "vegetarian",
      "What is payasam?",
      "Payasam is a South Indian milk-based pudding, typically made by simmering rice (or "
      "vermicelli, sago, etc.) with milk and sweetened with sugar or jaggery, often flavoured "
      "with cardamom. It's the South Indian name for what's called kheer elsewhere in India.",
      "https://en.wikipedia.org/wiki/Payasam",
      "Kheer, khir, payesh, fenni, or payasam is a pudding or porridge (specifically rice "
      "pudding) popular in the Indian subcontinent, usually made by boiling milk, sugar or "
      "jaggery, and rice.",
      ["dish:payasam"]),

    _c("definition", "tamil_nadu", "vegetarian",
      "What is pongal (the dish, not the festival)?",
      "Pongal is rice cooked in boiling milk, and is the dish traditionally prepared for the "
      "Tamil harvest festival of the same name. The savoury version, ven pongal, adds lentils, "
      "pepper, and ghee; the sweet version, sakkarai pongal, is made with jaggery.",
      "https://en.wikipedia.org/wiki/Pongal_(dish)",
      "Pongal (lit. 'to boil over') is a dish of rice cooked in boiling milk...Several "
      "varieties such as the Venn pongal and Sakkarai pongal are part of the Tamil cuisine.",
      ["dish:pongal"]),

    _c("definition", "kerala", "vegetarian",
      "What is puttu?",
      "Puttu is a steamed breakfast dish from Kerala and Tamil Nadu made of coarsely ground "
      "rice flour layered with grated coconut, steamed in cylindrical layers in a special "
      "puttu kutti vessel.",
      "https://en.wikipedia.org/wiki/Puttu",
      "It is made of steamed cylinders of ground rice layered with coconut shavings...Puttu "
      "is generally cooked in an aluminium puttu kutti vessel with two sections.",
      ["dish:puttu"]),

    _c("definition", "kerala", "vegetarian",
      "What is appam?",
      "Appam is a thin, bowl-shaped pancake made from fermented rice batter and coconut milk, "
      "traditionally cooked in a curved pan called an appachatti. It's popular in Kerala and "
      "Tamil Nadu, usually served for breakfast or dinner with a curry.",
      "https://en.wikipedia.org/wiki/Appam",
      "It is made with fermented rice batter and coconut milk, traditionally cooked in an "
      "appachatti, a deep pan similar in shape to a wok.",
      ["dish:appam"]),

    _c("definition", "karnataka", "vegetarian",
      "What is bisi bele bath?",
      "Bisi bele bath is a spicy rice-and-lentil dish from Karnataka, said to have originated "
      "in the Mysore Palace kitchens. It's made with toor dal, rice, vegetables, ghee, and a "
      "distinctive spice blend (huliyanna masala).",
      "https://en.wikipedia.org/wiki/Bisi_bele_bath",
      "Bisi bele bhath...is a spicy, rice-based dish with origins in the state of Karnataka, "
      "India. It is said to have originated in the Mysore Palace.",
      ["dish:bisi-bele-bath"]),

    # -- technique_explanation -------------------------------------------------
    _c("technique_explanation", "pan_south", "na",
      "How is idli batter fermented and why does it rise?",
      "Idli batter (rice and de-husked black gram, roughly 4:1) is left to ferment overnight, "
      "during which naturally occurring lactic acid bacteria -- mainly Leuconostoc "
      "mesenteroides -- produce both lactic acid and carbon dioxide. The carbon dioxide is "
      "what makes the batter rise, often more than doubling in volume.",
      "https://en.wikipedia.org/wiki/Idli",
      "The fermentation process is performed by lactic acid bacteria, particularly "
      "'Leuconostoc mesenteroides' and 'Enterococcus faecalis,' which generate both lactic "
      "acid and carbon dioxide...its volume will more than double.",
      ["tech:ferment", "dish:idli"]),

    _c("technique_explanation", "telangana", "non_vegetarian",
      "What does 'dum' cooking mean in Hyderabadi biryani?",
      "Dum cooking means sealing the cooking vessel (traditionally with a ring of dough "
      "around the lid) and letting the dish cook slowly in its own trapped steam. For "
      "Hyderabadi biryani, the meat and partly-cooked rice are layered in the pot, which is "
      "then sealed and cooked 'in dum' so the flavours combine without opening the lid.",
      "https://en.wikipedia.org/wiki/Hyderabadi_biryani",
      "The meat is layered at the bottom of special Hyderabadi Biryani degh (vessel) layered "
      "with fragrant basmati rice and cooked 'in dum' after sealing the handi (vessel) with "
      "dough.",
      ["tech:dum-cooking", "dish:hyderabadi-biryani"]),

    _c("technique_explanation", "kerala", "vegetarian",
      "How is puttu traditionally steamed?",
      "Puttu is steamed in a two-part vessel called a puttu kutti: the lower section holds "
      "boiling water, and the upper, perforated section holds the rice-flour-and-coconut "
      "mixture, so steam passes up through it and cooks the puttu without direct contact "
      "with the water.",
      "https://en.wikipedia.org/wiki/Puttu",
      "Puttu is generally cooked in an aluminium puttu kutti vessel with two sections. The "
      "lower section holds water and the upper section holds the puttu...Perforated lids "
      "separate the sections to allow the steam to pass between them.",
      ["tech:steaming", "equip:puttu-kutti", "dish:puttu"]),

    _c("technique_explanation", None, "na",
      "How is jaggery traditionally made from sugarcane?",
      "Sugarcane is crushed to extract the juice, which is then boiled for a long stretch in "
      "a large shallow vessel. As it boils, the water evaporates and the juice concentrates "
      "into a thick syrup, which is then poured into moulds to solidify into blocks -- "
      "nothing is separated out, which is why jaggery keeps the molasses and minerals that "
      "refined sugar loses.",
      "https://en.wikipedia.org/wiki/Jaggery",
      "The cut and cleaned sugarcane is crushed and the extracted cane juice is collected in "
      "a large vessel...The vessel is heated for about an hour.",
      ["ing:jaggery"]),

    # -- regional_comparison -----------------------------------------------
    _c("regional_comparison", None, "vegetarian",
      "How does Kerala-style avial differ from the classic/Tamil Nadu version?",
      "Classic avial is thick, while the Central Travancore (Kerala) variety has a "
      "noticeably thinner gravy. There's also local variation within Kerala itself: the "
      "Kozhikode-region version adds bitter gourd, and the Kollam-region version adds "
      "tomato, which aren't part of the base recipe.",
      "https://en.wikipedia.org/wiki/Avial",
      "Central Travancore has a slightly different variety of avial with its thin gravy; "
      "whereas the classic avial is thick...the Kozhikode region includes bitter gourd... "
      "Kollam region also includes tomato among other vegetables.",
      ["dish:avial"]),

    _c("regional_comparison", None, "vegetarian",
      "Is dosa the same everywhere in South India?",
      "No -- the Tamil Nadu style of dosa is traditionally softer and thicker, while the "
      "thinner, crispier dosa that later became popular across India is said to have "
      "originated in present-day Karnataka.",
      "https://en.wikipedia.org/wiki/Dosa_(food)",
      "The Tamil Nadu dosa is traditionally softer and thicker; the thinner and crispier "
      "version of dosa, which became popular across India, was first made in present-day "
      "Karnataka.",
      ["dish:dosa"]),

    _c("regional_comparison", None, "vegetarian",
      "How does sambar differ between Tamil Nadu and the Tulu-speaking coastal areas?",
      "Tamil Nadu sambar is a tamarind-based lentil-and-vegetable stew. In Tulu-speaking "
      "coastal areas, the equivalent dish (called koddel) is made with coconut instead -- "
      "grated coconut is ground with the spices rather than relying mainly on tamarind.",
      "https://en.wikipedia.org/wiki/Sambar_(dish)",
      "Coconut is predominantly used to prepare koddel (sambar). Grated coconut is ground "
      "with spices...Each region has its own version of making sambar; even though the same "
      "vegetables are used, the condiments and ingredients, as well as the methods, differ.",
      ["dish:sambar"]),

    # -- ingredient_substitution --------------------------------------------
    _c("ingredient_substitution", None, "vegetarian",
      "What can I use instead of garlic and onion in a dish that needs to avoid them?",
      "Asafoetida is the traditional substitute: it's used specifically to stand in for the "
      "savoury depth that garlic and onion would otherwise add, which is why it's common in "
      "Jain and other onion-garlic-free South Indian cooking.",
      "https://en.wikipedia.org/wiki/Asafoetida",
      "As a culinary ingredient, asafoetida is primarily used in Indian and Middle Eastern "
      "cuisines, where it is employed to complement or substitute for alliaceous ingredients "
      "such as garlic and onion.",
      ["ing:asafoetida"]),

    # -- recipe_request (kept to verifiable ingredients/process only -- no
    # fabricated quantities, since encyclopedia text doesn't give them) -------
    _c("recipe_request", "pan_south", "vegetarian",
      "What is idli made from and how is it prepared, broadly?",
      "Idli batter is made from idli rice (or parboiled rice) and de-husked black gram, "
      "roughly in a 4:1 ratio, ground and left to ferment overnight -- the batter should "
      "more than double in volume. The fermented batter is then poured into a perforated, "
      "greased idli tray and steamed, typically for 10-25 minutes depending on size.",
      "https://en.wikipedia.org/wiki/Idli",
      "four parts uncooked rice (idli rice or parboiled rice) to one part whole white lentil "
      "(black gram, Vigna mungo)...The finished idli batter is put into a perforated greased "
      "idli tray for steaming...about 10-25 minutes, depending on size.",
      ["dish:idli"]),

    _c("recipe_request", "andhra_pradesh", "vegetarian",
      "What is pesarattu made from and how is it cooked?",
      "Pesarattu batter is made by soaking whole green gram for four to seven hours, then "
      "grinding it to a smooth paste with green chillies, ginger, and salt. The batter is "
      "poured onto a hot griddle like a dosa and cooked until crisp, usually flipped partway "
      "to cook it through, and served with ginger or peanut chutney.",
      "https://en.wikipedia.org/wiki/Pesarattu",
      "soaking green gram for four to seven hours, then grinding it to a smooth paste with a "
      "couple of green chilies, a small piece of ginger, and some salt. The batter is then "
      "poured onto a heated pan...the pesarattu is usually flipped, to cook it completely... "
      "typically served with ginger chutney, peanut chutney or tamarind chutney.",
      ["dish:pesarattu"]),

    # -- abstention: behavior, not facts -- no source citation needed --------
    _c("abstention", None, "na",
      "Can you give me a recipe for Punjabi chole bhature?",
      "I'm specialized in South Indian cuisine specifically, so I don't have reliable "
      "knowledge of North Indian dishes like chole bhature. I'd be glad to suggest a South "
      "Indian dish instead.",
      "", ""),

    _c("abstention", None, "na",
      "What's the best way to cook a Thai green curry?",
      "That's outside what I know about -- I'm focused specifically on South Indian cuisine, "
      "not Thai food.",
      "", ""),

    _c("abstention", None, "na",
      "How do I treat a burn from hot oil while cooking?",
      "I can't give medical advice -- please run cool water over it and see a doctor if it's "
      "serious. I'm a cooking specialist, not a medical resource.",
      "", ""),

    # =========================================================================
    # Batch 2: broader regional coverage (esp. Puducherry, Telangana, Karnataka,
    # Andhra Pradesh), more techniques/ingredients, and cross-region naming
    # comparisons -- sourced from fetch_golden_sources.py's second URL batch.
    # =========================================================================

    # -- definition: dishes, Puducherry -----------------------------------
    _c("definition", "puducherry", "na",
      "What makes Puducherry's cuisine distinctive?",
      "Puducherry's cuisine is a fusion of Tamil and French influence, a legacy of its "
      "history as a French colonial outpost. French dishes like crepes and crème brulee are "
      "still prepared there alongside South Indian staples like dosa and sambar, and Tamil "
      "cooking in the region was refined by French influence to use noticeably less spice.",
      "https://en.wikipedia.org/wiki/Cuisine_of_Pondicherry",
      "Food in Pondicherry is strongly influenced by the French. Salade nicoise, crepes (both "
      "sweet and savoury) and creme brulee are among popular French dishes still prepared in "
      "Pondicherry...Tamil food was refined by French influence, using much less spice.",
      []),

    _c("definition", "puducherry", "vegetarian",
      "What is paruppusilli?",
      "Paruppusilli is a traditional Tamil dish made by cooking toor dal and bengal gram dal "
      "with red chillies and french beans (or cluster beans), flavoured with asafoetida, "
      "curry leaves, and mustard seeds. It's typically served with rice and mor kuzhambu.",
      "https://en.wikipedia.org/wiki/Paruppusilli",
      "It is made by cooking toor dal and bengal gram dal with red chillies and french "
      "beans/cluster beans/plantain flowers, and is flavored with asafoetida, curry leaves "
      "and mustard seeds. It is typically served with rice and Mor Kuzhambu during a meal.",
      ["dish:paruppusilli"]),

    # -- definition: dishes, Telangana ----------------------------------------
    _c("definition", "telangana", "vegetarian",
      "What is sakinalu?",
      "Sakinalu is a crunchy snack from northern Telangana made of concentric circles of "
      "rice-flour dough, deep fried. It's traditionally prepared for the Makar Sankranti "
      "harvest festival.",
      "https://en.wikipedia.org/wiki/Sakinalu",
      "It consists of concentric circles made of rice flour dough, fried in oil. It is "
      "prepared during the Makar Sankranti festival.",
      ["dish:sakinalu"]),

    _c("definition", "telangana", "vegetarian",
      "What is khubani ka meetha?",
      "Khubani ka meetha is a Hyderabadi dessert made from dried apricots boiled with syrup "
      "until it reaches a thick, compote-like consistency, usually topped with blanched "
      "almonds and served with thick cream (malai).",
      "https://en.wikipedia.org/wiki/Qubani_ka_meetha",
      "Preparation of the dish involves boiling apricots with syrup until they assume the "
      "consistency of a thick soup or compote. The dessert is topped with blanched almonds "
      "or apricot kernels, and is traditionally garnished with malai.",
      ["dish:khubani-ka-meetha"]),

    _c("definition", "telangana", "non_vegetarian",
      "What is haleem?",
      "Haleem is a slow-cooked stew of wheat or barley, lentils, and meat (commonly beef, "
      "mutton, or chicken), cooked for hours and then beaten to a thick, paste-like "
      "consistency. The Hyderabadi version is a well-known regional variant.",
      "https://en.wikipedia.org/wiki/Haleem",
      "It is made by slow cooking the meat in barley...This dish is slow-cooked for seven to "
      "eight hours and then vigorously stirred or beaten with a pestle-like stirring stick. "
      "This results in a paste-like consistency.",
      ["dish:haleem"]),

    _c("definition", "telangana", "vegetarian",
      "What is mirchi ka salan?",
      "Mirchi ka salan is a Hyderabadi curry of green chilli peppers in a ground-peanut-based "
      "gravy, usually served alongside biryani. Sometimes tomato or eggplant is substituted "
      "for the chillies instead.",
      "https://en.wikipedia.org/wiki/Mirchi_ka_salan",
      "The mirchi (chilli peppers) are cooked in spices and mixed with a ground peanut paste "
      "which gives the dish a grainy texture. Sometimes the chillies can also be substituted "
      "with tomatoes...and eggplant.",
      ["dish:mirchi-ka-salan"]),

    # -- definition: dishes, Karnataka -----------------------------------------
    _c("definition", "karnataka", "vegetarian",
      "What is Mysore pak?",
      "Mysore pak is a dense, fudge-like Karnataka sweet made from gram flour, sugar, and "
      "ghee, often flavoured with cardamom. It was created in the royal kitchens of the "
      "Kingdom of Mysore.",
      "https://en.wikipedia.org/wiki/Mysore_Pak",
      "a traditional Indian confectionery made primarily from gram flour (besan), sugar, and "
      "ghee and often flavoured with cardamom. It originated in the royal kitchens of the "
      "Kingdom of Mysore.",
      ["dish:mysore-pak"]),

    _c("definition", "karnataka", "vegetarian",
      "What is ragi mudde?",
      "Ragi mudde is a staple finger-millet dish from Karnataka, made by stirring ragi flour "
      "into boiling water until it forms a smooth, lump-free dough, then shaping it into "
      "rounds. It has only two ingredients: ragi flour and water.",
      "https://en.wikipedia.org/wiki/Ragi_mudde",
      "Ragi mudde has only two ingredients: ragi (finger millet) flour, and water...it "
      "requires immediate mixing (to avoid lumps)...the flour is beaten to a smooth "
      "dough-like consistency with no lumps.",
      ["dish:ragi-mudde"]),

    _c("definition", "karnataka", "vegetarian",
      "What is neer dosa?",
      "Neer dosa is a thin rice crepe from the Tulu Nadu region of coastal Karnataka and "
      "Malabar in Kerala. Unlike most dosas, it needs no fermentation -- just soaked, "
      "finely-ground rice batter and salt.",
      "https://en.wikipedia.org/wiki/Neer_dosa",
      "Unlike other dosas neer dosa is known for its simple preparation method and lack of "
      "fermentation...Fermentation of the rice is not required to prepare Neer dosa.",
      ["dish:neer-dosa"]),

    # -- definition: dishes, Andhra Pradesh -------------------------------------
    _c("definition", "andhra_pradesh", "vegetarian",
      "What is pulihora?",
      "Pulihora (also called puliyogare or tamarind rice) is a tangy South Indian rice dish "
      "souring with tamarind, kokum, or green mango, seasoned with lentils, peanuts, jaggery, "
      "curry leaves, and turmeric. A recipe for it is recorded in Thanjavur temple-kitchen "
      "inscriptions dating back to around 1010 CE.",
      "https://en.wikipedia.org/wiki/Pulihora",
      "The pulihora is prepared with kokum, tamarind, lemon or green mango as main souring "
      "ingredient, along with a blend of spices, lentils, peanuts and jaggery cooked in oil "
      "and mixed with cooked rice along with fried curry leaves and turmeric.",
      ["dish:pulihora"]),

    _c("definition", "andhra_pradesh", "vegetarian",
      "What is pootharekulu?",
      "Pootharekulu is an Andhra Pradesh sweet made of a wafer-thin, paper-like layer of rice "
      "starch, stuffed with sugar or jaggery, ghee, and nuts. It originated in Atreyapuram, "
      "East Godavari district, and has a Geographic Indication (GI) tag.",
      "https://en.wikipedia.org/wiki/Pootharekulu",
      "The sweet is wrapped in a wafer-thin rice starch layer resembling paper and is stuffed "
      "with sugar, dry fruits and nuts...created in Atreyapuram, a village...in the East "
      "Godavari district of Andhra Pradesh.",
      ["dish:pootharekulu"]),

    # -- definition: dishes, Tamil Nadu -----------------------------------------
    _c("definition", "tamil_nadu", "vegetarian",
      "What is murukku?",
      "Murukku is a crunchy, spiral-shaped fried snack made mainly from rice flour and urad "
      "dal flour mixed with water, salt, chilli powder, asafoetida, and sesame or cumin "
      "seeds. Its name comes from the Tamil word for 'twisting,' referring to its shape.",
      "https://en.wikipedia.org/wiki/Murukku",
      "Murukku is typically made from rice flour and urad dal flour. The flours are mixed "
      "with water, salt, chili powder, asafoetida and either sesame seeds or cumin seeds..."
      "The name muRukku 'twisting' in Tamil refers to its shape.",
      ["dish:murukku"]),

    _c("definition", "tamil_nadu", "vegetarian",
      "What is adhirasam?",
      "Adhirasam is a doughnut-shaped fried sweet made from rice flour, jaggery, and ghee, "
      "sometimes with coconut and spices like cardamom. It's traditionally offered during "
      "Deepavali and Ganesha Chaturthi in Tamil Nadu and Karnataka.",
      "https://en.wikipedia.org/wiki/Adhirasam",
      "a type of Indian sweet made out of rice, jaggery, ghee and sometimes coconut and with "
      "spices like cardamom...Adhirasam is a popular as an offering...during Deepavali and "
      "Ganesha Chaturthi festivals.",
      ["dish:adhirasam"]),

    _c("definition", "tamil_nadu", "vegetarian",
      "What is kuzhambu?",
      "Kuzhambu is a tamarind-based curry from Tamil cuisine that can be made with "
      "vegetables, meat, or dal, ranging from a thin broth to a thick gravy. It's a common "
      "misconception that it always contains lentils -- most varieties don't, beyond what's "
      "used in the tempering.",
      "https://en.wikipedia.org/wiki/Kuzhambu",
      "is a tamarind-based curry in Tamil cuisine...It is a common misconception that "
      "lentils (dal) are a staple ingredient in Kuzhambu...the vast majority do not use dal "
      "beyond the small quantity used as a spice or during tempering.",
      ["dish:kuzhambu"]),

    # -- definition: dishes, Kerala -----------------------------------------
    _c("definition", "kerala", "vegetarian",
      "What is thoran?",
      "Thoran is a dry Kerala side dish of finely chopped vegetables (like cabbage, beans, or "
      "jackfruit) sauteed briefly with grated coconut, mustard seeds, curry leaves, and "
      "turmeric. It's part of the traditional Keralite sadhya meal, and is called poriyal in "
      "Tamil Nadu.",
      "https://en.wikipedia.org/wiki/Thoran",
      "a class of dry vegetable dishes combined with coconut...In Tamil Nadu, a similar dish "
      "is known as Poriyal...The chopped vegetable is mixed with grated coconut, mustard "
      "seeds, curry leaves and turmeric powder and briefly sauteed.",
      ["dish:thoran"]),

    _c("definition", "kerala", "non_vegetarian",
      "What is fish molee?",
      "Fish molee (meen molee) is a mild, coconut-based fish curry of possible Portuguese or "
      "Indian origin, common in Kerala and also found in Malaysia and Singapore due to "
      "historical trade and migration links.",
      "https://en.wikipedia.org/wiki/Fish_molee",
      "Fish moilee/moily or fish molee (meen molee) is a spicy fish and coconut dish of "
      "possible Portuguese or Indian origin. It is common in India, Malaysia and Singapore.",
      ["dish:fish-molee"]),

    # -- technique_explanation -------------------------------------------------
    _c("technique_explanation", "pan_south", "na",
      "What is tempering (tadka) and when do you do it?",
      "Tempering (tadka) is briefly frying whole spices -- like mustard seeds, cumin, dried "
      "chillies, or curry leaves -- in hot oil or ghee to release their essential oils, then "
      "pouring the spiced oil into the dish. It's usually done at the start of cooking before "
      "the other ingredients go in, but for dals, sambar, or similar dishes it's often done "
      "at the end instead, right before serving.",
      "https://en.wikipedia.org/wiki/Tadka",
      "whole spices...are cooked briefly in oil or ghee to liberate essential oils from cells "
      "and thus enhance their flavours, before being poured, together with the oil, into a "
      "dish...Tempering is typically done at the beginning of cooking...or it may be done at "
      "the end...(as with a dal, sambar or stew).",
      ["tech:tempering"]),

    _c("technique_explanation", "pan_south", "na",
      "In what order should you add spices when tempering?",
      "When tempering with multiple ingredients, add them in succession based on how long "
      "each needs to cook -- ones that need longer cooking go in earlier, and quicker-cooking "
      "or more delicate ones (like curry leaves) go in last, so nothing burns.",
      "https://en.wikipedia.org/wiki/Tadka",
      "When using multiple ingredients in tempering, they are often added in succession, "
      "with those requiring longer cooking added earlier, and those requiring less cooking "
      "added later.",
      ["tech:tempering"]),

    _c("technique_explanation", None, "na",
      "What actually happens during fermentation?",
      "Fermentation is the conversion of carbohydrates into acids, gases, or alcohol by "
      "microorganisms like yeast or bacteria, without needing an external oxidizing agent. "
      "In South Indian cooking, this is what makes idli and dosa batter rise: bacteria in the "
      "batter produce both lactic acid (for the tang) and carbon dioxide (for the rise).",
      "https://en.wikipedia.org/wiki/Fermentation_in_food_processing",
      "fermentation is the conversion of carbohydrates to alcohol or organic acids using "
      "microorganisms -- yeasts or bacteria -- without an oxidizing agent being used in the "
      "reaction.",
      ["tech:ferment"]),

    # -- regional_comparison -----------------------------------------------
    _c("regional_comparison", None, "vegetarian",
      "Is thoran the same thing as poriyal?",
      "Essentially yes -- thoran (Kerala) and poriyal (Tamil Nadu) are the same style of "
      "dish: vegetables finely chopped, sauteed briefly with grated coconut, mustard seeds, "
      "and curry leaves. The name just changes with the region.",
      "https://en.wikipedia.org/wiki/Thoran",
      "This common dish is usually eaten with rice and curry...In Tamil Nadu, a similar dish "
      "is known as Poriyal.",
      ["dish:thoran"]),

    _c("regional_comparison", None, "vegetarian",
      "Is ragi mudde called something different in other South Indian states?",
      "Yes -- it's the same finger-millet dish across Karnataka, Andhra Pradesh, and Tamil "
      "Nadu, just under different names: ragi mudde or hittu in Karnataka, ragi sangati in "
      "Andhra's Rayalaseema region, and ragi kali in Tamil Nadu.",
      "https://en.wikipedia.org/wiki/Ragi_mudde",
      "In Rayalaseema Region in Andhra Pradesh it is called Ragi Sangati. In Tamil Nadu, it "
      "is called ragi kali.",
      ["dish:ragi-mudde"]),

    _c("regional_comparison", None, "vegetarian",
      "How is neer dosa different from a regular South Indian dosa?",
      "Regular dosa batter is fermented, which gives it its characteristic tang and texture. "
      "Neer dosa skips fermentation entirely -- it's just soaked rice ground fine with water "
      "and salt, used right away, which gives a softer, more delicate crepe.",
      "https://en.wikipedia.org/wiki/Neer_dosa",
      "Unlike other dosas neer dosa is known for its simple preparation method and lack of "
      "fermentation...Fermentation of the rice is not required to prepare Neer dosa.",
      ["dish:neer-dosa", "dish:dosa"]),

    _c("regional_comparison", None, "non_vegetarian",
      "How does Malabar/Thalassery biryani differ from Hyderabadi biryani?",
      "The biggest difference is the rice: Thalassery biryani uses a short, fragrant rice "
      "called kaima (jeerakasala) instead of the basmati rice used in Hyderabadi biryani. "
      "Thalassery cuisine also carries more Persian, Arab, and European influence from its "
      "history as a trading port, on top of the Mughlai influence found in Hyderabadi food.",
      "https://en.wikipedia.org/wiki/Malabar_biryani",
      "Unlike other biryani dishes, Thalassery biryani is made using kaima (also known as "
      "jeerakasala), an Indian aromatic rice, instead of the usual basmati rice.",
      ["dish:malabar-biryani", "dish:hyderabadi-biryani"]),

    # -- definition: ingredients ------------------------------------------
    _c("definition", None, "vegetarian",
      "What is curry leaf and what does it do in a dish?",
      "Curry leaf comes from the curry tree (Bergera koenigii), native to the Indian "
      "subcontinent. The aromatic leaves are used widely in South Indian cooking, usually "
      "added whole during tempering to infuse the oil with their fragrance.",
      "https://en.wikipedia.org/wiki/Curry_tree",
      "Its leaves are used in many culinary dishes in India, Sri Lanka and Bangladesh.",
      ["ing:curry-leaves"]),

    _c("definition", None, "vegetarian",
      "What is urad dal?",
      "Urad dal (black gram, Vigna mungo) is a bean widely used in South Asian cooking, "
      "especially in South India, where it's a key ingredient in idli and dosa batter. The "
      "whole bean is called black gram; the split, de-husked version is called white lentil.",
      "https://en.wikipedia.org/wiki/Urad_bean",
      "The product sold as black gram is usually the whole bean, whereas the split bean (the "
      "interior being white) is called white lentil...It is very widely used in South Asian "
      "cuisine.",
      ["ing:urad-dal"]),

    _c("definition", None, "vegetarian",
      "What is moringa (drumstick) and how is it used?",
      "Moringa (Moringa oleifera), also called the drumstick tree, is a fast-growing tree "
      "native to northern India. Its long, slender seed pods (drumsticks) and leaves are both "
      "used as vegetables in South Indian cooking, for example in sambar and thoran.",
      "https://en.wikipedia.org/wiki/Moringa_oleifera",
      "Common names include moringa, drumstick tree (from the long, slender, triangular "
      "seed-pods)...It is widely cultivated for its young seed pods and leaves, used as "
      "vegetables.",
      ["ing:drumstick"]),

    _c("definition", None, "vegetarian",
      "What is fenugreek used for?",
      "Fenugreek is a plant whose leaves and seeds have been used as a culinary ingredient "
      "since ancient times, especially common in South Asian cooking. The seeds are often "
      "part of a tempering, and the dried leaves (kasuri methi) are used to flavour curries.",
      "https://en.wikipedia.org/wiki/Fenugreek",
      "Its leaves and seeds have been used as a culinary ingredient since ancient times, and "
      "are common ingredients in dishes from the Indian subcontinent.",
      ["ing:fenugreek"]),

    _c("definition", None, "vegetarian",
      "What gives turmeric its colour and where does it come from?",
      "Turmeric comes from the rhizome (underground stem) of the Curcuma longa plant, native "
      "to the Indian subcontinent and Southeast Asia. The rhizomes are boiled, dried, and "
      "ground into the deep orange-yellow powder used as both a colouring and flavouring "
      "agent in South Indian curries.",
      "https://en.wikipedia.org/wiki/Turmeric",
      "The rhizomes can be used fresh, but they are often boiled in water and dried, after "
      "which they are ground into a deep orange-yellow shelf-stable spice powder commonly "
      "used as a coloring and flavoring agent.",
      ["ing:turmeric"]),
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing_questions = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_questions.add(json.loads(line)["question"])

    new = [c for c in CANDIDATES if c["question"] not in existing_questions]
    with OUT.open("a", encoding="utf-8") as fh:
        for c in new:
            # deterministic (unlike Python's built-in hash(), which is salted
            # per-process) so re-running doesn't reassign ids to the same item
            digest = hashlib.sha256(c["question"].encode()).hexdigest()[:8]
            c["id"] = f"golden:{digest}"
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"appended {len(new)} new candidates ({len(CANDIDATES) - len(new)} already present)")
    print(f"total candidates in {OUT}: {len(existing_questions) + len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
