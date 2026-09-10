"""Aggregate + normalise + classify candidate dish names for the v1 1,000-dish list
(150M dataset plan, Phase 1). No GPU, no LLM.

  harvest_titles()  — pull raw dish-name candidates from corpus text / recipe JSONL
  normalise_name()  — strip 'recipe' / 'how to make' / trailing clauses, collapse
  classify()        — region + dish category (schema enum) + veg/non-veg from keywords
  aggregate()       — group near-duplicates, score by document frequency
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from food_slm.data.recipe_sources import _SI_DISHES, _SI_INGREDIENTS

# --- name normalisation -------------------------------------------------------

_STRIP_TAIL = re.compile(
    r"\s*[–—|:\-]\s*(recipe|authentic.*|traditional.*|easy.*|quick.*|homemade.*|"
    r"south indian.*|kerala.*|tamil.*|andhra.*|step by step.*|with video.*|"
    r"restaurant style.*|how to make.*|no onion no garlic.*|instant pot.*)$", re.I)
_LEAD = re.compile(r"^(how to make|recipe for|homemade|easy|quick|authentic|traditional)\s+", re.I)
_PARENS = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")
_NOISE = re.compile(r"[^\w\s&'-]")
_WS = re.compile(r"\s+")
_RECIPE_WORD = re.compile(r"\brecipe[s]?\b", re.I)


def normalise_name(raw: str) -> str:
    s = _PARENS.sub("", raw).strip()
    s = _LEAD.sub("", s)
    prev = None
    while prev != s:
        prev = s
        s = _STRIP_TAIL.sub("", s).strip()
    s = _RECIPE_WORD.sub("", s)
    s = _NOISE.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    return s


def dedupe_key(name: str) -> str:
    """Order-independent token key so 'pongal ven' == 'ven pongal'. Drops a few
    filler tokens that vary between sources."""
    drop = {"style", "the", "a", "of", "and", "with", "veg", "non", "type", "kind"}
    toks = [t for t in name.split() if t not in drop and len(t) > 1]
    return " ".join(sorted(toks))


# --- classification ---------------------------------------------------------

_REGION_KW = [
    ("tamil_nadu", r"chettinad|kongunad|madurai|tirunelveli|nanjil|thanjavur|tamil|"
                   r"\bcpk\b|kari dosa|paruppu urundai"),
    ("kerala", r"kerala|malabar|nadan|syrian|kottayam|thrissur|travancore|kudampuli|"
               r"puttu|idiyappam|avial|olan|thoran|erissery|kaalan|pulissery|meen |"
               r"\bmoilee\b|molee|sadya|appam"),
    ("karnataka", r"karnataka|udupi|mangalore|mangalorean|coorg|kodava|dharwad|"
                  r"bisi bele|akki roti|ragi mudde|jolada|neer dosa|kori |ghee roast|"
                  r"gojju|saagu|majjige"),
    ("andhra_pradesh", r"andhra|rayalaseema|guntur|nellore|pulusu|pappu|gongura|"
                       r"\bvepudu\b|pachadi|avakaya|pesarattu|\bkodi\b|\bchepala\b"),
    ("telangana", r"telangana|hyderabad|nizami|sarva pinди|\bqubani\b|\bharis\b|"
                  r"\bhaleem\b|bagara|talawa|\bpachi\b"),
    ("puducherry", r"puducherry|pondicherr|creole|tamil-?french"),
]
_REGION_RE = [(r, re.compile(p, re.I)) for r, p in _REGION_KW]

_CATEGORY_KW = [
    ("biryani", r"\bbiryani|biriyani\b"), ("pulao", r"\bpulao|pilaf\b"),
    ("kurma", r"\bkurma|korma|kuruma\b"),
    ("rasam", r"\brasam|charu\b"), ("sambar", r"\bsambar|sambhar\b"),
    ("saaru", r"\bsaaru\b"), ("pulusu", r"\bpulusu\b"),
    ("kuzhambu", r"\bkuzhambu|kulambu|gojju\b"),
    ("kootu", r"\bkootu|koottu\b"), ("poriyal", r"\bporiyal|palya|vepudu|stir.?fry|thoran\b"),
    ("masiyal", r"\bmasiyal|masayal\b"), ("thogayal", r"\bthogayal|thuvaiyal\b"),
    ("pachadi", r"\bpachadi|raita\b"), ("chutney", r"\bchutney|chammanthi\b"),
    ("podi", r"\bpodi|milagai podi|gunpowder\b"),
    ("avial", r"\bavial|aviyal\b"), ("olan", r"\bolan\b"),
    ("appalam_vadagam", r"\bappalam|vadagam|vathal|sandige|fryums\b"),
    ("pickle", r"\bpickle|oorugai|avakaya|urugai\b"),
    ("sweet", r"\bhalwa|halva|payasam|kheer|pradhaman|laddu|ladoo|mysore pak|"
              r"burfi|barfi|jangiri|jalebi|adhirasam|athirasam|obbattu|holige|poli|"
              r"bobbatlu|ariselu|kesari|pootharekulu|unni ?appam|sweet\b"),
    ("beverage", r"\bcoffee|tea|kashayam|panakam|neer mor|buttermilk|jigarthanda|"
                 r"rose milk|badam milk|juice|smoothie\b"),
    ("bread", r"\bparotta|parota|paratha|roti|rotti|chapathi|poori|puri|naan\b"),
    ("tiffin", r"\bdosa|dosai|idli|idly|vada|vadai|pongal|upma|uttapam|uthappam|"
               r"paniyaram|adai|appam|puttu|idiyappam|pesarattu|kuzhi paniyaram|"
               r"sevai|semiya|bonda|bajji|pakoda|pakora\b"),
    ("rice_dish", r"\b(lemon|curd|tamarind|coconut|tomato|puliyodarai|puliyogare|"
                  r"bisi bele|vangi|ghee|jeera|sesame|ellu|milagu|kalyana) (rice|bath|sadam)\b"),
    ("seafood_dish", r"\bfish|meen|prawn|shrimp|crab|nandu|squid|mussel|karimeen|"
                     r"chepala|royyala\b"),
    ("egg_dish", r"\begg|muttai|mutta|anda|kodi guddu\b"),
    ("non_veg_curry", r"\bchicken|mutton|lamb|goat|beef|pork|kozhi|kodi|aatu|"
                      r"attu|erachi|mamsam|keema\b"),
    ("stew", r"\bstew|ishtu\b"),
]
_CATEGORY_RE = [(c, re.compile(p, re.I)) for c, p in _CATEGORY_KW]

_NONVEG_RE = re.compile(
    r"\b(chicken|mutton|lamb|goat|beef|pork|fish|meen|prawn|shrimp|crab|squid|egg|"
    r"muttai|mutta|anda|kozhi|kodi|aatu|attu|erachi|mamsam|keema|karimeen|"
    r"chepala|royyala|nandu|liver|brain|trotters)\b", re.I)
_EGG_RE = re.compile(r"\b(egg|muttai|mutta|anda|kodi guddu)\b", re.I)
_SEAFOOD_RE = re.compile(r"\b(fish|meen|prawn|shrimp|crab|squid|mussel|karimeen|"
                         r"chepala|royyala|nandu|anchov|sardine|mackerel)\b", re.I)


def classify(name: str, hint_regions: list[str] | None = None) -> dict:
    lo = name.lower()
    region = next((r for r, rx in _REGION_RE if rx.search(lo)), None)
    if region is None and hint_regions:
        region = next((r for r in hint_regions if r != "pan_south"), None)
    category = next((c for c, rx in _CATEGORY_RE if rx.search(lo)), "other")
    non_veg = bool(_NONVEG_RE.search(lo))
    diet = {
        "vegetarian": not non_veg, "vegan": False,
        "contains_egg": bool(_EGG_RE.search(lo)),
        "contains_seafood": bool(_SEAFOOD_RE.search(lo)),
        "contains_poultry": bool(re.search(r"\b(chicken|kozhi|kodi)\b", lo)),
        "contains_red_meat": bool(re.search(r"\b(mutton|lamb|goat|beef|pork|aatu|attu)\b", lo)),
    }
    return {"region": region or "pan_south", "category": category, "diet": diet,
            "non_veg": non_veg}


# --- harvesting -------------------------------------------------------------

def _titles_from_jsonl(path: Path) -> Iterator[tuple[str, list[str]]]:
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        t = row.get("title") or row.get("canonical_name") or ""
        if t:
            yield t, row.get("region_tags", []) or []


_NOT_A_DISH = re.compile(
    r"^(ingredients?|directions?|method|instructions?|notes?|tips?|steps?|serves?|"
    r"serving suggestions?|variations?|nutrition.*|prep time.*|cook time.*|"
    r"reply|leave a (comment|reply)|name|email.*|website|subscribe.*|newsletter|"
    r"thank ?(you|u).*|thanks.*|thanx.*|hi\b.*|hello\b.*|hey\b.*|dear\b.*|"
    r"nice|great|awesome|yummy|delicious|wow|super|perfect|amazing|good|"
    r"yes\b.*|no\b.*|ok\b.*|sure\b.*|welcome.*|enjoy.*|regards.*|"
    r"related (posts?|recipes?).*|you (may|might) (also )?like.*|"
    r"about (me|us).*|contact.*|privacy.*|home\b.*|search.*|menu\b.*|categor.*|"
    r"\d+ (comments?|replies).*|print.*|share.*|pin.*|save.*)$", re.I)
_LOOKS_ITEMY = re.compile(r"^[\-\*•]|^\d+[.)]\s")


def _titles_from_corpus_txt(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Each blank-line-separated doc's first line is the title (blog / recipenlg format).
    RecipeNLG fragments each recipe into Title / Ingredients: / Directions: blocks —
    the section-header + list-item filter drops those."""
    buf: list[str] = []
    for line in path.open(encoding="utf-8"):
        if line.startswith("#"):
            continue
        if line.strip() == "":
            if buf:
                t = buf[0].strip()
                if t and not _NOT_A_DISH.match(t) and not _LOOKS_ITEMY.match(t):
                    yield t, []
                buf = []
        else:
            buf.append(line)
    if buf and buf[0].strip() and not _NOT_A_DISH.match(buf[0].strip()):
        yield buf[0].strip(), []


def harvest(sources: list[Path]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for p in sources:
        if not p.exists():
            continue
        it = _titles_from_jsonl(p) if p.suffix == ".jsonl" else _titles_from_corpus_txt(p)
        out.extend(it)
    return out


_NON_SI = re.compile(
    r"\b(thai|goan|mexican|italian|chinese|french|greek|korean|japanese|"
    r"punjabi|gujarati|bengali|rajasthani|maharashtrian|kashmiri|sindhi|"
    r"continental|western|american|pudding|casserole|pasta|pizza|burger|"
    r"sandwich|salad bowl|smoothie bowl|quesadilla|taco|burrito)\b", re.I)


def _si_signal(name: str, regs: list[str]) -> bool:
    lo = name.lower()
    if _NON_SI.search(lo):
        return False
    if regs and any(r in {"tamil_nadu", "kerala", "karnataka", "andhra_pradesh",
                          "telangana", "puducherry"} for r in regs):
        return True
    if _SI_DISHES.search(lo):
        return True
    return sum(1 for s in _SI_INGREDIENTS if s in lo) >= 1 and any(
        w in lo for w in ("curry", "kuzhambu", "kulambu", "rasam", "sambar", "poriyal",
                          "thoran", "kootu", "pachadi", "chutney", "podi", "rice", "dosa",
                          "idli", "vada", "pongal", "upma", "biryani", "sadam", "fry",
                          "roast", "masala", "gojju", "pulusu", "pappu", "vepudu"))


def aggregate(raw: list[tuple[str, list[str]]], *, min_len: int = 3,
              require_si: bool = True) -> list[dict]:
    """Group by dedupe_key, score by frequency, classify. Returns ranked list."""
    groups: dict[str, dict] = {}
    for title, regs in raw:
        if _NOT_A_DISH.match(title.strip()):
            continue
        name = normalise_name(title)
        if len(name) < min_len or not (2 <= len(name.split()) <= 6):
            continue
        if _NOT_A_DISH.match(name):
            continue
        if require_si and not _si_signal(name, regs):
            continue
        k = dedupe_key(name)
        if not k:
            continue
        g = groups.setdefault(k, {"names": Counter(), "regions": Counter(), "count": 0})
        g["names"][name] += 1
        g["count"] += 1
        for r in regs:
            g["regions"][r] += 1

    result = []
    for k, g in groups.items():
        canonical = g["names"].most_common(1)[0][0]
        hint = [r for r, _ in g["regions"].most_common()]
        cls = classify(canonical, hint)
        result.append({
            "key": k, "canonical_name": canonical.title(),
            "aliases": sorted({n.title() for n in g["names"] if n != canonical}),
            "prominence": g["count"],
            "region": cls["region"], "category": cls["category"],
            "diet": cls["diet"], "non_veg": cls["non_veg"],
        })
    result.sort(key=lambda d: -d["prominence"])
    return result
