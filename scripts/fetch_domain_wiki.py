"""Fetch a permissive South-Indian-cuisine prose slice for the 40M domain corpus
(plan_40m §C3).

Wikipedia article *extracts* (plain text) via the MediaWiki API. All Wikipedia prose is
CC-BY-SA 4.0 — the same licence as the wikitext base — so it is corpus-eligible with
attribution/share-alike tracked. Facts still come only from the canonical store; this is
a *language* slice (register, vocabulary, dish/technique names in running text).

    python scripts/fetch_domain_wiki.py --out data/interim/domain_40m

Filters: min length, English (fasttext), exact-dedup. Skips titles already covered by a
forbidden entity (held-out eval probes must not leak into training).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
API = "https://en.wikipedia.org/w/api.php"
# Wikimedia API etiquette: descriptive UA + low request rate + backoff on 429.
UA = "food-slm-research/0.1 (open-source South Indian cuisine SLM; non-commercial research)"

# South Indian cuisine — 6 regions, veg + non-veg, techniques, staples, context.
SEED_TITLES = [
    "South Indian cuisine", "Tamil cuisine", "Kerala cuisine", "Karnataka cuisine",
    "Andhra cuisine", "Telangana cuisine", "Udupi cuisine", "Chettinad cuisine",
    "Mangalorean cuisine", "Cuisine of Coorg", "Hyderabadi cuisine", "Malabar cuisine",
    "Idli", "Dosa (food)", "Vada (food)", "Uttapam", "Pongal (dish)", "Ven pongal",
    "Upma", "Pesarattu", "Appam", "Idiyappam", "Puttu", "Neer dosa", "Set dosa",
    "Rava dosa", "Masala dosa", "Paniyaram", "Kuzhi paniyaram", "Adai", "Rasam",
    "Sambar (dish)", "Kootu", "Poriyal", "Aviyal", "Thoran", "Olan", "Erissery",
    "Kaalan", "Pulissery", "Mor kuzhambu", "Vatha kuzhambu", "Kara kuzhambu",
    "Puli kuzhambu", "Bisi Bele Bath", "Vangi Bath", "Puliyogare", "Curd rice",
    "Lemon rice", "Tamarind rice", "Coconut rice", "Tomato rice", "Bagara rice",
    "Ghee rice", "Hyderabadi biryani", "Ambur biryani", "Dindigul biryani",
    "Thalassery biryani", "Donne biryani", "Chicken 65", "Chicken Chettinad",
    "Andhra chicken curry", "Kerala fish curry", "Meen moilee", "Fish moolie",
    "Prawn curry", "Nandu masala", "Kongunadu cuisine", "Mutton curry",
    "Nihari", "Haleem", "Keema", "Kothu parotta", "Parotta", "Malabar parotta",
    "Appam with mutton stew", "Ishtu", "Payasam", "Mysore pak", "Obbattu",
    "Adhirasam", "Kozhukattai", "Modak", "Unni appam", "Ariselu", "Bobbatlu",
    "Pootharekulu", "Dharwad pedha", "Filter coffee", "Sukku coffee",
    "Neer mor", "Panakam", "Jigarthanda", "Rose milk",
    "Sadya", "Onam", "Banana leaf rice", "Meals (South Indian)",
    "Chutney", "Coconut chutney", "Tomato chutney", "Gongura", "Pachadi",
    "Thogayal", "Podi (spice)", "Milagai podi", "Paruppu podi", "Idli podi",
    "Sambar powder", "Rasam powder", "Garam masala", "Curry tree", "Curry leaf",
    "Asafoetida", "Tamarind", "Kokum", "Garcinia gummi-gutta", "Mustard seed",
    "Urad dal", "Toor dal", "Moong dal", "Chana dal", "Coconut", "Coconut milk",
    "Jaggery", "Ghee", "Coconut oil", "Sesame oil", "Fermentation in food processing",
    "Tempering (cooking)", "Tadka", "Idli rice", "Parboiled rice", "Ponni rice",
    "Sona Masuri", "Semolina", "Rice flour", "Ragi", "Finger millet", "Foxtail millet",
    "Ragi mudde", "Akki rotti", "Jolada rotti", "Thalipeeth",
]


def _cache_path(cache_dir: Path, title: str, *, intro_only: bool = True) -> Path:
    sub = "intro" if intro_only else "full"
    (cache_dir / sub).mkdir(parents=True, exist_ok=True)
    return cache_dir / sub / (hashlib.sha1(title.encode()).hexdigest() + ".txt")  # noqa: S324


def _fetch_batch(client, titles: list[str], cache_dir: Path, *, intro_only: bool) -> dict[str, str]:
    """Fetch up to 20 article extracts in one request. Wikimedia allows exlimit=max only
    with exintro; full-text extracts must be requested one title at a time."""
    out: dict[str, str] = {}
    todo = []
    for t in titles:
        cp = _cache_path(cache_dir, t, intro_only=intro_only)
        if cp.exists():
            out[t] = cp.read_text(encoding="utf-8")
        else:
            todo.append(t)

    params_base = {"action": "query", "prop": "extracts", "explaintext": "1",
                   "redirects": "1", "format": "json"}
    if intro_only:
        params_base |= {"exintro": "1", "exlimit": "max"}
    step = 20 if intro_only else 1
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        params = params_base | {"titles": "|".join(chunk)}
        for attempt in range(5):
            r = client.get(API, params=params)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json().get("query", {})
            norm = {n["from"]: n["to"] for n in data.get("normalized", [])}
            redir = {n["from"]: n["to"] for n in data.get("redirects", [])}
            for p in data.get("pages", {}).values():
                out_title = p.get("title", "")
                text = p.get("extract", "")
                # map resolved title back to a requested one
                src = out_title
                for a, b in {**norm, **redir}.items():
                    if b == out_title and a in chunk:
                        src = a
                key = src if src in chunk else out_title
                _cache_path(cache_dir, key, intro_only=intro_only).write_text(text, encoding="utf-8")
                out[key] = text
            break
        time.sleep(0.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "interim" / "domain_40m")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--full", action="store_true",
                    help="also try full-article extracts (slow, often rate-limited)")
    ap.add_argument("--extra-titles", type=Path, default=None,
                    help="newline-delimited file of extra titles")
    args = ap.parse_args()

    import httpx

    from food_slm.data.forbidden import load_forbidden
    from food_slm.data.langid import ensure_model, is_english

    ensure_model()
    forbidden = load_forbidden(REPO / "data" / "eval")

    titles = list(dict.fromkeys(SEED_TITLES))
    if args.extra_titles and args.extra_titles.exists():
        titles += [t.strip() for t in args.extra_titles.read_text(encoding="utf-8").splitlines()
                   if t.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "_cache"
    cache_dir.mkdir(exist_ok=True)
    corpus = args.out / "corpus.txt"
    seen: set[str] = set()
    kept: list[dict] = []
    skipped: dict[str, list[str]] = {"short": [], "non_english": [], "forbidden": [], "dup": []}
    chars = 0

    with httpx.Client(headers={"User-Agent": UA}, timeout=30.0) as client:
        texts = _fetch_batch(client, titles, cache_dir, intro_only=True)
        if args.full:
            long = [t for t, x in texts.items() if len(x) >= args.min_chars]
            texts |= _fetch_batch(client, long, cache_dir, intro_only=False)

    with corpus.open("w", encoding="utf-8") as fh:
        fh.write("# --- South Indian cuisine (Wikipedia extracts, CC-BY-SA 4.0) ---\n")
        for title in titles:
            text = texts.get(title, "").strip()
            if len(text) < args.min_chars:
                skipped["short"].append(title)
                continue
            hit = sorted(forbidden.scan_text(text))
            if hit:
                skipped["forbidden"].append(f"{title} -> {hit[:3]}")
                continue
            if not is_english(text):
                skipped["non_english"].append(title)
                continue
            key = " ".join(text.split())[:400].lower()
            if key in seen:
                skipped["dup"].append(title)
                continue
            seen.add(key)
            fh.write(text)
            fh.write("\n\n")
            chars += len(text) + 2
            kept.append({"title": title, "chars": len(text)})

    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "en.wikipedia.org (MediaWiki API extracts, intro" + ("+full" if args.full else "") + ")",
        "license": "CC-BY-SA-4.0",
        "purpose": "40M domain language slice (plan_40m §C3) — prose register, not facts",
        "articles_kept": len(kept), "chars": chars, "est_tokens": chars // 4,
        "kept": kept,
        "skipped": {k: v for k, v in skipped.items() if v},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                            + "\n", encoding="utf-8")
    print(f"kept {len(kept)} articles / ~{chars // 4 / 1e3:.0f}k tok est -> {corpus}")
    for k, v in skipped.items():
        if v:
            print(f"  skipped[{k}]: {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
