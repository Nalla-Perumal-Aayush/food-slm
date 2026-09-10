"""Crawl the curated South Indian food blogs for the 150M domain corpus (Level 1, method B6).

    python scripts/fetch_food_blogs.py                 # all sites in configs/domain_sources.yaml
    python scripts/fetch_food_blogs.py --only kannamma_cooks sailus_food

WordPress sites are pulled via /wp-json/wp/v2/posts (fast, clean); others via sitemap +
trafilatura. Every post is forbidden-scanned and deduped. Output:
data/interim/domain_150m/blogs/<name>.txt (each post: title + cleaned body) + a manifest
with per-site provenance (name, url, region, robots-allowed, post count).

Openly published web content, robots.txt respected, rate-limited. Weights only, not
redistributed. No GPU.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.recipe_sources import _SI_DISHES, _SI_INGREDIENTS  # noqa: E402
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UA = "food-slm-research/0.1 (open-source South Indian cuisine SLM; non-commercial research)"

_FOOD_WORDS = re.compile(
    r"\b(recipe|ingredient|cook|curry|rice|dal|dosa|idli|sambar|rasam|chutney|"
    r"masala|spice|tempering|tadka|fry|roast|grind|batter|coconut|tamarind|"
    r"vegetable|paneer|chicken|mutton|fish|prawn|breakfast|lunch|dinner|snack|"
    r"sweet|dessert|kuzhambu|poriyal|thoran|biryani|pickle|podi|payasam)\b", re.I)


def _is_food_post(title: str, body: str) -> bool:
    head = (title + " " + body[:800]).lower()
    return bool(_FOOD_WORDS.search(head) or _SI_DISHES.search(head)
               or sum(1 for s in _SI_INGREDIENTS if s in head) >= 2)


def _consume(st: dict, seen: set, fh, forbidden, title: str, body: str, min_chars: int) -> None:
    if len(body) < min_chars:
        st["short"] += 1
        return
    tl = title.lower().strip()
    if not tl or tl in seen:
        return
    seen.add(tl)
    if not _is_food_post(title, body):
        st["offtopic"] += 1
        return
    if forbidden.scan_text(f"{title} {body[:3000]}"):
        st["forbid"] += 1
        return
    fh.write(f"{title.strip()}\n{body}\n\n")
    st["kept"] += 1
    st["chars"] += len(body)

_TAG = re.compile(r"<[^>]+>")
_BOILER = re.compile(
    r"(jump to recipe|print recipe|pin recipe|you (may|might) also like|"
    r"leave a (comment|reply)|rate this recipe|save|share this|"
    r"subscribe|newsletter|follow me on|as an amazon associate)", re.I)


def _clean_html(raw: str) -> str:
    raw = re.sub(r"<(script|style|figure|figcaption)[^>]*>.*?</\1>", " ", raw,
                 flags=re.S | re.I)
    text = html.unescape(_TAG.sub("\n", raw))
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s and not _BOILER.search(s):
            lines.append(s)
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _robots(client, base: str) -> tuple:
    """(RobotFileParser, raw text). Fetched with our real UA, not urllib's."""
    rp = urllib.robotparser.RobotFileParser()
    txt = ""
    try:
        r = client.get(base.rstrip("/") + "/robots.txt")
        txt = r.text if r.status_code == 200 else ""
    except Exception:  # noqa: BLE001 - no robots.txt -> allow all
        pass
    rp.parse(txt.splitlines())
    return rp, txt


def _wp_posts(client, base: str, max_posts: int, sleep: float):
    # oldest-first: the genuine recipe archive lives in the older posts; many of these
    # blogs' recent posts are sponsored / SEO filler (the food filter drops those anyway).
    page, yielded = 1, 0
    while yielded < max_posts:
        r = client.get(f"{base}/wp-json/wp/v2/posts",
                       params={"per_page": 100, "page": page, "orderby": "date",
                               "order": "asc", "_fields": "title,content,link,date"})
        if r.status_code in (400, 404):
            break
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for row in rows:
            yield row
            yielded += 1
        page += 1
        time.sleep(sleep)


def _sitemap_urls(client, base: str, limit: int, robots_txt: str = "") -> list[str]:
    candidates = re.findall(r"(?im)^sitemap:\s*(\S+)", robots_txt) + [
        base + p for p in ("/wp-sitemap.xml", "/sitemap_index.xml", "/sitemap.xml",
                           "/sitemap-index.xml", "/api/sitemap")]
    seen_maps: set[str] = set()
    urls: list[str] = []
    queue = list(dict.fromkeys(candidates))
    while queue and len(urls) < limit:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        try:
            r = client.get(sm)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
        for loc in locs:
            if loc.lower().endswith(".xml"):
                queue.append(loc)
            elif not loc.rstrip("/").endswith((".jpg", ".png", ".webp", ".pdf")):
                urls.append(loc)
        time.sleep(0.4)
    return list(dict.fromkeys(urls))[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "domain_sources.yaml")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "blogs")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-posts", type=int, default=2500)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    import httpx
    import trafilatura

    cfg = load_yaml(args.config)
    forbidden = load_forbidden(REPO / "data" / "eval")
    args.out.mkdir(parents=True, exist_ok=True)
    sites = [b for b in cfg.get("blogs", [])
             if args.only is None or b["name"] in args.only]

    report = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=45.0, follow_redirects=True) as client:
        for site in sites:
            name, base = site["name"], site["url"].rstrip("/")
            print(f"\n=== {name}  ({base}) ===")
            rp, robots_txt = _robots(client, base)
            wp_ok = rp.can_fetch(UA, base + "/wp-json/wp/v2/posts")
            root_ok = rp.can_fetch(UA, base + "/some-recipe-post/")
            if not root_ok:
                print("  robots.txt disallows page crawling — skipping")
                report.append({"name": name, "url": base, "robots_allowed": False, "posts": 0})
                continue
            want_wp = site.get("type") == "wp" and wp_ok
            print(f"  wp-json allowed: {wp_ok}  (prefer wp: {want_wp})")
            seen: set[str] = set()
            st = {"kept": 0, "forbid": 0, "short": 0, "offtopic": 0, "chars": 0}
            fh = (args.out / f"{name}.txt").open("w", encoding="utf-8")
            fh.write(f"# --- {name} ({base}) — openly published; weights only ---\n")

            if want_wp:
                try:
                    for row in _wp_posts(client, base, args.max_posts, args.sleep):
                        _consume(st, seen, fh, forbidden,
                                 _clean_html(row.get("title", {}).get("rendered", "")),
                                 _clean_html(row.get("content", {}).get("rendered", "")),
                                 args.min_chars)
                except Exception as exc:  # noqa: BLE001
                    print(f"  wp-json failed after {st['kept']} ({type(exc).__name__}); trying sitemap")
            if st["kept"] < 20:                     # wp gave little / nothing -> sitemap
                try:
                    urls = _sitemap_urls(client, base, args.max_posts, robots_txt)
                    print(f"  {len(urls)} urls from sitemap")
                    for u in urls:
                        if not rp.can_fetch(UA, u):
                            continue
                        rr = client.get(u)
                        time.sleep(args.sleep)
                        if rr.status_code != 200:
                            continue
                        body = trafilatura.extract(rr.text, include_comments=False) or ""
                        m = re.search(r"<title>([^<]+)</title>", rr.text)
                        _consume(st, seen, fh, forbidden,
                                 html.unescape(m.group(1)) if m else u, body, args.min_chars)
                except Exception as exc:  # noqa: BLE001
                    print(f"  sitemap failed after {st['kept']} ({type(exc).__name__})")
            kept, forbid, short, offtopic, chars = (
                st["kept"], st["forbid"], st["short"], st["offtopic"], st["chars"])
            fh.close()
            print(f"  kept {kept}  (~{chars // 4:,} tok)  short {short}  "
                  f"offtopic {offtopic}  forbidden {forbid}")
            report.append({"name": name, "url": base, "region": site.get("region"),
                           "robots_allowed": True, "posts": kept, "est_tokens": chars // 4,
                           "offtopic_dropped": offtopic})

    (args.out / "manifest.json").write_text(json.dumps({
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": "openly published web content — weights only, not redistributed",
        "sites": report, "total_est_tokens": sum(s.get("est_tokens", 0) for s in report),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nTOTAL ~{sum(s.get('est_tokens', 0) for s in report):,} tok  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
