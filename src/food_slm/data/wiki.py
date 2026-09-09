"""MediaWiki API helpers: category expansion + batched plain-text extracts.

Wikipedia prose is CC-BY-SA 4.0. Used by scripts/fetch_domain_wiki.py (title list) and
scripts/fetch_wikipedia_food.py (category walk). No GPU, no LLM.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "food-slm-research/0.1 (open-source South Indian cuisine SLM; non-commercial research)"


def cache_path(cache_dir: Path, key: str, *, kind: str = "intro") -> Path:
    d = cache_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(key.encode()).hexdigest() + ".txt")  # noqa: S324


_LAST_CALL = [0.0]
MIN_INTERVAL = 3.0  # Wikimedia anon API rate limit is aggressive — go slow


def _get(client, params: dict, *, max_wait: float = 120.0) -> dict:
    for attempt in range(8):
        gap = MIN_INTERVAL - (time.time() - _LAST_CALL[0])
        if gap > 0:
            time.sleep(gap)
        r = client.get(API, params=params)
        _LAST_CALL[0] = time.time()
        if r.status_code in (429, 503):
            wait = min(float(r.headers.get("retry-after", 2 ** attempt)), max_wait)
            time.sleep(wait + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(
        f"MediaWiki API {params.get('list') or params.get('prop')}: rate-limited out")


def walk_categories(client, seeds: Iterable[str], *, max_depth: int = 2,
                    per_cat_limit: int = 500, exclude_substrings: Iterable[str] = ()) -> set[str]:
    """BFS over ``Category:`` members. Returns the set of article (ns=0) titles."""
    excl = tuple(s.lower() for s in exclude_substrings)
    articles: set[str] = set()
    seen_cats: set[str] = set()
    frontier = [(c if c.startswith("Category:") else f"Category:{c}", 0) for c in seeds]
    while frontier:
        cat, depth = frontier.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont: dict = {}
        got = 0
        while True:
            data = _get(client, {
                "action": "query", "list": "categorymembers", "cmtitle": cat,
                "cmlimit": "500", "cmtype": "page|subcat", "format": "json", **cont,
            })
            for m in data.get("query", {}).get("categorymembers", []):
                title = m["title"]
                if title.startswith("Category:"):
                    if depth < max_depth and not any(x in title.lower() for x in excl):
                        frontier.append((title, depth + 1))
                elif m.get("ns") == 0 and not any(x in title.lower() for x in excl):
                    articles.add(title)
            got += len(data.get("query", {}).get("categorymembers", []))
            cont = data.get("continue", {})
            if not cont or got >= per_cat_limit:
                break
            time.sleep(0.3)
        time.sleep(0.3)
    return articles


def fetch_extracts(client, titles: list[str], cache_dir: Path, *, intro_only: bool = True,
                   sleep: float = 0.5) -> dict[str, str]:
    """Plain-text extracts, cached per title. Intros batch 20/request; full text is 1/request."""
    kind = "intro" if intro_only else "full"
    out: dict[str, str] = {}
    todo: list[str] = []
    for t in titles:
        cp = cache_path(cache_dir, t, kind=kind)
        if cp.exists():
            out[t] = cp.read_text(encoding="utf-8")
        else:
            todo.append(t)

    base = {"action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1",
            "format": "json"}
    if intro_only:
        base |= {"exintro": "1", "exlimit": "max"}
    step = 20 if intro_only else 1
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        data = _get(client, {**base, "titles": "|".join(chunk)})
        q = data.get("query", {})
        remap = {n["from"]: n["to"] for n in (*q.get("normalized", []), *q.get("redirects", []))}
        for p in q.get("pages", {}).values():
            resolved = p.get("title", "")
            src = next((a for a, b in remap.items() if b == resolved and a in chunk), resolved)
            key = src if src in chunk else resolved
            text = p.get("extract", "")
            cache_path(cache_dir, key, kind=kind).write_text(text, encoding="utf-8")
            out[key] = text
        time.sleep(sleep)
    return out
