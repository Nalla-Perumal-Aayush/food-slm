"""Fetch primary-source pages for golden-QA-set drafting, via this project's own
self-hosted scraper (food_slm.data.ingest.Fetcher: robots-respecting, rate-limited,
cached, trafilatura text extraction) + license classification -- NOT ad-hoc
WebFetch calls, so every fetch is cached, auditable, and reusable across sessions,
consistent with how every other source in this project gets collected.

    python scripts/fetch_golden_sources.py

Writes one JSON record per URL to data/eval/golden_sources/<hash>.json:
{"url", "title", "text", "license_class", "usable_for", "fetched_at"}.
Drafting the actual Q&A candidates from this text happens separately (by hand,
in-conversation, citing the fetched text) -- never AI-generated without a human
verification gate, per evaluation_strategy.md SS5.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.ingest import Fetcher  # noqa: E402
from food_slm.data.licensing import classify_license  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "interim" / "golden_source_fetch_cache"
OUT_DIR = REPO / "data" / "eval" / "golden_sources"

# Curated primary sources for an initial golden-QA batch, chosen to spread across
# the 6 regions and give each of the 8 task types something real to cite. More
# get added over time as the set grows toward the ~500-1000 target.
URLS = [
    "https://en.wikipedia.org/wiki/Sambar_(dish)",
    "https://en.wikipedia.org/wiki/Idli",
    "https://en.wikipedia.org/wiki/Avial",
    "https://en.wikipedia.org/wiki/Pesarattu",
    "https://en.wikipedia.org/wiki/Hyderabadi_biryani",
    "https://en.wikipedia.org/wiki/Dosa",
    "https://en.wikipedia.org/wiki/Rasam_(soup)",
    "https://en.wikipedia.org/wiki/Payasam",
    "https://en.wikipedia.org/wiki/Pongal_(dish)",
    "https://en.wikipedia.org/wiki/Vada_(food)",
    "https://en.wikipedia.org/wiki/Appam",
    "https://en.wikipedia.org/wiki/Puttu",
    "https://en.wikipedia.org/wiki/Bisi_bele_bath",
    "https://en.wikipedia.org/wiki/Gongura",
    "https://en.wikipedia.org/wiki/Chettinad_cuisine",
    "https://en.wikipedia.org/wiki/Udupi_cuisine",
    "https://en.wikipedia.org/wiki/Tamarind",
    "https://en.wikipedia.org/wiki/Asafoetida",
    "https://en.wikipedia.org/wiki/Toor_dal",
    "https://en.wikipedia.org/wiki/Jaggery",
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(CACHE_DIR)
    ok, failed = 0, []
    for url in URLS:
        res = fetcher.fetch(url)
        if not res.ok or not res.text.strip():
            failed.append((url, res.status, res.reason))
            print(f"FAILED {url}: status={res.status} reason={res.reason}")
            continue
        verdict = classify_license(res.html, url)
        out = {
            "url": url,
            "title": url.rsplit("/", 1)[-1].replace("_", " "),
            "text": res.text,
            "license_class": verdict.license_class,
            "usable_for": verdict.usable_for,
            "fetched_at": res.fetched_at,
        }
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        (OUT_DIR / f"{h}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
        ok += 1
        print(f"OK {url} ({verdict.license_class}, {len(res.text)} chars)")

    print(f"\n{ok}/{len(URLS)} fetched successfully")
    if failed:
        print("failed:")
        for url, status, reason in failed:
            print(f"  {url}: status={status} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
