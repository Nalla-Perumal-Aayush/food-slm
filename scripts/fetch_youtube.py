"""Crawl South Indian cooking-channel transcripts for the 150M domain corpus (method B7).

    python scripts/fetch_youtube.py                      # all channels in configs/youtube_channels.yaml
    python scripts/fetch_youtube.py --only salu_kitchen vahchef
    python scripts/fetch_youtube.py --max-videos 150 --sleep 1.5

For each channel: list recent uploads (yt-dlp, flat), keep the food-relevant titles
(keyword match OR a hit against configs/dish_list_v1.yaml), fetch the caption text
(regional languages come through YouTube's own English translation), clean it,
language-gate, forbidden-scan, and append to
data/interim/domain_150m/youtube/<channel>.txt.

Per-video transcripts are cached under _cache/<video_id>.json so re-runs are cheap
and resumable. A manifest records channel / video id / title / url / caption language
/ translated? for every kept video.

Transcripts used for weights only, NOT redistributed. Rate-limited. No GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data import langid  # noqa: E402
from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.transcript import (  # noqa: E402
    TranscriptBlocked,
    fetch_transcript,
    list_video_ids,
    title_is_food,
)
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "interim" / "domain_150m" / "youtube"


def _dish_title_regex() -> re.Pattern | None:
    f = REPO / "configs" / "dish_list_v1.yaml"
    if not f.exists():
        return None
    doc = load_yaml(f)
    toks: set[str] = set()
    for d in doc.get("dishes", []):
        for nm in [d.get("canonical_name", ""), *(d.get("also_known_as") or [])]:
            for w in re.split(r"[\s/()-]+", nm.lower()):
                if len(w) >= 4 and w.isalpha():
                    toks.add(re.escape(w))
    if not toks:
        return None
    return re.compile(r"\b(" + "|".join(sorted(toks)) + r")\b", re.I)


def _cache_get(cache: Path, vid: str):
    p = cache / f"{vid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_put(cache: Path, vid: str, payload: dict) -> None:
    (cache / f"{vid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "youtube_channels.yaml")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--min-conf", type=float, default=0.55,
                    help="min fastText English confidence after translation")
    args = ap.parse_args()

    from youtube_transcript_api import YouTubeTranscriptApi

    cfg = load_yaml(args.config)
    forbidden = load_forbidden(REPO / "data" / "eval")
    langid.ensure_model()
    api = YouTubeTranscriptApi()
    dish_re = _dish_title_regex()

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "_cache"
    cache.mkdir(exist_ok=True)

    channels = [c for c in cfg.get("channels", [])
                if args.only is None or c["name"] in args.only]
    report = []

    for ch in channels:
        name, handle, region = ch["name"], ch["handle"], ch.get("region", "pan_south")
        url = handle if handle.startswith("http") else f"https://www.youtube.com/{handle}"
        print(f"\n=== {name}  ({url})  [{region}] ===")
        try:
            vids = list_video_ids(url, args.max_videos)
        except Exception as exc:  # noqa: BLE001
            print(f"  channel listing failed ({type(exc).__name__}: {exc}) — skipping")
            report.append({"name": name, "handle": handle, "region": region,
                           "listed": 0, "kept": 0, "error": type(exc).__name__})
            continue
        print(f"  {len(vids)} uploads listed")

        st = {"seen": 0, "kept": 0, "chars": 0, "offtopic": 0, "no_caption": 0,
              "non_en": 0, "forbidden": 0, "cached": 0, "queued_mt": 0}
        kept_meta = []
        fh = (args.out / f"{name}.txt").open("w", encoding="utf-8")
        fh.write(f"# --- {name} ({url}) — YouTube captions; weights only, not redistributed ---\n")

        # regional-language channels rarely put English keywords in the title;
        # attempt the transcript anyway and let the english / dish / forbidden
        # gates below decide. English channels are filtered on the title (cheap).
        force = ch.get("lang", "en") != "en"
        blocks = 0

        for vid, title in vids:
            st["seen"] += 1
            if not force and not title_is_food(title, dish_re):
                st["offtopic"] += 1
                continue

            cached = _cache_get(cache, vid)
            if cached is not None:
                doc_text = cached.get("text", "")
                meta = cached
                st["cached"] += 1
            else:
                doc = None
                backoff = 30.0
                for attempt in range(4):
                    try:
                        doc = fetch_transcript(vid, title, name, api=api)
                        blocks = 0
                        break
                    except TranscriptBlocked as exc:
                        blocks += 1
                        print(f"    rate-limited ({exc}); sleeping {backoff:.0f}s "
                              f"[block {blocks}]")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 300)
                else:
                    print("    still blocked after retries — moving to next channel")
                    break
                time.sleep(args.sleep)
                if doc is None:
                    _cache_put(cache, vid, {"video_id": vid, "text": "", "empty": True})
                    st["no_caption"] += 1
                    continue
                meta = {"video_id": vid, "title": title, "source_lang": doc.source_lang,
                        "translated": doc.translated, "is_generated": doc.is_generated,
                        "needs_translation": doc.needs_translation,
                        "text": doc.text, "url": doc.url, "channel": name,
                        "region": region}
                _cache_put(cache, vid, meta)
                doc_text = doc.text

            if not doc_text or meta.get("empty"):
                st["no_caption"] += 1
                continue
            if meta.get("needs_translation"):
                # regional ASR track parked in the cache for translate_youtube_cache.py
                st["queued_mt"] += 1
                continue
            if not langid.is_english(doc_text[:4000], min_conf=args.min_conf):
                st["non_en"] += 1
                continue
            if force and not title_is_food(title, dish_re) \
                    and not title_is_food(doc_text[:1500], dish_re):
                st["offtopic"] += 1
                continue
            if forbidden.scan_text(f"{title}\n{doc_text[:4000]}"):
                st["forbidden"] += 1
                continue

            fh.write(f"\n## {title}\n{doc_text}\n")
            st["kept"] += 1
            st["chars"] += len(doc_text)
            kept_meta.append({k: meta.get(k) for k in
                              ("video_id", "title", "url", "source_lang", "translated")})
        fh.close()
        print(f"  kept {st['kept']}  (~{st['chars'] // 4:,} tok)  offtopic {st['offtopic']}  "
              f"no-caption {st['no_caption']}  non-en {st['non_en']}  forbidden {st['forbidden']}  "
              f"queued-for-MT {st['queued_mt']}  cached {st['cached']}")
        report.append({"name": name, "handle": handle, "region": region,
                       "tier": ch.get("tier"), "listed": len(vids), "kept": st["kept"],
                       "est_tokens": st["chars"] // 4, "queued_mt": st["queued_mt"],
                       "videos": kept_meta})

    (args.out / "manifest.json").write_text(json.dumps({
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": "YouTube caption text — weights only, not redistributed; provenance tracked",
        "channels": report,
        "total_est_tokens": sum(c.get("est_tokens", 0) for c in report),
        "total_videos": sum(c.get("kept", 0) for c in report),
        "total_queued_for_mt": sum(c.get("queued_mt", 0) for c in report),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nTOTAL ~{sum(c.get('est_tokens', 0) for c in report):,} tok  "
          f"({sum(c.get('kept', 0) for c in report)} videos, "
          f"{sum(c.get('queued_mt', 0) for c in report)} queued for local MT)  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
