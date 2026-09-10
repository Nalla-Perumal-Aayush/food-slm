"""Local machine-translation pass over regional YouTube transcripts parked by
scripts/fetch_youtube.py (150M dataset, method B7).

YouTube stopped exposing server-side translation for auto-generated captions, so
Malayalam / Tamil / Telugu / Kannada cooking-channel transcripts are cached raw
with `needs_translation: true`. This script translates them to English with a
local Ollama model (no paid API, fully offline), chunk by chunk, then applies the
same english / forbidden gates as the main crawler and appends the result to
data/interim/domain_150m/youtube/<channel>.txt.

    # best run AFTER a training run frees the GPU (Ollama will otherwise contend)
    python scripts/translate_youtube_cache.py --model qwen2.5:7b
    python scripts/translate_youtube_cache.py --only salu_kitchen --limit 50

Resumable: a translated cache entry gets `mt_done: true` and is skipped next run.
No GPU required for this script itself (Ollama may use CPU).
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
from food_slm.data.llm import OllamaProvider, ProviderSpec  # noqa: E402
from food_slm.data.transcript import title_is_food  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "interim" / "domain_150m" / "youtube" / "_cache"
OUT = REPO / "data" / "interim" / "domain_150m" / "youtube"

_LANG_NAME = {"ml": "Malayalam", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
              "hi": "Hindi", "ml-IN": "Malayalam", "ta-IN": "Tamil"}

_SYS = (
    "You are a translator specialising in South Indian cooking videos. Translate the "
    "given {lang} cooking-narration transcript into natural English. Keep every "
    "quantity, ingredient name, cooking time, temperature and technique exactly. "
    "Keep dish and ingredient names in their common transliterated form (e.g. "
    "'ven pongal', 'kudampuli', 'thoran'). Output ONLY the English translation, no "
    "notes, no preamble."
)


def _chunks(text: str, size: int = 1600) -> list[str]:
    out, buf = [], ""
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        if len(buf) + len(sent) > size and buf:
            out.append(buf.strip())
            buf = ""
        buf += sent + " "
    if buf.strip():
        out.append(buf.strip())
    return out


def _translate(provider, text: str, lang: str, *, retries: int = 2) -> str | None:
    name = _LANG_NAME.get(lang, lang)
    parts: list[str] = []
    for ch in _chunks(text):
        got = None
        for _ in range(retries + 1):
            try:
                got = provider.complete(_SYS.format(lang=name), ch, temperature=0.0)
                if got and got.strip():
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"    chunk failed ({type(exc).__name__}); retrying")
                time.sleep(2)
        if not got:
            return None
        parts.append(got.strip())
    return "\n".join(parts).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--only", nargs="*", default=None, help="channel name filter")
    ap.add_argument("--limit", type=int, default=0, help="max videos this run (0 = all)")
    ap.add_argument("--min-conf", type=float, default=0.55)
    args = ap.parse_args()

    if not CACHE.exists():
        print("no cache dir — run scripts/fetch_youtube.py first")
        return 1

    provider = OllamaProvider(ProviderSpec(
        name="mt", kind="ollama", model=args.model, family="open", local=True,
        max_tokens=2048))
    forbidden = load_forbidden(REPO / "data" / "eval")
    langid.ensure_model()

    pending = []
    for p in sorted(CACHE.glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not m.get("needs_translation") or m.get("mt_done") or not m.get("text"):
            continue
        if args.only and m.get("channel") not in args.only:
            continue
        pending.append((p, m))
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(pending)} transcripts to translate  (model {args.model})")

    dish_re = None
    per_channel: dict[str, list[dict]] = {}
    done = 0
    for p, m in pending:
        vid, lang, channel = m["video_id"], m.get("source_lang", "?"), m.get("channel", "?")
        title = m.get("title", "")
        print(f"  {channel}/{vid}  [{lang}]  {title[:60]}")
        en = _translate(provider, m["text"], lang)
        if not en or len(en) < 300:
            print("    translation empty/short — skipping")
            continue
        if not langid.is_english(en[:4000], min_conf=args.min_conf):
            print("    still not English — skipping")
            continue
        if not title_is_food(title, dish_re) and not title_is_food(en[:1500], dish_re):
            print("    not food-related — skipping")
            m["mt_done"] = True
            p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            continue
        if forbidden.scan_text(f"{title}\n{en[:4000]}"):
            print("    forbidden-entity hit — skipping")
            m["mt_done"] = True
            p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            continue
        m["text_en"] = en
        m["mt_done"] = True
        m["mt_model"] = args.model
        p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        per_channel.setdefault(channel, []).append(m)
        done += 1

    # append translated docs to <channel>.txt (create if missing)
    for channel, docs in per_channel.items():
        fp = OUT / f"{channel}.txt"
        with fp.open("a", encoding="utf-8") as fh:
            if fp.stat().st_size == 0:
                fh.write(f"# --- {channel} — YouTube captions (local MT); weights only ---\n")
            for m in docs:
                fh.write(f"\n## {m.get('title','')} [translated from {m.get('source_lang')}]\n"
                         f"{m['text_en']}\n")
        print(f"  +{len(docs)} -> {fp.name}")

    log = OUT / "mt_manifest.json"
    prev = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {"runs": []}
    prev["runs"].append({
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model, "translated": done,
        "channels": {c: len(v) for c, v in per_channel.items()},
    })
    log.write_text(json.dumps(prev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ntranslated {done} transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
