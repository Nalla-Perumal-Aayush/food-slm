"""YouTube transcript fetch + clean for the 150M domain corpus (Level 1, method B7).

Spoken cooking narration carries technique detail, quantities-by-feel, and
"what went wrong" asides that written recipes omit. We take the *caption text*
only — never video or audio. Regional-language captions are pulled through
YouTube's own English translation so the corpus stays English I/O.

  list_video_ids(channel_url, limit)     -> [(video_id, title), ...]   (yt-dlp, flat)
  fetch_transcript(video_id, ...)        -> TranscriptDoc | None
  clean_caption_text(snippets)           -> str   (de-dup auto-caption echo, strip [Music])

No GPU. Network + rate-limited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_BRACKET = re.compile(r"\[(music|applause|laughter|आवाज|संगीत|foreign)\]", re.I)
_SPEAKER = re.compile(r"(^\s*[-]\s+|\s*>>+\s*)", re.M)
_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class TranscriptDoc:
    video_id: str
    title: str
    channel: str
    source_lang: str
    translated: bool
    is_generated: bool
    text: str
    needs_translation: bool = False   # regional ASR track; queue for local MT
    n_chars: int = 0
    url: str = ""

    def __post_init__(self) -> None:
        self.n_chars = len(self.text)
        self.url = f"https://www.youtube.com/watch?v={self.video_id}"


def clean_caption_text(snippets: list[dict]) -> str:
    """snippets: [{'text', 'start', 'duration'}, ...]. Auto-captions repeat the
    tail of the previous line as the head of the next — collapse that, drop
    bracketed sound cues, re-flow into sentences."""
    parts: list[str] = []
    prev_tail = ""
    for s in snippets:
        t = _BRACKET.sub(" ", s.get("text", "") or "")
        t = t.replace("\n", " ").strip()
        if not t:
            continue
        low = t.lower()
        # drop leading overlap with the previous snippet's tail
        if prev_tail and low.startswith(prev_tail):
            t = t[len(prev_tail):].lstrip(" ,.-")
        if t:
            parts.append(t)
            words = t.lower().split()
            prev_tail = " ".join(words[-6:]) if len(words) >= 3 else ""
    text = _WS.sub(" ", " ".join(parts)).strip()
    text = _WS.sub(" ", _SPEAKER.sub(" ", text)).strip()
    # collapse immediate word/phrase doubling ("the the", "add water add water")
    for _ in range(2):
        text = re.sub(r"\b(\w+(?:\s+\w+){0,6})\s+\1\b", r"\1", text, flags=re.I)
    text = _SENT_END.sub("\n", text)
    return text.strip()


def list_video_ids(channel_url: str, limit: int = 400) -> list[tuple[str, str]]:
    import yt_dlp

    opts = {
        "quiet": True, "skip_download": True, "extract_flat": "in_playlist",
        "playlistend": limit, "ignoreerrors": True,
        "extractor_args": {"youtubetab": {"approximate_date": ["timestamp"]}},
    }
    base = channel_url.rstrip("/")
    if "list=" in base or base.endswith(("/videos", "/streams", "/shorts")):
        candidates = [base]
    else:
        candidates = [base + "/videos", base + "/streams"]

    entries = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in candidates:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception:  # noqa: BLE001
                continue
            raw = (info or {}).get("entries") or []
            flat = []
            for e in raw:
                if e and e.get("entries"):        # nested tab
                    flat.extend(e["entries"])
                elif e:
                    flat.append(e)
            if flat:
                entries = flat
                break

    out: list[tuple[str, str]] = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if vid and len(vid) == 11:
            out.append((vid, (e.get("title") or "").strip()))
    return out[:limit]


def fetch_transcript(video_id: str, title: str, channel: str, *,
                     prefer_langs: tuple[str, ...] = ("en", "en-US", "en-GB"),
                     api=None) -> TranscriptDoc | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import CouldNotRetrieveTranscript

    api = api or YouTubeTranscriptApi()
    try:
        tl = api.list(video_id)
    except CouldNotRetrieveTranscript:
        return None
    except Exception:  # noqa: BLE001 - transient network / parse
        return None

    tracks = list(tl)
    chosen = None
    translated = False
    needs_tr = False
    # 1) a manually-created English track
    for tr in tracks:
        if not tr.is_generated and tr.language_code in prefer_langs:
            chosen = tr
            break
    # 2) an auto-generated English track
    if chosen is None:
        for tr in tracks:
            if tr.language_code in prefer_langs:
                chosen = tr
                break
    # 3) a non-English track that YouTube can still translate server-side
    #    (manually-created subs only; ASR tracks are no longer translatable)
    if chosen is None:
        for tr in tracks:
            if tr.is_translatable:
                try:
                    chosen = tr.translate("en")
                    translated = True
                    break
                except Exception:  # noqa: BLE001
                    continue
    # 4) fall back to the raw regional track and flag it for local MT
    if chosen is None and tracks:
        chosen = tracks[0]
        needs_tr = True
    if chosen is None:
        return None

    try:
        fetched = chosen.fetch()
    except Exception:  # noqa: BLE001
        return None
    raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    text = clean_caption_text(raw)
    if len(text) < 300:
        return None
    src_lang = getattr(chosen, "language_code", "?")
    return TranscriptDoc(
        video_id=video_id, title=title, channel=channel,
        source_lang=src_lang,
        translated=translated,
        is_generated=bool(getattr(chosen, "is_generated", True)),
        text=text, needs_translation=needs_tr,
    )


# --- title relevance --------------------------------------------------------

_FOOD_TITLE = re.compile(
    r"\b(recipe|curry|rice|dal|dosa|idli|idly|vada|sambar|sambhar|rasam|chutney|"
    r"masala|gravy|fry|roast|kuzhambu|kulambu|poriyal|thoran|palya|pachadi|pickle|"
    r"biryani|biriyani|pulao|payasam|kheer|halwa|sweet|snack|breakfast|lunch|dinner|"
    r"tiffin|podi|kootu|pulusu|pappu|vepudu|kura|gojju|saaru|meen|prawn|fish|chicken|"
    r"mutton|egg|paneer|veg|banana|coconut|jaggery|ghee|tamarind|millet|ragi)\b", re.I)


def title_is_food(title: str, extra: re.Pattern | None = None) -> bool:
    if not title:
        return False
    if _FOOD_TITLE.search(title):
        return True
    return bool(extra and extra.search(title))


@dataclass
class ChannelResult:
    name: str
    handle: str
    region: str
    videos_seen: int = 0
    kept: int = 0
    est_tokens: int = 0
    docs: list[TranscriptDoc] = field(default_factory=list)
