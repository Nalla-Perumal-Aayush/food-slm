"""fastText language identification + Dravidian-script ratio gate (baseline §6 / §8).

``lid.176`` (compressed, ~0.9 MB) lives at ``artifacts/models/lid.176.ftz``. Download it
once with :func:`ensure_model`.

fastText 0.9.x calls ``np.array(..., copy=False)`` which NumPy >= 2 rejects; :func:`_load`
installs a scoped shim so the wheel keeps working on the pinned stack.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "models" / "lid.176.ftz"

# Dravidian Unicode blocks (Tamil, Malayalam, Kannada, Telugu) — the 4-script policy set.
_DRAVIDIAN_RANGES = (
    (0x0B80, 0x0BFF),  # Tamil
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0C80, 0x0CFF),  # Kannada
    (0x0C00, 0x0C7F),  # Telugu
)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download lid.176.ftz if missing. Returns the local path."""
    path = Path(path)
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed fbaipublicfiles URL
    return path


@lru_cache(maxsize=1)
def _load(path: str):
    import numpy as np

    orig = np.array

    def _shim(*a, **k):
        if k.get("copy", True) is False:
            k["copy"] = None
        return orig(*a, **k)

    np.array = _shim
    try:
        import fasttext

        return fasttext.load_model(str(path))
    finally:
        np.array = orig


def detect(text: str, *, path: Path = MODEL_PATH) -> tuple[str, float]:
    """(iso639-1 code, confidence) for the dominant language of *text*."""
    model = _load(str(path))
    import numpy as np

    orig = np.array

    def _shim(*a, **k):
        if k.get("copy", True) is False:
            k["copy"] = None
        return orig(*a, **k)

    np.array = _shim
    try:
        labels, probs = model.predict(text.replace("\n", " ")[:2048], k=1)
    finally:
        np.array = orig
    return labels[0].replace("__label__", ""), float(probs[0])


def is_english(text: str, *, min_conf: float = 0.65, path: Path = MODEL_PATH) -> bool:
    lang, conf = detect(text, path=path)
    return lang == "en" and conf >= min_conf


def dravidian_char_ratio(text: str) -> float:
    """Fraction of letters that are in a Dravidian block (NFC-normalised)."""
    text = unicodedata.normalize("NFC", text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    n = sum(1 for c in letters
            if any(lo <= ord(c) <= hi for lo, hi in _DRAVIDIAN_RANGES))
    return n / len(letters)
