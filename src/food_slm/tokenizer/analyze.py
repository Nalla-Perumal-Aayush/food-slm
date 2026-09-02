"""Tokenizer analysis report (baseline §6 selection criteria).

Metrics (``configs/tokenizer.yaml`` → ``analysis_report``):
  token_fertility            mean tokens per whitespace word
  compression_ratio          bytes per token (higher = better compression)
  vocab_utilization          fraction of vocab ids seen in the sample
  culinary_term_fragmentation mean tokens per canonical English dish/ingredient/technique name
  number_unit_efficiency     mean tokens for "<n> <unit>" quantity strings
  native_name_fragmentation  mean tokens per canonical native-script name, per script
  native_script_vocab_share  fraction of vocab tokens that decode to Dravidian script
  sequence_length_efficiency mean words packed into a `ctx`-token window
"""
from __future__ import annotations

from statistics import mean

from tokenizers import Tokenizer

# Dravidian Unicode blocks
_SCRIPTS = {
    "tamil": range(0x0B80, 0x0C00),
    "telugu": range(0x0C00, 0x0C80),
    "kannada": range(0x0C80, 0x0D00),
    "malayalam": range(0x0D00, 0x0D80),
}


def _unicode_to_bytes() -> dict[str, int]:
    """GPT-2 / byte-level reverse map: the printable char a byte was mapped to -> that byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs, strict=False)}


_U2B = _unicode_to_bytes()


def _token_text(tok: Tokenizer, tid: int) -> str:
    raw = tok.id_to_token(tid) or ""
    try:
        return bytes(_U2B[ch] for ch in raw if ch in _U2B).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return ""


def _script_of(s: str) -> str | None:
    for ch in s:
        cp = ord(ch)
        for name, rng in _SCRIPTS.items():
            if cp in rng:
                return name
    return None


_QTY_STRINGS = [
    "2 cups", "1.5 tsp", "200 g", "1/4 cup", "3 tbsp", "500 ml", "1 kg", "10 minutes",
    "180 C", "1 tsp", "2 sprigs", "a pinch",
]


def analyze_tokenizer(
    tok: Tokenizer,
    sample_texts: list[str],
    *,
    dish_names: list[str] | None = None,
    native_names: list[str] | None = None,
    ctx: int = 1024,
) -> dict:
    vocab = tok.get_vocab_size()
    enc = tok.encode_batch(sample_texts) if sample_texts else []

    n_tokens = sum(len(e.ids) for e in enc)
    n_bytes = sum(len(t.encode("utf-8")) for t in sample_texts)
    n_words = sum(len(t.split()) for t in sample_texts)
    seen = {i for e in enc for i in e.ids}

    report: dict = {
        "vocab_size": vocab,
        "sample_docs": len(sample_texts),
        "sample_tokens": n_tokens,
        "token_fertility": round(n_tokens / n_words, 4) if n_words else None,
        "compression_ratio": round(n_bytes / n_tokens, 4) if n_tokens else None,
        "vocab_utilization": round(len(seen) / vocab, 4) if vocab else None,
    }

    if dish_names:
        counts = [len(tok.encode(n).ids) for n in dish_names]
        report["culinary_term_fragmentation"] = {
            "mean": round(mean(counts), 3), "max": max(counts), "n": len(counts),
        }

    qty_counts = [len(tok.encode(q).ids) for q in _QTY_STRINGS]
    report["number_unit_efficiency"] = {
        "mean_tokens": round(mean(qty_counts), 3), "examples": dict(zip(_QTY_STRINGS, qty_counts, strict=False)),
    }

    if native_names:
        by_script: dict[str, list[int]] = {}
        for n in native_names:
            sc = _script_of(n) or "other"
            by_script.setdefault(sc, []).append(len(tok.encode(n).ids))
        report["native_name_fragmentation"] = {
            sc: {"mean": round(mean(v), 2), "max": max(v), "n": len(v)}
            for sc, v in sorted(by_script.items())
        }

    native_vocab = 0
    for i in range(vocab):
        if _script_of(_token_text(tok, i)):
            native_vocab += 1
    report["native_script_vocab_share"] = round(native_vocab / vocab, 4) if vocab else None
    report["native_script_vocab_count"] = native_vocab

    if enc and n_tokens:
        avg_tok_per_doc = n_tokens / len(enc)
        avg_words_per_doc = n_words / len(enc)
        report["sequence_length_efficiency"] = {
            "ctx": ctx,
            "mean_words_per_ctx": round(ctx / avg_tok_per_doc * avg_words_per_doc, 1),
        }

    return report
