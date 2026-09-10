"""Download public-domain cookery books from the Internet Archive (150M dataset, Level 1).

    python scripts/fetch_archive_books.py

Reads configs/domain_sources.yaml -> archive_pd_books, pulls each item's OCR text
(<id>_djvu.txt, or the largest text file in the item), light-cleans it (de-hyphenate,
drop page furniture), forbidden-scans, and writes
data/interim/domain_150m/books/<id>.txt + a manifest.

True public domain — cleanly reusable. No GPU.
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

from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UA = "food-slm-research/0.1 (open-source South Indian cuisine SLM; non-commercial research)"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)                      # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if re.fullmatch(r"[\divxlcIVXLC.\s\-—]{0,8}", s):      # bare page numbers / rules
            continue
        if re.fullmatch(r"(Digitized by Google|Google|INTERNET ARCHIVE).*", s, re.I):
            continue
        lines.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _fetch_text(client, item_id: str) -> tuple[str, str]:
    """Return (text, filename). Tries <id>_djvu.txt, then the largest .txt in metadata."""
    djvu = f"https://archive.org/download/{item_id}/{item_id}_djvu.txt"
    r = client.get(djvu)
    if r.status_code == 200 and len(r.text) > 2000:
        return r.text, f"{item_id}_djvu.txt"
    meta = client.get(f"https://archive.org/metadata/{item_id}").json()
    txts = [f for f in meta.get("files", []) if f.get("name", "").lower().endswith(".txt")]
    if not txts:
        return "", ""
    best = max(txts, key=lambda f: int(f.get("size", 0)))
    r = client.get(f"https://archive.org/download/{item_id}/{best['name']}")
    r.raise_for_status()
    return r.text, best["name"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "domain_sources.yaml")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "interim" / "domain_150m" / "books")
    args = ap.parse_args()

    import httpx

    cfg = load_yaml(args.config)
    forbidden = load_forbidden(REPO / "data" / "eval")
    args.out.mkdir(parents=True, exist_ok=True)
    got = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=60.0, follow_redirects=True) as client:
        for book in cfg.get("archive_pd_books", []):
            bid = book["id"]
            print(f"  {bid}  ({book['title'][:55]})")
            try:
                raw, fname = _fetch_text(client, bid)
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                continue
            if not raw:
                print("    no text file found")
                continue
            text = _clean(raw)
            hits = sorted(forbidden.scan_text(text))
            if hits:
                print(f"    forbidden-entity hits {hits[:3]} — trimming offending lines")
                text = "\n".join(ln for ln in text.split("\n")
                                 if not forbidden.scan_text(ln))
            (args.out / f"{bid}.txt").write_text(
                f"# {book['title']} (public domain, archive.org/{bid})\n\n{text}\n",
                encoding="utf-8")
            got.append({"id": bid, "title": book["title"], "region": book.get("region"),
                        "src_file": fname, "chars": len(text), "est_tokens": len(text) // 4})
            print(f"    {len(text):,} chars  ~{len(text) // 4:,} tok")
            time.sleep(1.5)

    (args.out / "manifest.json").write_text(json.dumps({
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "Internet Archive (public domain cookery books)", "license": "public-domain",
        "books": got, "total_est_tokens": sum(b["est_tokens"] for b in got),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(got)} books, ~{sum(b['est_tokens'] for b in got):,} tok -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
