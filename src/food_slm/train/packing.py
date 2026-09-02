"""Token-stream sequence packing + a memmap shard reader (baseline §7).

Packing: tokenize each document, append ``<|eos|>``, concatenate the whole stream,
then cut it into fixed-length ``seq_len`` windows. The trailing remainder shorter
than ``seq_len`` is dropped. Output is a flat ``uint16`` ``.bin`` (vocab <= 65536)
plus a JSON manifest with the seq_len, counts, dtype, and content hashes for
reproducibility.

``PackedDataset`` memory-maps the ``.bin`` and yields ``(input_ids, labels)`` with
``labels == input_ids`` — the model shifts internally.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DTYPE = np.uint16  # vocab_size <= 65536


def iter_text_documents(path: str | Path, *, sep: str = "\n\n") -> Iterator[str]:
    """Yield documents from a corpus text file, skipping ``#`` comment lines."""
    buf: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.strip() == "" and buf:
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                buf = []
            else:
                buf.append(line)
    if buf:
        doc = "".join(buf).strip()
        if doc:
            yield doc
    _ = sep  # kept for signature stability; splitting is blank-line based


@dataclass
class PackManifest:
    seq_len: int
    n_windows: int
    n_tokens_packed: int
    n_tokens_raw: int
    n_documents: int
    dropped_tail_tokens: int
    dtype: str
    eos_id: int
    bos_id: int | None
    add_bos_per_window: bool
    val_windows: int
    tokenizer_meta_sha256: str | None
    corpus_sha256: str
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def pack_documents(
    documents: Iterable[str],
    tokenizer,
    seq_len: int,
    out_dir: str | Path,
    *,
    eos_id: int,
    bos_id: int | None = None,
    add_bos_per_window: bool = False,
    val_fraction: float = 0.0,
    corpus_sha256: str = "",
    tokenizer_meta_sha256: str | None = None,
    flush_every: int = 4096,
) -> PackManifest:
    """Pack *documents* into ``out_dir/tokens.bin`` (+ ``manifest.json``, + ``val.bin``)."""
    if add_bos_per_window and bos_id is None:
        raise ValueError("add_bos_per_window requires bos_id")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / "tokens.bin"

    buf: list[int] = []
    windows: list[np.ndarray] = []
    n_raw = n_docs = n_windows = 0
    body = seq_len - 1 if add_bos_per_window else seq_len

    with bin_path.open("wb") as fh:
        def _flush() -> None:
            nonlocal windows
            if windows:
                np.concatenate(windows).astype(DTYPE).tofile(fh)
                windows = []

        for doc in documents:
            ids = tokenizer.encode(doc).ids
            n_raw += len(ids) + 1
            n_docs += 1
            buf.extend(ids)
            buf.append(eos_id)
            while len(buf) >= body:
                chunk = buf[:body]
                del buf[:body]
                if add_bos_per_window:
                    chunk = [bos_id, *chunk]
                windows.append(np.asarray(chunk, dtype=DTYPE))
                n_windows += 1
                if len(windows) >= flush_every:
                    _flush()
        _flush()

    dropped = len(buf)
    total_windows = n_windows

    val_windows = int(total_windows * val_fraction)
    train_windows = total_windows - val_windows
    if val_windows > 0:
        mm = np.memmap(bin_path, dtype=DTYPE, mode="r", shape=(total_windows, seq_len))
        np.ascontiguousarray(mm[train_windows:]).tofile(str(out / "val.bin"))
        np.ascontiguousarray(mm[:train_windows]).tofile(str(out / "train.bin"))
        del mm
        bin_path.unlink()
        (out / "train.bin").rename(bin_path)  # tokens.bin == the train shard

    man = PackManifest(
        seq_len=seq_len, n_windows=train_windows, n_tokens_packed=train_windows * seq_len,
        n_tokens_raw=n_raw, n_documents=n_docs, dropped_tail_tokens=dropped,
        dtype=str(np.dtype(DTYPE)), eos_id=eos_id, bos_id=bos_id,
        add_bos_per_window=add_bos_per_window, val_windows=val_windows,
        tokenizer_meta_sha256=tokenizer_meta_sha256, corpus_sha256=corpus_sha256,
    )
    (out / "manifest.json").write_text(man.to_json() + "\n", encoding="utf-8")
    return man


def pack_corpus_file(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    seq_len: int,
    out_dir: str | Path,
    *,
    add_bos_per_window: bool = False,
    val_fraction: float = 0.0,
) -> PackManifest:
    from food_slm.tokenizer.train import load_tokenizer

    corpus_path = Path(corpus_path)
    tok_dir = Path(tokenizer_dir)
    tok = load_tokenizer(tok_dir)
    meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    ids = meta["special_token_ids"]

    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    meta_sha = hashlib.sha256((tok_dir / "meta.json").read_bytes()).hexdigest()

    return pack_documents(
        iter_text_documents(corpus_path),
        tok,
        seq_len,
        out_dir,
        eos_id=ids["<|eos|>"],
        bos_id=ids["<|bos|>"],
        add_bos_per_window=add_bos_per_window,
        val_fraction=val_fraction,
        corpus_sha256=corpus_sha,
        tokenizer_meta_sha256=meta_sha,
    )


class PackedDataset:
    """Read-only memmap view of a packed shard. ``ds[i] -> (input_ids, labels)`` int64."""

    def __init__(self, shard_dir: str | Path, split: str = "train") -> None:
        d = Path(shard_dir)
        self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.seq_len = self.manifest["seq_len"]
        fname = "tokens.bin" if split == "train" else f"{split}.bin"
        path = d / fname
        if not path.exists():
            raise FileNotFoundError(path)
        n = path.stat().st_size // (np.dtype(DTYPE).itemsize * self.seq_len)
        self._mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(n, self.seq_len))

    def __len__(self) -> int:
        return self._mm.shape[0]

    def __getitem__(self, i: int):
        row = np.asarray(self._mm[i], dtype=np.int64)
        return row, row.copy()
