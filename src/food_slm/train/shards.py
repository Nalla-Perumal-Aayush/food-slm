"""Sharded, multi-source sequence packing for the 40M+ runs (plan_40m §E).

Extends ``packing.py`` (single in-memory corpus -> one ``tokens.bin``) to:
  - stream many source ``.txt`` shards without holding the corpus in memory,
  - mix sources with a per-source upsample factor (domain data repeated N times),
  - hold out a **document-disjoint** validation split, per source,
  - write fixed-size ``.bin`` shards + a manifest with per-shard SHA-256 and the
    realised source mix.

Windowing matches ``pack_documents``: tokenize each doc, append ``<|eos|>``, concat,
cut into ``seq_len`` windows, drop the trailing remainder.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from food_slm.train.packing import DTYPE, iter_text_documents


@dataclass
class ShardSource:
    name: str
    paths: list[Path]
    repeat: int = 1          # upsample factor (1 = general, >1 = domain)

    def docs(self) -> Iterator[str]:
        for p in self.paths:
            yield from iter_text_documents(p)


@dataclass
class ShardManifest:
    seq_len: int
    total_windows: int
    val_windows: int
    shards: list[dict]
    val_files: dict[str, dict]
    dtype: str
    eos_id: int
    bos_id: int | None
    tokenizer_meta_sha256: str | None
    source_mix: dict
    seed: int
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def _windows_from_docs(docs: Iterator[str], tokenizer, seq_len: int, eos_id: int):
    """Yield seq_len token windows (lists of int) from a doc stream."""
    buf: list[int] = []
    n_tokens = 0
    for doc in docs:
        ids = tokenizer.encode(doc).ids
        buf.extend(ids)
        buf.append(eos_id)
        n_tokens += len(ids) + 1
        while len(buf) >= seq_len:
            yield buf[:seq_len], n_tokens
            del buf[:seq_len]
            n_tokens = 0


def _write_bin(path: Path, windows: list[np.ndarray]) -> str:
    arr = np.concatenate(windows).astype(DTYPE) if windows else np.array([], dtype=DTYPE)
    path.write_bytes(arr.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_sharded(
    sources: list[ShardSource],
    tokenizer_dir: str | Path,
    seq_len: int,
    out_dir: str | Path,
    *,
    val_docs_per_source: int = 2000,
    shard_windows: int = 200_000,
    seed: int = 1337,
) -> ShardManifest:
    from food_slm.tokenizer.train import load_tokenizer

    tok_dir = Path(tokenizer_dir)
    tok = load_tokenizer(tok_dir)
    meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    eos_id = meta["special_token_ids"]["<|eos|>"]
    bos_id = meta["special_token_ids"].get("<|bos|>")
    meta_sha = hashlib.sha256((tok_dir / "meta.json").read_bytes()).hexdigest()

    out = Path(out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- validation: document-disjoint, held out per source BEFORE repeat/shuffle ---
    val_files: dict[str, dict] = {}
    train_doc_lists: list[tuple[str, int, list[str]]] = []
    val_all_windows: list[np.ndarray] = []
    for src in sources:
        all_docs = list(src.docs())
        rng.shuffle(all_docs)
        v = min(val_docs_per_source, len(all_docs) // 4)
        val_docs, train_docs = all_docs[:v], all_docs[v:]
        vw = [np.asarray(w, dtype=DTYPE)
              for w, _ in _windows_from_docs(iter(val_docs), tok, seq_len, eos_id)]
        if vw:
            vp = out / "val" / f"{src.name}.bin"
            sha = _write_bin(vp, vw)
            val_files[src.name] = {"file": str(vp.relative_to(out)), "n_windows": len(vw),
                                   "sha256": sha}
            val_all_windows.extend(vw)
        train_doc_lists.append((src.name, src.repeat, train_docs))

    if val_all_windows:
        _write_bin(out / "val.bin", val_all_windows)

    # --- training: repeat domain, interleave, shuffle at doc level, stream to shards ---
    mixed: list[tuple[str, str]] = []
    for name, repeat, docs in train_doc_lists:
        for _ in range(repeat):
            mixed.extend((name, d) for d in docs)
    rng.shuffle(mixed)

    src_tokens: dict[str, int] = {s.name: 0 for s in sources}
    src_docs: dict[str, int] = {s.name: 0 for s in sources}

    def _tagged_docs() -> Iterator[str]:
        for name, d in mixed:
            src_docs[name] += 1
            src_tokens[name] += len(tok.encode(d).ids) + 1
            yield d

    shards: list[dict] = []
    cur: list[np.ndarray] = []
    total_windows = shard_idx = 0
    for window, _ in _windows_from_docs(_tagged_docs(), tok, seq_len, eos_id):
        cur.append(np.asarray(window, dtype=DTYPE))
        total_windows += 1
        if len(cur) >= shard_windows:
            p = out / "train" / f"shard_{shard_idx:05d}.bin"
            shards.append({"file": str(p.relative_to(out)), "n_windows": len(cur),
                           "sha256": _write_bin(p, cur)})
            cur = []
            shard_idx += 1
    if cur:
        p = out / "train" / f"shard_{shard_idx:05d}.bin"
        shards.append({"file": str(p.relative_to(out)), "n_windows": len(cur),
                       "sha256": _write_bin(p, cur)})

    man = ShardManifest(
        seq_len=seq_len, total_windows=total_windows,
        val_windows=sum(v["n_windows"] for v in val_files.values()),
        shards=shards, val_files=val_files, dtype=str(np.dtype(DTYPE)),
        eos_id=eos_id, bos_id=bos_id, tokenizer_meta_sha256=meta_sha,
        source_mix={s.name: {"repeat": s.repeat, "train_docs": src_docs[s.name],
                             "train_tokens": src_tokens[s.name]} for s in sources},
        seed=seed,
    )
    (out / "manifest.json").write_text(man.to_json() + "\n", encoding="utf-8")
    return man


class ShardedPackedDataset:
    """Read-only view over ``train/shard_*.bin`` (or a single ``val*.bin``) as one
    contiguous window array. ``ds[i] -> (input_ids, labels)`` int64."""

    def __init__(self, shard_dir: str | Path, split: str = "train") -> None:
        d = Path(shard_dir)
        self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.seq_len = self.manifest["seq_len"]
        itemsize = np.dtype(DTYPE).itemsize

        if split == "train":
            files = [d / s["file"] for s in self.manifest["shards"]]
        elif split == "val":
            files = [d / "val.bin"]
        else:  # per-source val, e.g. "val:fineweb-edu"
            name = split.split(":", 1)[1]
            files = [d / self.manifest["val_files"][name]["file"]]

        self._mms: list[np.memmap] = []
        self._offsets = [0]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f)
            n = f.stat().st_size // (itemsize * self.seq_len)
            if n == 0:
                continue
            self._mms.append(np.memmap(f, dtype=DTYPE, mode="r", shape=(n, self.seq_len)))
            self._offsets.append(self._offsets[-1] + n)
        self._n = self._offsets[-1]

    def __len__(self) -> int:
        return self._n

    def _row(self, i: int) -> np.ndarray:
        s = int(np.searchsorted(self._offsets, i, side="right")) - 1
        return np.asarray(self._mms[s][i - self._offsets[s]], dtype=np.int64)

    def __getitem__(self, i: int):
        row = self._row(i)
        return row, row.copy()

    def batch(self, idx) -> np.ndarray:
        return np.stack([self._row(int(j)) for j in idx])
