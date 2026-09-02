"""Train a byte-level BPE tokenizer from ``configs/tokenizer.yaml`` (baseline §6).

Byte-level (no UNK — every byte is representable, so Dravidian proper nouns always
round-trip). Special + structural + chat tokens are atomic. BOS/EOS are *not* added
here — the sequence packer inserts them between documents.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from food_slm.utils import load_yaml


@dataclass
class TokenizerSpec:
    vocab_size: int
    byte_level: bool
    algorithm: str
    lowercase: bool
    nfkc_normalize: bool
    special_tokens: list[str]
    structural_tokens: list[str]
    chat_tokens: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> TokenizerSpec:
        d = load_yaml(path)
        return cls(
            vocab_size=d["vocab_size"],
            byte_level=d.get("byte_level", True),
            algorithm=d.get("algorithm", "bpe"),
            lowercase=d.get("lowercase", False),
            nfkc_normalize=d.get("nfkc_normalize", True),
            special_tokens=list(d.get("special_tokens", [])),
            structural_tokens=list(d.get("structural_tokens", [])),
            chat_tokens=list(d.get("chat_tokens", [])),
        )

    @property
    def all_special(self) -> list[str]:
        # order fixes the low ids: pad=0, bos=1, eos=2, unk=3, then structural, then chat
        seen, out = set(), []
        for t in [*self.special_tokens, *self.structural_tokens, *self.chat_tokens]:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def _build_normalizer(spec: TokenizerSpec):
    seq = []
    if spec.nfkc_normalize:
        seq.append(normalizers.NFKC())
    if spec.lowercase:
        seq.append(normalizers.Lowercase())
    return normalizers.Sequence(seq) if seq else None


def build_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    if spec.algorithm != "bpe":
        raise NotImplementedError(f"algorithm {spec.algorithm!r} not wired yet (BPE only)")
    if not spec.byte_level:
        raise NotImplementedError("only byte-level is supported (baseline §6)")

    tok = Tokenizer(models.BPE(unk_token=None))
    norm = _build_normalizer(spec)
    if norm is not None:
        tok.normalizer = norm
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    corpus_paths: list[str | Path],
    spec: TokenizerSpec,
    out_dir: str | Path,
    *,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> Path:
    """Train and save to ``out_dir/tokenizer.json`` (+ ``meta.json``). Returns ``out_dir``."""
    paths = [str(Path(p)) for p in corpus_paths]
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    tok = build_tokenizer(spec)
    trainer = trainers.BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=min_frequency,
        special_tokens=spec.all_special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train(paths, trainer)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    corpus_bytes = sum(Path(p).stat().st_size for p in paths)
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).read_bytes())
    meta = {
        "spec": spec.__dict__,
        "actual_vocab_size": tok.get_vocab_size(),
        "special_token_ids": {t: tok.token_to_id(t) for t in spec.all_special},
        "corpus": {"paths": paths, "bytes": corpus_bytes, "sha256": h.hexdigest()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    return out


def load_tokenizer(path: str | Path) -> Tokenizer:
    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer.json"
    return Tokenizer.from_file(str(p))
