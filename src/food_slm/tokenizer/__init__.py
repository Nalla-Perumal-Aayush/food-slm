"""Tokenizer training + analysis (baseline §6, Phases 17-19).

LOCKED: byte-level (no UNK), structural/control tokens.
BENCHMARKED: vocab size (16k vs 24k) + BPE vs Unigram algorithm on the final corpus.

  train.py     byte-level BPE training from configs/tokenizer.yaml
  analyze.py   fertility, compression, vocab utilization, culinary-term /
               native-name fragmentation, number/unit efficiency
"""
from food_slm.tokenizer.analyze import analyze_tokenizer
from food_slm.tokenizer.train import (
    TokenizerSpec,
    build_tokenizer,
    load_tokenizer,
    train_tokenizer,
)

__all__ = [
    "TokenizerSpec",
    "build_tokenizer",
    "train_tokenizer",
    "load_tokenizer",
    "analyze_tokenizer",
]
