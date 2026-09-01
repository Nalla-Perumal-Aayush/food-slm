"""Tokenizer training + analysis (baseline §6, Phases 17-19).

LOCKED: vocab_size 16384, byte-level, structural/control tokens.
BENCHMARKED: BPE vs Unigram algorithm on the final corpus.

Modules (to be implemented):
  train.py     train a HF `tokenizers` model from configs/tokenizer.yaml
  analyze.py   fertility, compression ratio, vocab utilization,
               culinary-term / Tamil-name fragmentation, number/unit efficiency
"""
