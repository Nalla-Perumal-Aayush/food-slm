"""Training loop (baseline §7).

  packing.py     IMPLEMENTED - token-stream sequence packing + memmap shard reader
  loop.py        bf16 AdamW + cosine, grad-accum, grad-clip, z-loss
  checkpoint.py  save/resume (model + optim + scheduler + rng + step) - must be bit-exact
  curriculum.py  1K -> 2K -> 4K schedule + corpus-stage ordering
"""
from food_slm.train.packing import (
    PackedDataset,
    PackManifest,
    iter_text_documents,
    pack_corpus_file,
    pack_documents,
)

__all__ = [
    "PackedDataset",
    "PackManifest",
    "iter_text_documents",
    "pack_documents",
    "pack_corpus_file",
]
