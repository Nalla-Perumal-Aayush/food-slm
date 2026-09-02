"""Training loop (baseline §7).

  packing.py     IMPLEMENTED - token-stream sequence packing + memmap shard reader
  checkpoint.py  IMPLEMENTED - bit-exact save/resume + deterministic BatchSampler
  loop.py        IMPLEMENTED - bf16 AdamW + warmup/cosine, grad-accum, grad-clip, z-loss
  curriculum.py  1K -> 2K -> 4K schedule + corpus-stage ordering
"""
from food_slm.train.checkpoint import (
    BatchSampler,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from food_slm.train.loop import TrainConfig, TrainResult, lr_at, train
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
    "BatchSampler",
    "save_checkpoint",
    "load_checkpoint",
    "prune_checkpoints",
    "TrainConfig",
    "TrainResult",
    "train",
    "lr_at",
]
