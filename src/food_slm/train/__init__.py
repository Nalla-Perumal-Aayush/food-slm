"""Training loop (baseline §7).

  packing.py     IMPLEMENTED - token-stream sequence packing + memmap shard reader
  shards.py      IMPLEMENTED - multi-source sharded pack + ShardedPackedDataset (plan_40m §E)
  checkpoint.py  IMPLEMENTED - bit-exact save/resume + deterministic BatchSampler
  loop.py        IMPLEMENTED - bf16 AdamW + warmup/cosine, grad-accum, grad-clip, z-loss,
                 context curriculum, per-source val, spike log, metrics.jsonl
  curriculum.py  IMPLEMENTED - 1K -> 2K -> 4K context schedule (plan_40m §F)
"""
from food_slm.train.checkpoint import (
    BatchSampler,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from food_slm.train.curriculum import Phase, parse_curriculum, seq_len_at
from food_slm.train.loop import TrainConfig, TrainResult, lr_at, train
from food_slm.train.packing import (
    PackedDataset,
    PackManifest,
    iter_text_documents,
    pack_corpus_file,
    pack_documents,
)
from food_slm.train.shards import ShardedPackedDataset, ShardSource, pack_sharded

__all__ = [
    "PackedDataset",
    "PackManifest",
    "iter_text_documents",
    "pack_documents",
    "pack_corpus_file",
    "ShardedPackedDataset",
    "ShardSource",
    "pack_sharded",
    "BatchSampler",
    "save_checkpoint",
    "load_checkpoint",
    "prune_checkpoints",
    "TrainConfig",
    "TrainResult",
    "train",
    "lr_at",
    "Phase",
    "parse_curriculum",
    "seq_len_at",
]
