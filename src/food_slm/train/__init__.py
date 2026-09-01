"""Training loop (baseline §7).

  packing.py     token-stream sequence packing + shard reader
  loop.py        bf16 AdamW + cosine, grad-accum, grad-clip, z-loss
  checkpoint.py  save/resume (model + optim + scheduler + rng + step) — must be bit-exact
  curriculum.py  1K -> 2K -> 4K schedule + corpus-stage ordering
"""
