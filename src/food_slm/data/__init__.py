"""Data pipeline (baseline §4, §8, Phases 4-12).

Flow:
  raw/  -> ingest -> clean (ftfy, NFKC, html strip) -> language id (fasttext)
        -> normalize (ingredients, units, TA/EN) -> dedup (MinHash LSH)
        -> knowledge extraction -> data/canonical/*.jsonl
        -> [synthetic.py] grounded generation from canonical store
        -> corpus assembly (general-EN base + culinary authentic + culinary synthetic)
        -> pack -> data/corpus/*  (gitignored)

Modules (to be implemented):
  ingest.py      raw source -> interim records + provenance
  clean.py       unicode/html/noise normalization
  langid.py      fasttext language identification
  normalize.py   ingredient / unit / TA-EN normalization
  dedup.py       exact + MinHash near-duplicate removal
  extract.py     interim -> canonical entities/relationships/processes
  synthetic.py   agent-orchestrated grounded synthesis (validated, ratio-capped)
  assemble.py    curriculum-ordered corpus assembly + entity-disjoint split
  contamination.py   n-gram train x eval scan (hard gate)
"""
