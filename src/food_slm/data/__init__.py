"""Data pipeline (baseline §4, §8, Phases 4-12).

Flow:
  raw/  -> ingest -> clean (ftfy, NFKC, html strip) -> language id (fasttext)
        -> normalize (ingredients, units, TA/EN) -> dedup (MinHash LSH)
        -> knowledge extraction -> data/canonical/*.jsonl
        -> [synthetic.py] grounded generation from canonical store
        -> corpus assembly (general-EN base + culinary authentic + culinary synthetic)
        -> pack -> data/corpus/*  (gitignored)

Modules:
  forbidden.py     IMPLEMENTED - frozen forbidden-entity list: lock verification + leak checks
  serialize.py     IMPLEMENTED (PoC-minimal) - canonical store -> training text
  llm.py           IMPLEMENTED - provider-agnostic multi-model client + Panel/consensus
  ingest.py        IMPLEMENTED - self-hosted fetch (robots.txt + rate-limit + cache + trafilatura)
  licensing.py     IMPLEMENTED - source license classifier -> license_class / usable_for
  coverage.py      IMPLEMENTED - store coverage vs coverage_targets.yaml -> ranked gaps (list_gaps)
  canon_staging.py IMPLEMENTED - stage_records / promote_staged (the store gate) + 2.5% calibration
  dedup.py         IMPLEMENTED - fastembed semantic + datasketch MinHash near-duplicate detection
  extract.py       page text -> draft records (Claude Code, or a local model via llm.py)
  mcp_server.py    OPTIONAL - expose the tools to a separate session via MCP
  clean.py       unicode/html/noise normalization
  langid.py      fasttext language id + Tamil-script ratio gate (>15% -> extraction-only)
  normalize.py   ingredient / unit normalization + Tamil NFC pipeline (tamil_script_policy.md §6)
  dedup.py       exact + MinHash near-duplicate removal
  extract.py     interim -> canonical entities/relationships/processes
  synthetic.py   agent-orchestrated grounded synthesis (validated, ratio-capped)
  assemble.py    curriculum-ordered corpus assembly + entity-disjoint split
                 + Tamil-script <=0.5% token gate
  contamination.py   n-gram train x eval scan + forbidden-entity corpus scan (hard gate)
"""
