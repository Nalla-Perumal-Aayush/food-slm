# Tokenizer vocab decision — 16k vs 24k

**Date:** 2026-09-08 · **Decision: LOCK vocab_size = 16384** (byte-level BPE)
**Basis:** `scripts/tokenizer_benchmark.py` on 424 MB of the real corpus (FineWeb-Edu +
the CC-BY-SA domain slice). Full data: `artifacts/reports/tokenizer_benchmark_2026-09-08.json`.

## Measured

| metric | 16k | 24k | 24k vs 16k |
|---|---|---|---|
| general fertility (tok/word) | 1.418 | 1.358 | −4.2% |
| general compression (bytes/tok) | 4.359 | 4.550 | **+4.4%** |
| domain fertility | 1.704 | 1.613 | −5.3% |
| domain compression (bytes/tok) | 3.756 | 3.967 | **+5.6%** |
| culinary-name fragmentation (tok/name) | 3.96 | 3.66 | −7.6% |
| vocab utilization | 98.1% | 96.7% | — |
| Dravidian-script vocab share | 0.0% | 0.0% | gate 8% — n/a for this corpus |
| structural tokens single-id | yes | yes | — |

## Why 16k

1. **The compression gain is real but small** (~4.5%). It buys ~4.5% more content per
   token — not decisive for a from-scratch model, and partly recoverable with a slightly
   longer run.
2. **24k eats the 150M param headroom.** Config B is locked at ~150M **±5%** (currently
   151.2M, +0.8%). Moving the tied embedding from 16384×768 to 24576×768 adds **6.29M
   params → 157.5M = +4.31% of target**, leaving almost no room for any other
   implementation fix. The ±5% tolerance is for rounding, not a budget to spend on vocab.
3. **16k is not starving** — 98.1% vocab utilization, structural tokens atomic, and the
   4-script Dravidian guardrail is irrelevant here (0% native script in an English corpus;
   it becomes live only when the 150M data programme adds native-script content).
4. Culinary-name fragmentation is mediocre at **both** sizes (~4 tokens/name) — byte-level
   BPE will not make rarely-seen proper nouns atomic regardless of vocab size. The
   serialization format relies on the structural tokens, which are single-id in both.

## Revisit criteria (would move to 24k)

- The 150M architecture is re-opened and gains param headroom, **or**
- The 150M data programme's compression on a much larger, native-script-inclusive corpus
  shows a materially larger 24k advantage (> ~8%), **or**
- The 40M ablation shows the model is context-bound in a way better compression would fix.

Otherwise 16k carries through to the 150M release.
