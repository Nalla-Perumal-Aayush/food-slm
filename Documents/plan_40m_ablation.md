# 40M Ablation — Execution Plan

**Status:** in progress (started 2026-09-08). Tracks the subtask checklist from baseline Rev 8.
This is a *tactical* working doc; the baseline (§12, Rev 8) is the reference for what the rung
is and why.

## Purpose of the rung (what it must answer)

1. **LR calibration** — the 15M lr (6e-4) will not transfer (not µP). Sweep and pick.
2. **Curriculum** — the 1K→2K→4K mechanism (`train/curriculum.py`) does not exist. Build and
   prove it runs clean across `seq_len` boundaries with bit-exact resume.
3. **At-scale data pipeline** — sharded streaming pack, multi-shard `PackedDataset`,
   source-split validation, corpus-level forbidden-entity / contamination scan.
4. **Long-run stability** — survive ≥20k steps unbabysat; loss-spike logging; checkpoint
   pruning + resume over a half-day run.
5. **Depth-scaling signal** — L=8 (15M) → **L=14 (40M)** → L=20 (150M). If clean, **skip 89M**.

It does **not** measure final model quality. A 53M model on ~1B tokens with a thin domain
slice is judged for pipeline correctness + training dynamics, not culinary competence.

## Locked decisions (Rev 8)

| Item | Value |
|---|---|
| Geometry | d=512, L=14, num_heads 8 (head_dim 64), num_kv_heads 8 (MHA), ffn 1408 (SwiGLU), vocab **TBD by §4 benchmark** → **53.4M** at vocab 16384 |
| General-English source | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`), ODC-By |
| General-English budget | ~1.0B tokens (≈ one Chinchilla-optimal pass for 53M) |
| Domain slice | serialized canonical store (×N upsample) + permissive culinary-prose slice (CC-BY-SA Wikipedia cuisine articles), target ~15–30M tokens, weighted to effective ~5–8% |
| Context | 4096 max; curriculum 1K → 2K → 4K |
| Precision | bf16 (locked) |
| 89M rung | SKIPPED if 40M clean; fallback only |

## Subtask checklist

### A. Decisions — DONE (Rev 8)

### B. General-English corpus
- [ ] B1. Scale `scripts/fetch_sample.py` (or a new `fetch_corpus.py`) to stream FineWeb-Edu to
  a token budget, writing **sharded** text on D: (`data/interim/corpus_40m/shard_*.txt`), not
  one in-memory file.
- [ ] B2. Language-ID filter → English only (fasttext `lid.176`).
- [ ] B3. Doc-level dedup (extend `data/dedup.py` MinHash from records to docs).
- [ ] B4. Per-shard provenance/license manifest (source, ODC-By, snapshot, token count).
- [ ] B5. Disk check (have 251 GB free; ~1B tokens raw ≈ 5–7 GB text + ~2 GB uint16 shards).

### C. Domain corpus slice
- [ ] C1. Grow canonical store toward ~200 records (continue extraction batches).
- [ ] C2. `build_poc_corpus.py` → generalize to emit a standalone domain shard with a
  configurable repeat factor.
- [ ] C3. Permissive culinary-prose slice: fetch ~200–400 CC-BY-SA Wikipedia South-Indian-food
  articles (same license as the wikitext base), clean via trafilatura, dedup.
- [ ] C4. Entity-disjoint check: domain slice vs `forbidden_entities.jsonl` and probe set.

### D. Tokenizer — retrain + lock vocab
- [ ] D1. Assemble a ~200–400 MB tokenizer-training sample (general + domain, proportional).
- [ ] D2. Retrain byte-level BPE at **16k and 24k**.
- [ ] D3. Benchmark: fertility (tokens/word), compression (bytes/token), Dravidian-script token
  share (**≤0.5% hard gate**), structural-token integrity, web-text vs wikitext behavior.
- [ ] D4. **Lock vocab size** (carries to 150M). Update `configs/tokenizer.yaml`, regenerate
  `meta.json`.
- [ ] D5. Re-run `analyze_tokenizer`; archive the report to `artifacts/reports/`.

### E. Packing at scale
- [ ] E1. `packing.py` — streaming/sharded pack: many `shard_*.bin` + a top-level manifest, no
  whole-corpus-in-memory.
- [ ] E2. `PackedDataset` — multi-shard reader; deterministic cross-shard shuffle
  (`BatchSampler` currently assumes one window array).
- [ ] E3. Manifest v2 — per-shard SHA, total tokens, per-source mix ratios.
- [ ] E4. Val split by **document/entity**, not random windows.
- [ ] E5. Weighted sampling between general + domain shard sets (upsample ratio).

### F. Curriculum — `train/curriculum.py`
- [ ] F1. Phase schedule (step ranges → `seq_len`).
- [ ] F2. Mechanism: pack at 4K, feed shorter windows per phase (sub-window slicing) — avoids
  re-packing per phase.
- [ ] F3. Continuous cosine LR across phase boundaries.
- [ ] F4. Checkpoint/resume across a `seq_len` change (bit-exact test).
- [ ] F5. `TrainConfig` curriculum fields; RoPE θ 10000 verified at 4K.

### G. Training-loop extensions
- [ ] G1. LR-sweep harness — 3–4 short runs (~2k steps) at {3e-4, 6e-4, 1e-3, 2e-3}, pick best.
- [ ] G2. Loss-spike detection + structured logging.
- [ ] G3. Grad-checkpointing tested at 4K (16 GB headroom) + loss-identical check.
- [ ] G4. Metrics: tokens-seen, throughput, MFU, per-phase; tracker = offline jsonl.
- [ ] G5. Confirm `keep_last_n` prune + `resume auto` over a mid-run kill.

### H. Model config + validation
- [ ] H1. Rewrite `configs/model_40m.yaml` → d=512, L=14, ffn 1408, max_seq_len 4096.
- [ ] H2. Mark `configs/model_89m.yaml` conditional (comment; keep the file).
- [ ] H3. `scripts/validate_model.py` on the new 40M — analytic == actual params,
  fwd/bwd finite, grad-ckpt loss-identical, overfit-single-batch.

### I. Eval harness (minimal)
- [ ] I1. Val perplexity **split by source** (general vs domain).
- [ ] I2. Corpus-level forbidden-entity / leakage scan (`data/forbidden.py` at scale).
- [ ] I3. Entity-disjoint split verification.
- [ ] I4. ~20–50 item hand domain probe set for the post-run eyeball.
- [ ] I5. `configs/train_40m.yaml` `acceptance:` block.

## Critical path

A ✓ → **B + C** (parallel, B1 download starts first, runs in background) → **D** (needs a
corpus sample) → **E** (needs tokenizer) → **F + G + H** (parallel, need packing) → **G1 LR
sweep** → **I** wired in → **40M ablation run**.

## Explicitly out of scope (→ 150M prep)

Full ~20k canonical store · full synthetic-data pipeline · ~1500 golden set + human
calibration loop · GB-scale domain curation · multi-week ops hardening · MinHash doc-dedup
(FineWeb-Edu is internally deduped) · weighted runtime source-sampling (40M bakes the mix
at pack time).

---

## Progress log

**2026-09-08 (session 2)**

- **E, F, G(most), H, I — DONE.** `train/shards.py` (`pack_sharded` + `ShardedPackedDataset`,
  per-source doc-disjoint val, per-shard SHA manifest, domain upsample). `train/curriculum.py`
  (1K→2K→4K, sub-window slicing, step-only LR, bit-exact resume across a phase — tested).
  `loop.py` auto-detects sharded packs, runs the curriculum, per-source val, spike-WARN,
  `metrics.jsonl`, `tokens_seen`. `model_40m.yaml` → L=14 (53.36M), `validate_model.py` PASS.
  Commits `9b2b5d3`, `48737b3`. 121 fast tests.
- **B — code done, download running.** `scripts/fetch_corpus.py` streaming FineWeb-Edu →
  `data/interim/corpus_40m/fineweb-edu/shard_*.txt`; **1B-token run ~50% done**. `data/langid.py`
  (fastText lid.176 + NumPy-2 shim). Commit `a0ca371`.
- **C — small slice fetched.** `scripts/fetch_domain_wiki.py` → ~87 CC-BY-SA Wikipedia articles
  (intros; `--full` pass running). ~15–40k tokens — minimal by design (the 40M rung does not
  test specialization). GB-scale curation stays 150M-prep.
- **CLIs staged for the run:** `pack_40m.py`, `tokenizer_benchmark.py`, `lr_sweep.py`,
  `scan_leakage.py`, `domain_probe.py`; `configs/train_40m.yaml` with the `acceptance:` block.

### Remaining before launch
1. ✅ FineWeb-Edu 1B (843k docs, 20 shards) + domain slice (86 articles / ~129k tok).
2. ✅ **D** — vocab **LOCKED 16k** (`artifacts/reports/tokenizer_decision_2026-09-08.md`);
   final tokenizer `artifacts/tokenizers/corpus-16k` (1 GB spread sample).
3. ✅ `scan_leakage.py` over 4.0B chars — **CLEAN** (after amending `fe:0009` to drop the
   place-name-colliding bare "Kadappa").
4. ✅ `pack_40m.py` → `data/corpus/40m/` — 228,282 × 4096 = **935M tokens**, val 589 + 8,
   domain share 0.2%.
5. 🔄 `lr_sweep.py` running — {5e-4, 8e-4, 1.5e-3} × 800 steps @ seq 1024 (~2.5 h).
6. ⏳ 4K-phase VRAM check (grad-ckpt on) — after the sweep.
7. ✅ LAUNCHED 2026-09-09 —  the **5500-step** run (~1.6B tokens; curriculum 1K/2K/4K at 45/35/20% of
   steps). Measured ~15–17k tok/s @ 70 W → **~1.5 days wall**. Depth check: 15M L=8 →
   40M **L=14** → 150M L=20.
