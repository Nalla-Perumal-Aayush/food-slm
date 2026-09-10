# 40M ablation — run report (2026-09-11)

**Run:** `ablation-40m-001` · `configs/model_40m.yaml` (d=512, L=14, 8 heads, ffn 1408,
vocab 16384, ctx 4096) → **53,362,944 params** · `configs/train_40m.yaml` (lr 8e-4,
curriculum `[[1024,0.45],[2048,0.80],[4096,1.0]]`, 5500 steps ≈ 1.6B tokens, micro_batch 8 ×
grad_accum 16, gradient checkpointing, bf16) · corpus `data/corpus/40m/` (228,282 windows ×
4096 = 935M train tokens; FineWeb-Edu substrate + upsampled domain slice).

Completed 2026-09-11. Survived two session-boundary process deaths mid-run; both resumed
bit-exact via `--resume auto` (loss matched the pre-stop trajectory in each case).

## Acceptance checklist (`configs/train_40m.yaml: acceptance:`)

| criterion | result | evidence |
|---|---|---|
| runs_end_to_end | ✅ | reached step 5500, `step_5500.pt` + `best.pt` written, clean exit |
| val_loss_monotone_to_5k | ✅ (monotone through 5500) | table below |
| no_sustained_spikes | ✅ | 0 spike-WARN lines over the full run (`spike_factor` 1.6× trailing-50) |
| curriculum_phases_ran: 3 | ✅ | phase boundaries exact: seq 1024 steps 0–2474, 2048 steps 2475–4399, 4096 steps 4400–5499 |
| resume_bit_exact_across_phase | ✅ | verified live — two real outages, not just a synthetic test; second resume crossed the 2048→4096 boundary |
| lr_calibrated | ✅ | 8e-4, chosen by `lr_sweep.py` over {5e-4, 8e-4, 1.5e-3} @ 800 steps/seq1024 |
| leakage_scan_clean | ✅ | `scan_leakage.py`: 21 files, 4.0B chars scanned, 0 hits (superset of the text packed into this run) |

**All 7 pass → the 89M (Config A) rung is skipped per the standing decision; proceed to
150M data completion, then the single 150M run.**

## Val loss curve

| step | val_loss | val_domain | seq_len |
|--:|--:|--:|--:|
| 500 | 4.3219 | 4.4036 | 1024 |
| 1000 | 3.8675 | 3.8523 | 1024 |
| 1500 | 3.7034 | 3.6397 | 1024 |
| 2000 | 3.6072 | 3.5317 | 1024 |
| 2500 | 3.5195 | 3.4064 | 2048 |
| 3000 | 3.4390 | 3.3215 | 2048 |
| 3500 | 3.3911 | 3.2533 | 2048 |
| 4000 | 3.3568 | 3.2269 | 2048 |
| 4500 | 3.3277 | 3.1784 | 4096 |
| 5000 | 3.3058 | 3.1627 | 4096 |
| 5500 | 3.2927 | 3.1476 | 4096 |

Clean monotone descent through every phase transition, including both curriculum jumps —
no regression when seq_len steps up, which is the main thing this rung was built to check.

## Domain probe (`scripts/domain_probe.py`, 26 items)

Qualitative, not scored — this rung deliberately runs on a domain fraction far too small
to expect real recall (its job is pipeline + LR + curriculum validation, not quality).
Results are consistent with that: structural fluency is good (grammatical, on-topic
sentence shapes, no repetition collapse outside a couple of prompts), but **factual recall
is weak-to-absent** — e.g. "Ven pongal is made from rice and" → "wheat" (want: moong dal),
"Pesarattu comes from the state of" → "Gujarat" (want: Andhra Pradesh). This is expected
at 53M params with a domain slice this thin and is **not a 40M acceptance failure** — it's
the reason the domain corpus (dish list + blog/YouTube channels, in progress) has to be
substantially larger before the 150M run.

One useful negative result: the two held-out forbidden-entity abstention probes
(`fe:0001` Arachuvitta sambar, `fe:0010` Maravalli kizhangu/tapioca) produced generic
nonsense rather than the withheld facts — i.e. no sign of leakage, though at this scale
that's more "doesn't know" than "correctly abstains."

## Depth-scaling signal

15M (L=8) → 40M (L=14) → 150M (L=20). The 40M run gives the first real mid-ladder point:
stable training at greater depth, no instability introduced by the extra layers, curriculum
and LR recipe both transfer cleanly from the 15M PoC. No depth-scaling red flags — supports
going straight to L=20 for the 150M run without an intermediate 89M rung.

## Open item

`configs/train_40m.yaml: acceptance:` still has a stale comment ("7000 steps, clean exit")
from an earlier draft of the plan; the executed run used 5500 steps. Cosmetic only — the
run described in this report is the one actually executed and accepted.
