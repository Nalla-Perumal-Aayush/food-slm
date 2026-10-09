"""data/instruction/*.jsonl -> tokenized, loss-masked, entity-disjoint-split SFT
training pack (master_baseline.md SS4.1's "instruction / SFT data" stage, feeding
food_slm.train.sft_loop).

Three things this does that aren't optional:

1. **Loss masking.** The model's forward() already supports an `ignore_index=-100`
   labels tensor independent of the input (food_slm.model.transformer.FoodSLM.forward)
   -- pretraining just never uses that, always calling `model(x, labels=x)`. SFT needs
   the opposite: compute loss ONLY on the assistant's response (+ the trailing <|eos|>),
   not on the prompt. Everything up to and including the <|assistant|> token is masked.

2. **Entity-disjoint exclusion against the golden QA set.** Any dish referenced by a
   golden_qa_set.jsonl item is held out of SFT training ENTIRELY (not train, not
   val) -- never fine-tuned on, so later AI-judge scoring against the golden set
   measures real generalization, not memorization of the exact dishes being tested.
   This is the baseline's "entity-disjoint split" rule (SS4.1), applied against the
   external eval set specifically, which is stricter than an ordinary internal split.

3. **Entity-disjoint internal train/val split.** Of what's left, grouped by the
   underlying dish/ingredient/technique/term entity (a recipe's `backing_record_ids`
   resolves to its `structure.dish_id`, not the recipe record's own id, so multiple
   recipes/examples for the same dish stay together) -- never split at the example
   level, which would leak the same dish's facts into both sides.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from food_slm.schema.store import CanonicalStore
from food_slm.tokenizer.train import load_tokenizer

SPECIALS = {"pad": "<|pad|>", "bos": "<|bos|>", "eos": "<|eos|>", "user": "<|user|>",
           "assistant": "<|assistant|>"}


def normalize_dish_slug(ref: str) -> str:
    """dish_list_v1-style ("fish_curry") and canonical-store-style
    ("dish:fish-curry") both normalize to the same slug for comparison."""
    return ref.removeprefix("dish:").replace("_", "-").lower()


def resolve_entity_key(record_id: str, store: CanonicalStore) -> str:
    """A recipe's OWN id (`recipe:{dish}-{source}`) is source-specific, not dish-
    specific -- two different recipes for the same dish must still land in the same
    split, so a recipe resolves to its `structure.dish_id` instead of its own id.
    Every other backing type (dish/ingredient/technique/term/...) is already
    entity-level, so its own id is the right grouping key."""
    rtype = record_id.split(":", 1)[0]
    if rtype == "recipe":
        rec = store.records.get(record_id)
        if rec is not None:
            return rec.structure.dish_id
    return record_id


def load_golden_dish_slugs(golden_path: Path) -> set[str]:
    if not golden_path.exists():
        return set()
    slugs = set()
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for d in row.get("dish_ids") or []:
            slugs.add(normalize_dish_slug(d))
    return slugs


@dataclass
class SplitResult:
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    excluded_golden_overlap: list[dict] = field(default_factory=list)
    excluded_dish_slugs: set[str] = field(default_factory=set)


def entity_disjoint_split(
    rows: list[dict],
    store: CanonicalStore,
    golden_dish_slugs: set[str],
    *,
    val_frac: float = 0.08,
    seed: int = 1337,
) -> SplitResult:
    """Groups by resolved entity key (see resolve_entity_key); a row with no
    backing_record_ids (e.g. a hand-authored abstention item -- general behaviour,
    not a dish-specific fact) has nothing to leak and goes straight to train."""
    res = SplitResult()
    ungrouped: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for r in rows:
        backing = r.get("provenance", {}).get("backing_record_ids") or []
        if not backing:
            ungrouped.append(r)
            continue
        keys = {resolve_entity_key(b, store) for b in backing}
        # Multiple backing ids resolving to different entities (e.g. a
        # regional_comparison's two dishes) -- group under a combined key so the
        # whole example moves together; golden-overlap exclusion checks each
        # resolved key independently, so overlap with EITHER dish excludes it.
        combined_key = "+".join(sorted(keys))
        groups.setdefault(combined_key, []).append(r)
        for k in keys:
            slug = normalize_dish_slug(k)
            if slug in golden_dish_slugs:
                res.excluded_dish_slugs.add(k)

    res.train.extend(ungrouped)

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    n_val_target = int(len(rows) * val_frac)
    val_count = 0
    for gk in group_keys:
        group_rows = groups[gk]
        resolved = gk.split("+")
        slugs = {normalize_dish_slug(k) for k in resolved}
        if slugs & golden_dish_slugs:
            res.excluded_golden_overlap.extend(group_rows)
            continue
        if val_count < n_val_target:
            res.val.extend(group_rows)
            val_count += len(group_rows)
        else:
            res.train.extend(group_rows)
    return res


def tokenize_example(tok, user: str, assistant: str, max_seq_len: int,
                     special_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Builds `<|bos|><|user|>{q}<|assistant|>{a}<|eos|>`, masks everything up to
    and including <|assistant|> (loss only on the response + trailing <|eos|>),
    right-pads with <|pad|> (safe under this model's strictly-causal attention --
    no padding-mask parameter exists or is needed, a real token never attends
    forward into padding). Returns None if the example doesn't fit even after
    this template, so the caller can count/report truncation-free coverage rather
    than silently cut content."""
    bos, eos, pad = special_ids["bos"], special_ids["eos"], special_ids["pad"]
    user_ids = [special_ids["user"], *tok.encode(user).ids]
    asst_ids = [special_ids["assistant"], *tok.encode(assistant).ids, eos]
    full = [bos, *user_ids, *asst_ids]
    if len(full) > max_seq_len:
        return None

    # input_ids: uint16 on disk (vocab_size=16384 fits easily), matching this
    # project's existing packed-corpus convention (food_slm.train.packing.DTYPE) --
    # cast to int64 at batch-read time (sft_loop.SFTDataset.batch), same pattern.
    # labels needs a SIGNED dtype (-100 for masked positions), so int32 here.
    input_ids = np.full(max_seq_len, pad, dtype=np.uint16)
    labels = np.full(max_seq_len, -100, dtype=np.int32)
    input_ids[:len(full)] = full

    # Shift semantics (FoodSLM.forward does shift_logits = logits[:, :-1],
    # shift_labels = labels[:, 1:]): the prediction computed FROM position i is
    # scored against labels[i+1]. So to train "predict response token at position
    # p", set labels[p] = input_ids[p] -- that label gets compared against the
    # prediction made from position p-1. The <|assistant|> marker itself, at
    # resp_start, is never a prediction TARGET we care about (nothing meaningful
    # hinges on the model "predicting" a fixed template delimiter, and the
    # inference harness supplies that marker explicitly rather than ever asking
    # the model to generate it) -- so resp_start's label stays masked (-100), and
    # only resp_start+1 .. end (the real response tokens + trailing <|eos|>) carry
    # a real label. Exact indices verified in tests/test_sft_pack.py, not just
    # reasoned about here, since off-by-one errors in loss masking are easy to
    # introduce and hard to notice (training still "runs", just on the wrong span).
    resp_start = 1 + len(user_ids)  # position of the <|assistant|> marker itself
    labels[resp_start + 1:len(full)] = full[resp_start + 1:]
    return input_ids, labels


@dataclass
class PackStats:
    train: int = 0
    val: int = 0
    excluded_golden_overlap: int = 0
    excluded_golden_dish_count: int = 0
    skipped_too_long: int = 0


def build_sft_pack(
    instruction_path: Path,
    canonical_dir: Path,
    golden_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    *,
    max_seq_len: int = 768,
    val_frac: float = 0.08,
    seed: int = 1337,
) -> PackStats:
    from food_slm.schema import load_store

    rows = [json.loads(line) for line in instruction_path.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    store = load_store(canonical_dir, strict=False)
    golden_slugs = load_golden_dish_slugs(golden_path)

    split = entity_disjoint_split(rows, store, golden_slugs, val_frac=val_frac, seed=seed)

    tok = load_tokenizer(tokenizer_path)
    meta = json.loads((tokenizer_path / "meta.json").read_text(encoding="utf-8"))
    special_ids = {name: meta["special_token_ids"][tag] for name, tag in SPECIALS.items()}

    stats = PackStats(excluded_golden_overlap=len(split.excluded_golden_overlap),
                      excluded_golden_dish_count=len(split.excluded_dish_slugs))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in (("train", split.train), ("val", split.val)):
        all_input_ids, all_labels = [], []
        for r in split_rows:
            user = r["messages"][0]["content"]
            assistant = r["messages"][1]["content"]
            packed = tokenize_example(tok, user, assistant, max_seq_len, special_ids)
            if packed is None:
                stats.skipped_too_long += 1
                continue
            all_input_ids.append(packed[0])
            all_labels.append(packed[1])
        if split_name == "train":
            stats.train = len(all_input_ids)
        else:
            stats.val = len(all_input_ids)
        np.savez(out_dir / f"{split_name}.npz",
                input_ids=np.stack(all_input_ids) if all_input_ids else np.zeros((0, max_seq_len), dtype=np.uint16),
                labels=np.stack(all_labels) if all_labels else np.zeros((0, max_seq_len), dtype=np.int32))

    manifest = {
        "max_seq_len": max_seq_len, "val_frac": val_frac, "seed": seed,
        "tokenizer": str(tokenizer_path), "special_token_ids": special_ids,
        "train_examples": stats.train, "val_examples": stats.val,
        "excluded_golden_overlap_examples": stats.excluded_golden_overlap,
        "excluded_golden_dish_count": stats.excluded_golden_dish_count,
        "skipped_too_long": stats.skipped_too_long,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats
