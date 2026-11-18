"""SFT training loop -- a dedicated, simpler sibling of food_slm.train.loop's
pretraining loop, not a retrofit of it.

Why separate rather than extending `train()`: pretraining's loop is built around
`PackedDataset`/`ShardedPackedDataset` (memory-mapped, fixed-length, uint16-packed
shards sized for a multi-billion-token corpus that can't fit in RAM) and always
calls `model(x, labels=x)` -- every position contributes to the loss. SFT's data
is a few thousand short, padded, PER-EXAMPLE sequences (fits in RAM uncompressed)
with masked labels (`labels != x`, loss only on the response span -- see
food_slm.data.sft_pack). Retrofitting one loop to cleanly cover both shapes would
cost more in awkward abstraction than this ~150-line sibling costs in duplication.
Shared pieces (lr_at, BatchSampler, save_checkpoint/load_checkpoint, the model
class itself) are reused as-is, not reimplemented.

Initializes from a PRETRAINED checkpoint's model weights only (fresh AdamW state,
fresh step counter) -- `load_checkpoint(..., optimizer=None, restore_rng=False)`
already supports exactly this call shape without any checkpoint.py changes.
Full-parameter fine-tuning (no LoRA/PEFT -- consistent with this project's
existing "v1 needs none of bitsandbytes/peft/trl" decision for a dense 150M model).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from food_slm.model import FoodSLM, ModelConfig
from food_slm.train.checkpoint import (
    BatchSampler,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from food_slm.train.loop import lr_at
from food_slm.utils import load_yaml, set_seed


@dataclass
class SFTTrainConfig:
    model_config: str
    pack_dir: str                 # dir with train.npz / val.npz / manifest.json
    checkpoint_dir: str
    init_from: str                # pretrained checkpoint path (model weights only)
    run_name: str = "sft-run"
    seed: int = 1337
    lr: float = 2.0e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    micro_batch_size: int = 16
    grad_accum_steps: int = 2
    epochs: float = 3.0
    checkpoint_every_steps: int = 50
    eval_every_steps: int = 25
    log_every_steps: int = 10
    keep_last_n: int = 3
    metrics_jsonl: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTTrainConfig:
        d = load_yaml(path)
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "betas" in keep:
            keep["betas"] = tuple(keep["betas"])
        return cls(**keep)


class SFTDataset:
    """Loads a train.npz/val.npz written by food_slm.data.sft_pack.build_sft_pack.
    `.batch(idx)` returns (input_ids, labels) -- the one place this interface
    differs from PackedDataset's single-array `.batch(idx)`, since SFT needs
    per-position labels that aren't just the input shifted."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        if len(self.input_ids) == 0:
            raise ValueError(f"{path} has zero examples")

    def __len__(self) -> int:
        return len(self.input_ids)

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Stored as uint16 (input_ids) / int32 (labels, needs signed for -100);
        # cast to int64 here, matching food_slm.train.packing.PackedDataset's
        # same on-disk-compact / cast-at-read-time convention. nn.Embedding and
        # F.cross_entropy's target both require int64 (confirmed live: F.cross_entropy
        # raises "expected target dtype to be Long" on a plain int32 array).
        return self.input_ids[idx].astype(np.int64), self.labels[idx].astype(np.int64)


@dataclass
class SFTTrainResult:
    final_step: int
    final_loss: float
    best_val: float | None
    history: list[dict] = field(default_factory=list)
    eval_history: list[dict] = field(default_factory=list)


def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device, non_blocking=True)


def _masked_eval_loss(model, ds: SFTDataset, *, batch_size: int, device: torch.device,
                      use_amp: bool, max_batches: int = 20) -> float:
    model.eval()
    n = min(len(ds), batch_size * max_batches)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            x_np, y_np = ds.batch(idx)
            x, y = _to_device(x_np, device), _to_device(y_np, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x, labels=y)
            n_real = (y[:, 1:] != -100).sum().item()
            total_loss += out.ce_loss.item() * max(1, n_real)
            total_tokens += n_real
    model.train()
    return total_loss / max(1, total_tokens)


def train_sft(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    device: str | None = None,
    resume: str | None = None,
    overrides: dict | None = None,
) -> SFTTrainResult:
    root = Path(repo_root) if repo_root else Path(config_path).resolve().parents[1]
    cfg = SFTTrainConfig.from_yaml(config_path)
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown SFT train override {k!r}")
        setattr(cfg, k, v)
    set_seed(cfg.seed)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    use_amp = device.type == "cuda"

    model_cfg = ModelConfig.from_yaml(root / cfg.model_config)
    model = FoodSLM(model_cfg).to(device)
    model.train()

    pack_dir = root / cfg.pack_dir
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    train_ds = SFTDataset(pack_dir / "train.npz")
    val_ds = SFTDataset(pack_dir / "val.npz") if (pack_dir / "val.npz").exists() else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)

    ckpt_dir = root / cfg.checkpoint_dir
    start_step, best_val = 0, None
    if resume:
        rp = Path(resume)
        if resume == "auto":
            cands = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            rp = cands[-1] if cands else None
        if rp and Path(rp).exists():
            info = load_checkpoint(rp, model=model, optimizer=opt, map_location=device)
            start_step, best_val = info["step"], info["best_val"]
            print(f"resumed SFT run from {rp} at step {start_step}")
    else:
        # Fresh SFT run: load ONLY the pretrained model weights -- fresh AdamW
        # state, fresh step counter. optimizer=None / restore_rng=False are
        # exactly what load_checkpoint already supports for this.
        init_path = root / cfg.init_from
        load_checkpoint(init_path, model=model, optimizer=None, map_location=device,
                        restore_rng=False)
        print(f"initialized from pretrained checkpoint {init_path} (weights only)")

    steps_per_epoch = max(1, len(train_ds) // (cfg.micro_batch_size * cfg.grad_accum_steps))
    max_steps = max(1, int(steps_per_epoch * cfg.epochs))
    print(f"SFT pack: {len(train_ds)} train / {len(val_ds) if val_ds else 0} val examples "
         f"(max_seq_len={manifest['max_seq_len']}) -> {steps_per_epoch} steps/epoch, "
         f"{max_steps} max_steps ({cfg.epochs} epochs)")

    if start_step >= max_steps:
        print(f"nothing to do: start_step {start_step} >= max_steps {max_steps}")
        return SFTTrainResult(final_step=start_step, final_loss=0.0, best_val=best_val)

    warmup = max(1, int(max_steps * cfg.warmup_ratio))
    sampler = BatchSampler(len(train_ds), cfg.micro_batch_size, cfg.seed,
                           start_batch=start_step * cfg.grad_accum_steps)
    it = iter(sampler)

    metrics_fh = None
    if cfg.metrics_jsonl:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_fh = (ckpt_dir / "metrics.jsonl").open("a", encoding="utf-8")

    res = SFTTrainResult(final_step=start_step, final_loss=float("nan"), best_val=best_val)
    t0 = time.time()
    tokens_seen = 0
    for step in range(start_step, max_steps):
        lr = lr_at(step, peak=cfg.lr, warmup=warmup, total=max_steps, min_ratio=cfg.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_sum = ce_sum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x_np, y_np = train_ds.batch(next(it))
            x, y = _to_device(x_np, device), _to_device(y_np, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x, labels=y)
            (out.loss / cfg.grad_accum_steps).backward()
            loss_sum += out.loss.item() / cfg.grad_accum_steps
            ce_sum += out.ce_loss.item() / cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        tokens_seen += cfg.grad_accum_steps * cfg.micro_batch_size * manifest["max_seq_len"]

        rec = {"step": step, "lr": round(lr, 8), "loss": loss_sum, "ce_loss": ce_sum,
              "grad_norm": grad_norm, "tokens_seen": tokens_seen, "t": round(time.time(), 3)}
        res.history.append(rec)
        res.final_step, res.final_loss = step + 1, loss_sum
        if metrics_fh:
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()

        if step % cfg.log_every_steps == 0:
            tok_s = tokens_seen / max(1e-6, time.time() - t0)
            print(f"step {step:>5}  loss {loss_sum:.4f}  ce {ce_sum:.4f}  "
                 f"lr {lr:.2e}  gnorm {grad_norm:.2f}  {tok_s:,.0f} tok/s")

        do_eval = val_ds is not None and (step + 1) % cfg.eval_every_steps == 0
        if do_eval:
            vl = _masked_eval_loss(model, val_ds, batch_size=cfg.micro_batch_size,
                                   device=device, use_amp=use_amp)
            ev = {"step": step + 1, "val_loss": vl, "t": round(time.time(), 3)}
            res.eval_history.append(ev)
            print(f"  [eval] step {step + 1}  val_loss {vl:.4f}")
            if metrics_fh:
                metrics_fh.write(json.dumps({"eval": ev}) + "\n")
                metrics_fh.flush()
            if best_val is None or vl < best_val:
                best_val = res.best_val = vl
                save_checkpoint(ckpt_dir / "best.pt", model=model, optimizer=opt,
                               step=step + 1, best_val=best_val, config=cfg.__dict__)

        if (step + 1) % cfg.checkpoint_every_steps == 0:
            save_checkpoint(ckpt_dir / f"step_{step + 1}.pt", model=model, optimizer=opt,
                           step=step + 1, best_val=best_val, config=cfg.__dict__)
            prune_checkpoints(ckpt_dir, cfg.keep_last_n)

    save_checkpoint(ckpt_dir / f"step_{res.final_step}.pt", model=model, optimizer=opt,
                   step=res.final_step, best_val=best_val, config=cfg.__dict__)
    if metrics_fh:
        metrics_fh.close()
    res.best_val = best_val
    return res
