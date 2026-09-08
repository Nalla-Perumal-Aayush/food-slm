"""Training loop (baseline §7 / §12 / plan_40m §F–G).

Causal LM, bf16 autocast on CUDA, AdamW, linear warmup + cosine decay to
``min_lr_ratio``, gradient accumulation, global grad-norm clip, z-loss (in the
model). Periodic eval on the val shard(s) + periodic checkpoints; resume is bit-exact
(deterministic :class:`BatchSampler` + restored RNG state). Optional 1K->2K->4K
context curriculum and per-source validation for multi-source sharded corpora.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from food_slm.model import FoodSLM, ModelConfig
from food_slm.train.checkpoint import (
    BatchSampler,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from food_slm.train.curriculum import parse_curriculum, phase_boundaries, seq_len_at
from food_slm.train.packing import PackedDataset
from food_slm.utils import load_yaml, set_seed


def _open_dataset(shard_dir: Path, split: str):
    """PackedDataset for a single-file pack, ShardedPackedDataset for a multi-shard pack."""
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    if "shards" in manifest:
        from food_slm.train.shards import ShardedPackedDataset

        return ShardedPackedDataset(shard_dir, split)
    return PackedDataset(shard_dir, split)


@dataclass
class TrainConfig:
    model_config: str
    shard_dir: str
    checkpoint_dir: str
    tokenizer: str = "artifacts/tokenizers/poc-16k"
    run_name: str = "run"
    seed: int = 1337
    sequence_length: int = 1024
    lr: float = 6.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 2000
    gradient_checkpointing: bool = False
    checkpoint_every_steps: int = 250
    eval_every_steps: int = 250
    eval_batches: int = 20
    log_every_steps: int = 10
    keep_last_n: int = 3
    tracker: str = "none"
    wandb_project: str = "food-slm-poc"
    curriculum: list | None = None       # [[seq_len, until_frac], ...]; None = full length
    metrics_jsonl: bool = True           # write <checkpoint_dir>/metrics.jsonl
    spike_factor: float = 1.6            # log a WARN if loss > spike_factor * trailing mean

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        d = load_yaml(path)
        # tolerate scaffold keys we do not use
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "betas" in keep:
            keep["betas"] = tuple(keep["betas"])
        if "shard_dir" not in keep and "train_shards" in d:
            keep["shard_dir"] = str(Path(d["train_shards"]).parent)
        return cls(**keep)


def lr_at(step: int, *, peak: float, warmup: int, total: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * coeff)


@dataclass
class TrainResult:
    final_step: int
    final_loss: float
    best_val: float | None
    history: list[dict] = field(default_factory=list)
    eval_history: list[dict] = field(default_factory=list)


def _batch(ds, idx: np.ndarray, device: torch.device, seq_len: int | None = None) -> torch.Tensor:
    arr = ds.batch(idx)
    if seq_len is not None and seq_len < arr.shape[1]:
        arr = arr[:, :seq_len]
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device, non_blocking=True)


def _init_tracker(cfg: TrainConfig, model_cfg: ModelConfig):
    if cfg.tracker != "wandb":
        return None
    try:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   config={**cfg.__dict__, **model_cfg.model_dump()})
        return wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[tracker] wandb unavailable ({exc}); continuing without it")
        return None


def train(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    stop_after: int | None = None,
    resume: str | None = None,
    repo_root: str | Path | None = None,
    deterministic: bool = False,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    overrides: dict | None = None,
) -> TrainResult:
    """Train from *config_path*. ``stop_after`` breaks the loop after that optimizer step
    (saving a checkpoint) without changing the LR schedule — used for resume testing.
    ``deterministic`` pins threads + deterministic kernels so resume is bit-exact (slower)."""
    from food_slm.eval.perplexity import evaluate_loss  # lazy: avoids eval<->train import cycle

    root = Path(repo_root) if repo_root else Path(config_path).resolve().parents[1]
    cfg = TrainConfig.from_yaml(config_path)
    if max_steps is not None:
        cfg.max_steps = max_steps
    if checkpoint_dir is not None:
        cfg.checkpoint_dir = str(checkpoint_dir)
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown train override {k!r}")
        setattr(cfg, k, v)
    if deterministic:
        torch.set_num_threads(1)
    set_seed(cfg.seed, deterministic=deterministic)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    use_amp = device.type == "cuda"

    model_cfg = ModelConfig.from_yaml(root / cfg.model_config)
    model = FoodSLM(model_cfg).to(device)
    model.grad_checkpoint = cfg.gradient_checkpointing
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)

    shard_dir = root / cfg.shard_dir
    train_ds = _open_dataset(shard_dir, "train")
    try:
        val_ds = _open_dataset(shard_dir, "val")
    except FileNotFoundError:
        val_ds = None
    if train_ds.seq_len != cfg.sequence_length:
        raise ValueError(f"packed seq_len {train_ds.seq_len} != config {cfg.sequence_length}")

    # per-source val datasets (multi-source sharded packs only)
    val_by_source: dict[str, object] = {}
    if val_ds is not None and "val_files" in getattr(train_ds, "manifest", {}):
        for name in train_ds.manifest["val_files"]:
            with contextlib.suppress(FileNotFoundError, KeyError):
                val_by_source[name] = _open_dataset(shard_dir, f"val:{name}")

    phases = parse_curriculum(cfg.curriculum, cfg.sequence_length)
    if cfg.curriculum:
        bounds = phase_boundaries(cfg.max_steps, phases)
        print("curriculum:", [(seq_len_at(b, cfg.max_steps, phases), b) for b in bounds])

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
            print(f"resumed from {rp} at step {start_step}")
        elif resume != "auto":
            raise FileNotFoundError(f"resume checkpoint not found: {resume}")

    if start_step >= cfg.max_steps:
        print(f"nothing to do: start_step {start_step} >= max_steps {cfg.max_steps}")
        return TrainResult(final_step=start_step, final_loss=0.0, best_val=best_val)

    warmup = max(1, int(cfg.max_steps * cfg.warmup_ratio))
    sampler = BatchSampler(len(train_ds), cfg.micro_batch_size, cfg.seed,
                           start_batch=start_step * cfg.grad_accum_steps)
    it = iter(sampler)
    tracker = _init_tracker(cfg, model_cfg)

    metrics_fh = None
    if cfg.metrics_jsonl:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_fh = (ckpt_dir / "metrics.jsonl").open("a", encoding="utf-8")

    res = TrainResult(final_step=start_step, final_loss=float("nan"), best_val=best_val)
    t0 = time.time()
    tokens_seen = 0
    trailing: list[float] = []
    cur_seq = seq_len_at(start_step, cfg.max_steps, phases)
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, peak=cfg.lr, warmup=warmup, total=cfg.max_steps,
                   min_ratio=cfg.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        seq_len = seq_len_at(step, cfg.max_steps, phases)
        if seq_len != cur_seq:
            print(f"  [curriculum] step {step}  seq_len {cur_seq} -> {seq_len}")
            cur_seq = seq_len

        opt.zero_grad(set_to_none=True)
        loss_sum = ce_sum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x = _batch(train_ds, next(it), device, seq_len)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x, labels=x)
            (out.loss / cfg.grad_accum_steps).backward()
            loss_sum += out.loss.item() / cfg.grad_accum_steps
            ce_sum += out.ce_loss.item() / cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()
        tokens_seen += cfg.grad_accum_steps * cfg.micro_batch_size * seq_len

        spike = bool(trailing) and loss_sum > cfg.spike_factor * (sum(trailing) / len(trailing))
        trailing = [*trailing[-49:], loss_sum]

        rec = {"step": step, "lr": round(lr, 8), "loss": loss_sum, "ce_loss": ce_sum,
               "grad_norm": grad_norm, "seq_len": seq_len, "tokens_seen": tokens_seen}
        res.history.append(rec)
        res.final_step, res.final_loss = step + 1, loss_sum
        if metrics_fh:
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()
        if spike:
            print(f"  [WARN] loss spike at step {step}: {loss_sum:.4f} "
                  f"(trailing mean {sum(trailing) / len(trailing):.4f}, gnorm {grad_norm:.2f})")

        if step % cfg.log_every_steps == 0:
            tok_s = tokens_seen / max(1e-6, time.time() - t0)
            print(f"step {step:>5}  loss {loss_sum:.4f}  ce {ce_sum:.4f}  "
                  f"lr {lr:.2e}  gnorm {grad_norm:.2f}  L{seq_len}  {tok_s:,.0f} tok/s")
        if tracker:
            tracker.log(rec, step=step)

        do_eval = val_ds is not None and (step + 1) % cfg.eval_every_steps == 0
        if do_eval:
            eval_seq = min(phases[0].seq_len, cfg.sequence_length)  # fixed -> comparable curve
            vl = evaluate_loss(model, val_ds, batch_size=cfg.micro_batch_size,
                               n_batches=cfg.eval_batches, device=device, use_amp=use_amp,
                               seq_len=eval_seq)
            per_src = {n: evaluate_loss(model, ds, batch_size=cfg.micro_batch_size,
                                        n_batches=cfg.eval_batches, device=device,
                                        use_amp=use_amp, seq_len=eval_seq)
                       for n, ds in val_by_source.items()}
            ev = {"step": step + 1, "val_loss": vl, **{f"val_{n}": v for n, v in per_src.items()}}
            res.eval_history.append(ev)
            extra = "  ".join(f"{n} {v:.3f}" for n, v in per_src.items())
            print(f"  [eval] step {step + 1}  val_loss {vl:.4f}"
                  + (f"  ({extra})" if extra else ""))
            if metrics_fh:
                metrics_fh.write(json.dumps({"eval": ev}) + "\n")
                metrics_fh.flush()
            if tracker:
                tracker.log({"val_loss": vl}, step=step)
            if best_val is None or vl < best_val:
                best_val = res.best_val = vl
                save_checkpoint(ckpt_dir / "best.pt", model=model, optimizer=opt,
                                step=step + 1, best_val=best_val, config=cfg.__dict__)

        hit_stop = stop_after is not None and step + 1 >= stop_after
        if hit_stop or (step + 1) % cfg.checkpoint_every_steps == 0:
            save_checkpoint(ckpt_dir / f"step_{step + 1}.pt", model=model, optimizer=opt,
                            step=step + 1, best_val=best_val, config=cfg.__dict__)
            prune_checkpoints(ckpt_dir, cfg.keep_last_n)
        if hit_stop:
            if metrics_fh:
                metrics_fh.close()
            if tracker:
                tracker.finish()
            return res

    save_checkpoint(ckpt_dir / f"step_{res.final_step}.pt", model=model, optimizer=opt,
                    step=res.final_step, best_val=best_val, config=cfg.__dict__)
    if metrics_fh:
        metrics_fh.close()
    if tracker:
        tracker.finish()
    res.best_val = best_val
    return res
