"""Training orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from omni_vsr.config import ExperimentConfig
from omni_vsr.data import LipReadingDataset, collate_lipreading_batch
from omni_vsr.models import VSRModel, build_model_from_config
from omni_vsr.tokenizer import TOKENIZER
from omni_vsr.training.losses import JointCTCAttentionLoss
from omni_vsr.utils.checkpoints import load_model_state, save_training_checkpoint
from omni_vsr.utils.distributed import DistributedContext, cleanup_distributed, setup_distributed
from omni_vsr.utils.metrics import corpus_word_error_rate
from omni_vsr.utils.runtime import seed_everything


def build_train_val_datasets(config: ExperimentConfig) -> tuple[Subset, Subset]:
    train_dataset = LipReadingDataset(
        roi_root=config.resolve(config.project.roi_dir),
        txt_root=config.resolve(config.project.data_dir),
        split=config.data.train_split,
        augment=True,
        max_frames=config.data.max_frames,
        min_frames=config.data.min_frames,
        flip_prob=config.data.train_flip_prob,
        temporal_mask_prob=config.data.temporal_mask_prob,
        spatial_cutout_prob=config.data.spatial_cutout_prob,
        speed_perturb_prob=config.data.speed_perturb_prob,
        require_transcripts=True,
    )
    val_dataset = LipReadingDataset(
        roi_root=config.resolve(config.project.roi_dir),
        txt_root=config.resolve(config.project.data_dir),
        split=config.data.train_split,
        augment=False,
        max_frames=config.data.max_frames,
        min_frames=config.data.min_frames,
        require_transcripts=True,
    )
    index_count = len(train_dataset)
    if index_count < 2:
        raise ValueError("Need at least 2 training samples to create train/validation splits.")
    val_count = max(1, int(index_count * config.data.val_fraction))
    train_count = index_count - val_count
    generator = torch.Generator().manual_seed(config.training.seed)
    permutation = torch.randperm(index_count, generator=generator).tolist()
    train_indices = permutation[:train_count]
    val_indices = permutation[train_count:]
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def build_loaders(config: ExperimentConfig, context: DistributedContext) -> tuple[DataLoader, DataLoader | None]:
    train_dataset, val_dataset = build_train_val_datasets(config)
    persistent_workers = config.data.persistent_workers and config.data.num_workers > 0

    if context.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_per_gpu,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            persistent_workers=persistent_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_lipreading_batch,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_per_gpu,
            shuffle=True,
            num_workers=config.data.num_workers,
            persistent_workers=persistent_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_lipreading_batch,
            drop_last=True,
        )

    if not context.is_main_process:
        return train_loader, None

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        num_workers=max(0, min(2, config.data.num_workers)),
        persistent_workers=False,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_lipreading_batch,
    )
    return train_loader, val_loader


def build_parameter_groups(model: VSRModel, config: ExperimentConfig) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    groups.append({"params": list(model.frontend.parameters()), "lr": config.training.lr_frontend})
    encoder_stem_lr = config.training.lr_encoder * (
        config.training.layer_decay ** max(1, len(model.encoder.layers))
    )
    groups.append({"params": list(model.proj.parameters()), "lr": encoder_stem_lr})
    groups.append({"params": list(model.encoder.input_norm.parameters()), "lr": encoder_stem_lr})

    layer_count = len(model.encoder.layers)
    for layer_index, layer in enumerate(model.encoder.layers):
        depth = layer_count - layer_index - 1
        layer_lr = config.training.lr_encoder * (config.training.layer_decay ** depth)
        groups.append({"params": list(layer.parameters()), "lr": layer_lr})

    groups.append({"params": list(model.ctc_head.parameters()), "lr": config.training.lr_decoder})
    groups.append({"params": list(model.inter_ctc_head.parameters()), "lr": config.training.lr_decoder})
    groups.append({"params": list(model.decoder.parameters()), "lr": config.training.lr_decoder})
    return groups


def apply_cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_factor: float,
) -> None:
    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def reduce_loss_dict(losses: dict[str, torch.Tensor], context: DistributedContext) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, tensor in losses.items():
        value = tensor.detach()
        if context.is_distributed:
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value = value / context.world_size
        reduced[name] = float(value.item())
    return reduced


def decode_reference_from_batch(batch: dict[str, Any]) -> list[str]:
    references: list[str] = []
    ctc_targets = batch["ctc_targets"]
    ctc_lengths = batch["ctc_target_lengths"]
    for index in range(ctc_targets.shape[0]):
        token_ids = ctc_targets[index, : int(ctc_lengths[index].item())].tolist()
        references.append(TOKENIZER.decode_sequence(token_ids))
    return references


@torch.no_grad()
def evaluate_greedy_wer(
    model: VSRModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        predictions = model.greedy_decode(batch["videos"], video_lengths=batch["video_lengths"])
        references.extend(decode_reference_from_batch(batch))
        hypotheses.extend(predictions)
    return corpus_word_error_rate(references, hypotheses)


def maybe_load_initial_weights(
    model: VSRModel,
    config: ExperimentConfig,
    device: torch.device,
    is_main_process: bool,
) -> None:
    checkpoint_path = config.resolve(config.project.pretrained_checkpoint)
    if not checkpoint_path.exists():
        return
    metadata = load_model_state(model, checkpoint_path, map_location=device, strict=False)
    if is_main_process:
        print(
            f"Loaded pretrained weights from {checkpoint_path} "
            f"(missing={len(metadata['missing_keys'])}, unexpected={len(metadata['unexpected_keys'])})"
        )


def maybe_resume_training(
    model: VSRModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: ExperimentConfig,
    device: torch.device,
    is_main_process: bool,
) -> tuple[int, int, float]:
    resume_path_raw = config.training.resume_checkpoint
    if not resume_path_raw:
        return 0, 0, float("inf")

    resume_path = config.resolve(resume_path_raw)
    checkpoint = torch.load(resume_path, map_location=device)
    metadata = load_model_state(model, resume_path, map_location=device, strict=False)
    optimizer_state = checkpoint.get("optimizer_state")
    scaler_state = checkpoint.get("scaler_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_val_wer = float(checkpoint.get("best_val_wer", float("inf")))
    if is_main_process:
        print(
            f"Resumed from {resume_path} "
            f"(epoch={start_epoch}, missing={len(metadata['missing_keys'])}, unexpected={len(metadata['unexpected_keys'])})"
        )
    return start_epoch, global_step, best_val_wer


def _append_epoch_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def train_model(config: ExperimentConfig) -> Path | None:
    context = setup_distributed(config.training.backend)
    seed_everything(config.training.seed + context.rank)
    try:
        model = build_model_from_config(config).to(context.device)
        maybe_load_initial_weights(model, config, context.device, context.is_main_process)

        parameter_groups = build_parameter_groups(model, config)
        for group in parameter_groups:
            group["initial_lr"] = group["lr"]

        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.training.weight_decay,
            betas=tuple(config.training.adam_betas),
            eps=config.training.adam_eps,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=(config.training.use_amp and context.device.type == "cuda")
        )
        criterion = JointCTCAttentionLoss(
            ctc_weight=config.loss.ctc_weight,
            inter_ctc_weight=config.loss.inter_ctc_weight,
            label_smoothing=config.loss.label_smoothing,
        )

        start_epoch, global_step, best_val_wer = maybe_resume_training(
            model,
            optimizer,
            scaler,
            config,
            context.device,
            context.is_main_process,
        )

        wrapped_model: torch.nn.Module
        if context.is_distributed:
            wrapped_model = DDP(model, device_ids=[context.local_rank], find_unused_parameters=False)
        else:
            wrapped_model = model

        train_loader, val_loader = build_loaders(config, context)
        total_steps = max(1, config.training.epochs * len(train_loader))
        checkpoint_dir = config.resolve(config.project.checkpoint_dir)
        metrics_log_path = config.resolve(config.project.run_dir) / "train_metrics.jsonl"

        if context.is_main_process:
            params = model.count_parameters()
            print(
                f"Training on {context.world_size} process(es) with effective batch size "
                f"{config.training.batch_per_gpu * context.world_size}"
            )
            print(
                f"Model params: {params['total_M']:.1f}M total, {params['trainable_M']:.1f}M trainable"
            )

        best_checkpoint_path: Path | None = None
        for epoch in range(start_epoch, config.training.epochs):
            if context.is_distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            wrapped_model.train()
            running = {"total": 0.0, "ctc": 0.0, "attn": 0.0, "inter": 0.0}
            batch_count = 0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:02d}",
                disable=not context.is_main_process,
            )
            for batch in progress:
                batch = move_batch_to_device(batch, context.device)
                apply_cosine_warmup_schedule(
                    optimizer,
                    step=global_step,
                    warmup_steps=config.training.warmup_steps,
                    total_steps=total_steps,
                    min_lr_factor=config.training.min_lr_factor,
                )

                optimizer.zero_grad(set_to_none=True)
                autocast_enabled = config.training.use_amp and context.device.type == "cuda"
                with torch.autocast(device_type=context.device.type, dtype=torch.float16, enabled=autocast_enabled):
                    outputs = wrapped_model(
                        batch["videos"],
                        batch["decoder_inputs"],
                        batch["video_lengths"],
                        batch["decoder_target_lengths"],
                    )
                    losses = criterion(
                        outputs,
                        batch["ctc_targets"],
                        batch["ctc_target_lengths"],
                        batch["decoder_targets"],
                        batch["video_lengths"],
                    )

                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                reduced = reduce_loss_dict(losses, context)
                for key in running:
                    running[key] += reduced[key]
                batch_count += 1
                global_step += 1

                if context.is_main_process and global_step % config.training.log_every == 0:
                    averages = {key: value / max(1, batch_count) for key, value in running.items()}
                    progress.set_postfix(
                        loss=f"{averages['total']:.3f}",
                        ctc=f"{averages['ctc']:.3f}",
                        attn=f"{averages['attn']:.3f}",
                    )

            if context.is_main_process and val_loader is not None:
                model.eval()
                avg_train_loss = running["total"] / max(1, batch_count)
                val_wer = evaluate_greedy_wer(
                    model,
                    val_loader,
                    context.device,
                    max_batches=config.training.validation_max_batches,
                )
                print(f"Epoch {epoch + 1:02d} | train_loss={avg_train_loss:.4f} | val_wer={val_wer:.4f}")

                epoch_log = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": avg_train_loss,
                    "val_wer": val_wer,
                }
                _append_epoch_log(metrics_log_path, epoch_log)

                should_save_epoch = (epoch + 1) % config.training.save_every == 0
                improved = val_wer < best_val_wer
                if improved:
                    best_val_wer = val_wer
                if should_save_epoch or improved:
                    epoch_checkpoint = checkpoint_dir / f"epoch_{epoch + 1:02d}_wer{val_wer:.3f}.pt"
                    save_training_checkpoint(
                        checkpoint_path=epoch_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_val_wer=best_val_wer,
                        config=config.as_dict(),
                    )
                    if improved:
                        best_checkpoint_path = checkpoint_dir / "best.pt"
                        save_training_checkpoint(
                            checkpoint_path=best_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            epoch=epoch + 1,
                            global_step=global_step,
                            best_val_wer=best_val_wer,
                            config=config.as_dict(),
                        )
                        print(f"Saved improved checkpoint to {best_checkpoint_path}")

            if context.is_distributed:
                dist.barrier()

        return best_checkpoint_path
    finally:
        cleanup_distributed(context)
