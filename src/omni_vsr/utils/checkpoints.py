"""Checkpoint load/save helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "model_state."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {_strip_prefix(k): v for k, v in value.items() if torch.is_tensor(v)}
        if all(isinstance(key, str) for key in checkpoint.keys()):
            state_dict = {_strip_prefix(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}
            if state_dict:
                return state_dict
    raise ValueError("Unsupported checkpoint format.")


def load_model_state(model: torch.nn.Module, checkpoint_path: str | Path, map_location: str | torch.device = "cpu", strict: bool = False) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return {
        "checkpoint": checkpoint,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "metadata": metadata,
    }


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_val_wer: float,
    config: dict[str, Any],
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_wer": best_val_wer,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "cfg": config,
    }
    torch.save(payload, path)
