"""Inference pipeline and submission generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from omni_vsr.config import ExperimentConfig
from omni_vsr.inference.ctc_decode import build_beam_decoder, decode_batch_ctc
from omni_vsr.models import VSRModel, build_model_from_config
from omni_vsr.preprocessing import LipROIExtractor
from omni_vsr.utils.checkpoints import load_model_state
from omni_vsr.utils.runtime import dump_json


def resolve_video_path(raw_path: str, data_dir: Path) -> Path:
    candidate = Path(raw_path)
    candidates = [candidate, data_dir / candidate]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[-1].resolve()


def resolve_roi_path(raw_path: str, data_dir: Path, roi_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(data_dir.resolve())
        except ValueError:
            relative = Path(candidate.name)
    else:
        relative = candidate
    return (roi_dir / relative).with_suffix(".npy")


def predict_log_probs(
    model: VSRModel,
    videos: torch.Tensor,
    video_lengths: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        _, _, ctc_logits = model.encode(videos, video_lengths=video_lengths)
        log_probs = torch.log_softmax(ctc_logits, dim=-1)
        if not use_tta:
            return log_probs
        flipped = torch.flip(videos, dims=[-1])
        _, _, flipped_logits = model.encode(flipped, video_lengths=video_lengths)
        flipped_log_probs = torch.log_softmax(flipped_logits, dim=-1)
        return 0.5 * (log_probs + flipped_log_probs)


def _load_frames_for_path(
    raw_path: str,
    data_dir: Path,
    roi_dir: Path,
    extractor: LipROIExtractor | None,
    save_cache: bool,
) -> tuple[np.ndarray | None, str | None]:
    roi_path = resolve_roi_path(raw_path, data_dir=data_dir, roi_dir=roi_dir)
    if roi_path.exists():
        return np.load(str(roi_path)).astype(np.float32), None

    if extractor is None:
        return None, "ROI cache miss and live preprocessing disabled"

    video_path = resolve_video_path(raw_path, data_dir=data_dir)
    if not video_path.exists():
        return None, f"Video not found: {video_path}"

    try:
        frames = extractor.extract(video_path).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        return None, str(exc)

    if save_cache:
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(roi_path), frames.astype(np.float16))
    return frames, None


def run_inference(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    sample_csv: str | Path | None = None,
    device: str | None = None,
    beam_width: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    batch_size: int | None = None,
    use_tta: bool | None = None,
) -> pd.DataFrame:
    inference_device = device or config.inference.device
    resolved_device = torch.device(
        inference_device if inference_device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = build_model_from_config(config).to(resolved_device)
    metadata = load_model_state(model, checkpoint_path, map_location=resolved_device, strict=False)
    model.eval()

    sample_csv_path = config.resolve(sample_csv or config.project.sample_submission)
    output_csv_path = config.resolve(output_csv)
    data_dir = config.resolve(config.project.data_dir)
    roi_dir = config.resolve(config.project.roi_dir)
    lm_path = config.resolve(config.project.lm_path)

    frame_batch_size = batch_size or config.inference.batch_size
    beam_width = beam_width if beam_width is not None else config.inference.beam_width
    alpha = alpha if alpha is not None else config.inference.alpha
    beta = beta if beta is not None else config.inference.beta
    use_tta = config.inference.use_tta if use_tta is None else use_tta
    decoder = build_beam_decoder(lm_path=lm_path, alpha=alpha, beta=beta)

    dataframe = pd.read_csv(sample_csv_path)
    path_column = "path" if "path" in dataframe.columns else dataframe.columns[0]
    entries = dataframe[path_column].tolist()

    extractor = LipROIExtractor(target_size=config.data.target_size) if config.inference.live_preprocess_fallback else None
    predictions: list[str] = []
    failures: dict[str, str] = {}

    try:
        for start in tqdm(range(0, len(entries), frame_batch_size), desc="Inference", unit="batch"):
            batch_entries = entries[start : start + frame_batch_size]
            frames_list: list[torch.Tensor] = []
            valid_flags: list[bool] = []

            for raw_path in batch_entries:
                frames, failure_reason = _load_frames_for_path(
                    raw_path=raw_path,
                    data_dir=data_dir,
                    roi_dir=roi_dir,
                    extractor=extractor,
                    save_cache=True,
                )
                if frames is None:
                    valid_flags.append(False)
                    frames_list.append(torch.empty(0))
                    failures[raw_path] = failure_reason or "Unknown failure"
                    continue

                valid_flags.append(True)
                frames_list.append(torch.from_numpy(frames).unsqueeze(1))

            valid_frames = [frames for frames, is_valid in zip(frames_list, valid_flags) if is_valid]
            if not valid_frames:
                predictions.extend(["" for _ in batch_entries])
                continue

            max_frames = max(frame.shape[0] for frame in valid_frames)
            padded = torch.zeros(len(valid_frames), max_frames, 1, config.data.target_size, config.data.target_size)
            video_lengths = torch.tensor([frame.shape[0] for frame in valid_frames], dtype=torch.long)
            for index, frame in enumerate(valid_frames):
                padded[index, : frame.shape[0]] = frame

            padded = padded.to(resolved_device)
            video_lengths = video_lengths.to(resolved_device)
            log_probs = predict_log_probs(model, padded, video_lengths, use_tta=use_tta)
            log_probs_np = [
                log_probs[index, : int(video_lengths[index].item())].cpu().numpy()
                for index in range(log_probs.shape[0])
            ]
            decoded = decode_batch_ctc(
                log_probs_batch=log_probs_np,
                decoder=decoder,
                beam_width=beam_width,
                decode_workers=config.inference.decode_workers,
            )

            decoded_index = 0
            for is_valid in valid_flags:
                if is_valid:
                    predictions.append(decoded[decoded_index])
                    decoded_index += 1
                else:
                    predictions.append("")
    finally:
        if extractor is not None:
            extractor.close()

    dataframe["transcription"] = predictions
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)

    if failures:
        dump_json(
            {
                "checkpoint": str(checkpoint_path),
                "missing_keys": metadata["missing_keys"],
                "unexpected_keys": metadata["unexpected_keys"],
                "failures": failures,
            },
            config.resolve(config.inference.save_failures_path),
        )

    return dataframe
