"""Configuration loading and override helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProjectConfig:
    root_dir: str = "."
    data_dir: str = "data"
    roi_dir: str = "data/rois"
    checkpoint_dir: str = "checkpoints/finetuned"
    pretrained_checkpoint: str = "checkpoints/pretrained/model_vsr_lrs3vox_base_s3fd.pth"
    sample_submission: str = "sample_submission.csv"
    lm_dir: str = "lm"
    lm_path: str = "lm/librispeech_4gram.bin"
    run_dir: str = "runs"


@dataclass
class DataConfig:
    train_split: str = "train"
    test_split: str = "test"
    target_size: int = 96
    max_frames: int = 400
    min_frames: int = 2
    val_fraction: float = 0.05
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    train_flip_prob: float = 0.5
    temporal_mask_prob: float = 0.8
    spatial_cutout_prob: float = 0.5
    speed_perturb_prob: float = 0.5


@dataclass
class ModelConfig:
    d_model: int = 256
    encoder_layers: int = 18
    decoder_layers: int = 6
    encoder_heads: int = 4
    decoder_heads: int = 4
    ff_expand: int = 4
    decoder_ff_dim: int = 2048
    conv_kernel: int = 31
    dropout: float = 0.1
    inter_ctc_every: int = 4
    gradient_checkpointing: bool = False


@dataclass
class LossConfig:
    ctc_weight: float = 0.3
    inter_ctc_weight: float = 0.1
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 25
    batch_per_gpu: int = 16
    validation_batch_size: int = 4
    validation_max_batches: int = 50
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    lr_frontend: float = 1.0e-5
    lr_encoder: float = 1.0e-4
    lr_decoder: float = 3.0e-4
    layer_decay: float = 0.8
    min_lr_factor: float = 0.05
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1.0e-9
    save_every: int = 5
    log_every: int = 50
    use_amp: bool = True
    backend: str = "auto"
    resume_checkpoint: str | None = None


@dataclass
class InferenceConfig:
    alpha: float = 0.6
    beta: float = 0.4
    beam_width: int = 80
    batch_size: int = 8
    decode_workers: int = 4
    device: str = "cuda"
    use_tta: bool = False
    save_failures_path: str = "runs/inference_failures.json"
    live_preprocess_fallback: bool = True


@dataclass
class KaggleConfig:
    competition: str = ""


@dataclass
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    inference: InferenceConfig
    kaggle: KaggleConfig

    @property
    def root_path(self) -> Path:
        return resolve_path(self.project.root_dir)

    def resolve(self, raw_path: str | Path) -> Path:
        return resolve_path(raw_path, base_dir=self.root_path)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _instantiate(section_cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    allowed = {field.name for field in fields(section_cls)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return section_cls(**filtered)


def resolve_path(raw_path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base = base_dir or PROJECT_ROOT
    return (base / path).resolve()


def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(),
        data=DataConfig(),
        model=ModelConfig(),
        loss=LossConfig(),
        training=TrainingConfig(),
        inference=InferenceConfig(),
        kaggle=KaggleConfig(),
    )


def _deep_merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_override(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def apply_overrides(config_dict: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = dict(config_dict)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected dotted.path=value.")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        cursor = merged
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = _coerce_override(raw_value)
    return merged


def load_config(config_path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    config = default_config()
    config_dict = config.as_dict()

    if config_path is not None:
        path = resolve_path(config_path)
        loaded = yaml.safe_load(path.read_text()) or {}
        config_dict = _deep_merge_dict(config_dict, loaded)

    config_dict = apply_overrides(config_dict, overrides)
    return ExperimentConfig(
        project=_instantiate(ProjectConfig, config_dict.get("project")),
        data=_instantiate(DataConfig, config_dict.get("data")),
        model=_instantiate(ModelConfig, config_dict.get("model")),
        loss=_instantiate(LossConfig, config_dict.get("loss")),
        training=_instantiate(TrainingConfig, config_dict.get("training")),
        inference=_instantiate(InferenceConfig, config_dict.get("inference")),
        kaggle=_instantiate(KaggleConfig, config_dict.get("kaggle")),
    )
