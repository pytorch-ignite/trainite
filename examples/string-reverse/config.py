from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TransformerModelConfig(BaseModel):
    vocab_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    max_seq_len: int = 128


class StringReverseDatasetConfig(BaseModel):
    vocab_size: int = 32
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    seq_len: int = 16
    num_workers: int = 0
    seed: int = 7


class PreTrainerConfig(BaseModel):
    device: str = "cpu"
    type: str = "pre"
    epochs: int = 3
    learning_rate: float = 1e-3
    log_every_steps: int = 10
    grad_clip_norm: float | None = None


class OutputConfig(BaseModel):
    root: str = "output"
    run_name: str = "transformer__string-reverse"


class ProjectConfig(BaseModel):
    model_name: str = "transformer"
    dataset_name: str = "string-reverse"
    trainer_name: str = "pretrainer"
    model: TransformerModelConfig = Field(default_factory=TransformerModelConfig)
    dataset: StringReverseDatasetConfig = Field(
        default_factory=StringReverseDatasetConfig
    )
    trainer: PreTrainerConfig = Field(default_factory=PreTrainerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 42


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def dump_yaml(data: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def load_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml(path))


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    dump_yaml(config.model_dump(), path)


def default_config() -> ProjectConfig:
    return ProjectConfig()
