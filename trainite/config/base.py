from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from trainite.config.dataset import StringReverseDatasetConfig
from trainite.config.model import TransformerModelConfig
from trainite.config.trainer import PreTrainerConfig


class OutputConfig(BaseModel):
    root: str = "output"
    run_name: str = "dummy-pretrain"


class ProjectConfig(BaseModel):
    model: Any = Field(default_factory=TransformerModelConfig)
    dataset: Any = Field(default_factory=StringReverseDatasetConfig)
    trainer: Any = Field(default_factory=PreTrainerConfig)
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


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    dump_yaml(config.model_dump(by_alias=True), path)


def load_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml(path))


def default_config() -> ProjectConfig:
    return ProjectConfig()
