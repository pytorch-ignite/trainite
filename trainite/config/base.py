from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class OutputConfig(BaseModel):
    root: str
    run_name: str


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target: str = Field(alias="_target_")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_every_steps: int = 10


class OptimizerConfig(ComponentConfig):
    target: str = Field(alias="_target_", default="torch.optim.AdamW")
    lr: float = 1e-3


class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 2
    collate_fn: ComponentConfig | None = None


class SplitConfig(BaseModel):
    dataset: ComponentConfig
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class DataConfig(BaseModel):
    train: SplitConfig
    val: SplitConfig | None = None
    test: SplitConfig | None = None


class ProjectConfig(BaseModel):
    model: ComponentConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig
    trainer: TrainerConfig
    output: OutputConfig
    seed: int = 42
    device: str = "auto"


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def dump_yaml(data: dict["str", Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    data = config.model_dump(by_alias=True, polymorphic_serialization=True)
    dump_yaml(data, path)


def load_config(path: str | Path) -> ProjectConfig:
    raw_conf = OmegaConf.load(path)
    return ProjectConfig.model_validate(raw_conf)
