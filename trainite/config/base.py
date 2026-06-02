from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class OutputConfig(BaseModel):
    root: str
    run_name: str


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target: str = Field(alias="_target_")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_every_steps: int = 10
    epochs: int = 10


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
    # Option 1: Explicit splits
    train: SplitConfig | None = None
    val: SplitConfig | None = None
    test: SplitConfig | None = None

    # Option 2: Automatic splitting
    dataset: ComponentConfig | None = None
    dataloader: DataLoaderConfig | None = None
    train_ratio: float | None = None
    val_ratio: float | None = None

    @model_validator(mode="after")
    def validate_options(self) -> "DataConfig":
        option1_fields = {"train", "val", "test"}
        option2_fields = {"dataset", "train_ratio", "val_ratio", "dataloader"}

        present_option1 = {f for f in option1_fields if getattr(self, f) is not None}
        present_option2 = {f for f in option2_fields if getattr(self, f) is not None}

        if present_option1 and present_option2:
            if "dataset" in present_option2 or "dataloader" in present_option2:
                raise ValueError(
                    r"Cannot provide train/val/test levels when 'dataset' or 'dataloader' is provided at the data level"
                )
            raise ValueError(
                "Cannot provide train_ratio or val_ratio at the data level when explicit splits are used"
            )

        if not present_option1 and not present_option2:
            raise ValueError(
                "Must provide either explicit splits (train) or automatic splitting (dataset)"
            )

        if present_option1 and "train" not in present_option1:
            raise ValueError(
                "Explicit splits mode (Option 1) requires at least the 'train' split"
            )

        if present_option2 and "dataset" not in present_option2:
            raise ValueError(
                "Automatic splitting mode (Option 2) requires the 'dataset' field"
            )

        if self.dataset is not None:
            train_ratio = self.train_ratio if self.train_ratio is not None else 1.0
            val_ratio = self.val_ratio if self.val_ratio is not None else 0.0

            if train_ratio <= 0.0 or val_ratio < 0:
                raise ValueError(
                    f"train_ratio must be between 0 and 1. Got train_ratio={train_ratio}, val_ratio={val_ratio}"
                )

            if train_ratio + val_ratio > 1.0:
                raise ValueError(
                    f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) exceeds 1.0"
                )

        return self


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
