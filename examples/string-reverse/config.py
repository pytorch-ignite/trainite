from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


ALPHABET_PRESETS = {
    "@alpha": "abcdefghijklmnopqrstuvwxyz",
    "@digits": "0123456789",
    "@alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
}


def resolve_alphabet(alphabet: str) -> str:
    return ALPHABET_PRESETS.get(alphabet, alphabet)


def _component_value(component: BaseModel, name: str) -> Any:
    value = getattr(component, name, None)
    if value is not None:
        return value
    return component.model_extra.get(name) if component.model_extra else None


class OutputConfig(BaseModel):
    root: str
    run_name: str


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target: str = Field(alias="_target_")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-3
    log_every_steps: int = 10


class ProjectConfig(BaseModel):
    model: ComponentConfig
    dataset: ComponentConfig
    trainer: TrainerConfig
    output: OutputConfig
    seed: int = 42

    @model_validator(mode="after")
    def validate_vocab_size(self) -> "ProjectConfig":
        dataset_vocab_size = _component_value(self.dataset, "vocab_size")
        alphabet = _component_value(self.dataset, "alphabet")

        if dataset_vocab_size is not None and alphabet is not None:
            expected_vocab_size = len(resolve_alphabet(str(alphabet)))
            if int(dataset_vocab_size) != expected_vocab_size:
                raise ValueError(
                    "dataset.vocab_size is derived from dataset.alphabet; "
                    f"expected {expected_vocab_size}, got {dataset_vocab_size}"
                )

        model_vocab_size = _component_value(self.model, "vocab_size")
        if model_vocab_size is not None and dataset_vocab_size is not None:
            if int(model_vocab_size) < int(dataset_vocab_size):
                raise ValueError(
                    "model.vocab_size must be greater than or equal to "
                    "dataset.vocab_size"
                )

        return self


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def dump_yaml(data: Any, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    data = config.model_dump(by_alias=True, polymorphic_serialization=True)
    dump_yaml(data, path)


def load_config(path: str | Path) -> ProjectConfig:
    raw_conf = OmegaConf.load(path)
    data = OmegaConf.to_container(raw_conf, resolve=True)
    return ProjectConfig.model_validate(data)
