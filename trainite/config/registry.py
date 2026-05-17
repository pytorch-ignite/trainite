from dataclasses import dataclass, field
from pathlib import Path

from trainite.config.dataset import StringReverseDatasetConfig
from trainite.config.model import TransformerModelConfig
from trainite.config.trainer import PreTrainerConfig


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    implementation_path: Path
    config_cls: type
    implementation_symbol: str
    builder_symbol: str
    template_replacements: list[tuple[str, str]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainerSpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class ModelSpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class DatasetSpec(ComponentSpec):
    pass


MODEL_SPECS = {
    "transformer": ModelSpec(
        name="transformer",
        implementation_path=Path("trainite/models/transformer.py"),
        config_cls=TransformerModelConfig,
        implementation_symbol="TransformerModel",
        builder_symbol="build_transformer_model",
        template_replacements=[],
        dependencies=["torch"],
    ),
}

DATASET_SPECS = {
    "string-reverse": DatasetSpec(
        name="string-reverse",
        implementation_path=Path("trainite/datasets/string_reverse.py"),
        config_cls=StringReverseDatasetConfig,
        implementation_symbol="StringReverseDataset",
        builder_symbol="build_string_reverse_dataloaders",
        template_replacements=[],
        dependencies=["torch"],
    ),
}

TRAINER_SPECS = {
    "pretrainer": TrainerSpec(
        name="pretrainer",
        implementation_path=Path("trainite/trainers/pretrainer.py"),
        config_cls=PreTrainerConfig,
        implementation_symbol="PreTrainer",
        builder_symbol="PreTrainer",
        template_replacements=[
            (
                "trainite.config",
                "config",
            ),
            ("trainite.utils", "utils"),
        ],
        dependencies=["torch", "pytorch-ignite", "tensorboard"],
    ),
}


REGISTRY = {
    "models": MODEL_SPECS,
    "datasets": DATASET_SPECS,
    "trainers": TRAINER_SPECS,
}


MODEL_CONFIGS = {name: spec.config_cls for name, spec in MODEL_SPECS.items()}
DATASET_CONFIGS = {name: spec.config_cls for name, spec in DATASET_SPECS.items()}
TRAINER_CONFIGS = {name: spec.config_cls for name, spec in TRAINER_SPECS.items()}


def get_model_config_cls(name: str):
    return MODEL_CONFIGS[name]


def get_dataset_config_cls(name: str):
    return DATASET_CONFIGS[name]


def get_trainer_config_cls(name: str):
    return TRAINER_CONFIGS[name]


def get_model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name]


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[name]


def get_trainer_spec(name: str) -> TrainerSpec:
    return TRAINER_SPECS[name]
