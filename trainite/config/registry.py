from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trainite.config.dataset import StringReverseDatasetConfig
from trainite.config.model import TransformerModelConfig
from trainite.config.trainer import PreTrainerConfig


class ComponentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    implementation_path: Path
    config_cls: type
    implementation_symbol: str
    template_replacements: list[tuple[str, str]] = []
    dependencies: list[str] = []


class TrainerSpec(ComponentSpec):
    pass


class ModelSpec(ComponentSpec):
    builder_symbol: str


class DatasetSpec(ComponentSpec):
    builder_symbol: str
    collate_fn_symbol: str | None = None


MODEL_SPECS = {
    "transformer": ModelSpec(
        name="transformer",
        implementation_path=Path("trainite/models/transformer.py"),
        config_cls=TransformerModelConfig,
        implementation_symbol="TransformerModel",
        builder_symbol="build_transformer_model",
    ),
}

DATASET_SPECS = {
    "string-reverse": DatasetSpec(
        name="string_reverse",
        implementation_path=Path("trainite/datasets/string_reverse.py"),
        config_cls=StringReverseDatasetConfig,
        implementation_symbol="StringReverseDataset",
        builder_symbol="build_string_reverse_dataset",
        collate_fn_symbol="collate_fn",
    ),
}

TRAINER_SPECS = {
    "pretrainer": TrainerSpec(
        name="pretrainer",
        implementation_path=Path("trainite/trainers/pretrainer.py"),
        config_cls=PreTrainerConfig,
        implementation_symbol="PreTrainer",
        template_replacements=[
            (
                "trainite.config",
                "config",
            ),
            ("trainite.utils", "utils"),
        ],
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
