from dataclasses import dataclass
from pathlib import Path

from trainite.config.dataset import DATASET_CONFIGS
from trainite.config.model import MODEL_CONFIGS
from trainite.config.trainer import TRAINER_CONFIGS


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    implementation_path: Path
    config_cls: type
    implementation_symbol: str
    builder_symbol: str


@dataclass(frozen=True)
class TrainerSpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class ModelSpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class DatasetSpec(ComponentSpec):
    pass


REGISTRY = {
    "models": {
        "transformer": ModelSpec(
            name="transformer",
            implementation_path=Path("trainite/models/transformer.py"),
            config_cls=MODEL_CONFIGS["transformer"],
            implementation_symbol="TransformerModel",
            builder_symbol="build_transformer_model",
        ),
    },
    "datasets": {
        "string-reverse": DatasetSpec(
            name="string-reverse",
            implementation_path=Path("trainite/datasets/string_reverse.py"),
            config_cls=DATASET_CONFIGS["string-reverse"],
            implementation_symbol="StringReverseDataset",
            builder_symbol="build_string_reverse_dataloaders",
        ),
    },
    "trainers": {
        "pretrainer": TrainerSpec(
            name="pretrainer",
            implementation_path=Path("trainite/trainers/pretrainer.py"),
            config_cls=TRAINER_CONFIGS["pretrainer"],
            implementation_symbol="PreTrainer",
            builder_symbol="PreTrainer",
        ),
    },
}


def get_model_config_cls(name: str):
    return MODEL_CONFIGS[name]


def get_dataset_config_cls(name: str):
    return DATASET_CONFIGS[name]


def get_trainer_config_cls(name: str):
    return TRAINER_CONFIGS[name]


def get_model_spec(name: str) -> ModelSpec:
    return REGISTRY["models"][name]


def get_dataset_spec(name: str) -> DatasetSpec:
    return REGISTRY["datasets"][name]


def get_trainer_spec(name: str) -> TrainerSpec:
    return REGISTRY["trainers"][name]
