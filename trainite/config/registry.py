from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from trainite.shared.utils import get_target


class ComponentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    implementation_path: Path
    config_cls_path: str
    implementation_symbol: str
    readme_template_path: Path | None = None
    dependencies: list[str] = []

    @property
    def config_cls(self) -> Any:
        return get_target(self.config_cls_path)


class TrainerSpec(ComponentSpec):
    pass


class ModelSpec(ComponentSpec):
    builder_symbol: str
    collate_fn_target: str | None = None
    collate_fn_config_cls_path: str | None = None

    @property
    def collate_fn_config_cls(self) -> type[BaseModel] | None:
        return get_target(self.collate_fn_config_cls_path) if self.collate_fn_config_cls_path else None


class DatasetSpec(ComponentSpec):
    builder_symbol: str
    dataset_config_cls_path: str
    preprocessor_spec_name: str | None = None

    @property
    def dataset_config_cls(self) -> type[BaseModel]:
        return get_target(self.dataset_config_cls_path)


class PreProcessorSpec(ComponentSpec):
    pass


MODEL_SPECS = {
    "transformer": ModelSpec(
        name="transformer",
        implementation_path=Path("trainite/models/transformer.py"),
        config_cls_path="trainite.config.models.TransformerModelConfig",
        implementation_symbol="TransformerModel",
        builder_symbol="TransformerModel",
        collate_fn_target="trainite.models.transformer.CausalLMCollateFn",
        collate_fn_config_cls_path="trainite.config.models.CausalLMCollateFnConfig",
        readme_template_path=Path("trainite/templates/components/models/transformer.md"),
    ),
}

DATASET_SPECS = {
    "string-reverse": DatasetSpec(
        name="string_reverse",
        implementation_path=Path("trainite/datasets/string_reverse.py"),
        config_cls_path="trainite.config.datasets.StringReverseDataConfig",
        dataset_config_cls_path="trainite.config.datasets.StringReverseDatasetConfig",
        implementation_symbol="StringReverseDataset",
        builder_symbol="StringReverseDataset",
        readme_template_path=Path("trainite/templates/components/datasets/string_reverse.md"),
        preprocessor_spec_name="char",
    ),
}

TRAINER_SPECS = {
    "decoder-trainer": TrainerSpec(
        name="decoder_trainer",
        implementation_path=Path("trainite/trainers/decoder_trainer.py"),
        config_cls_path="trainite.config.trainers.TrainerConfig",
        implementation_symbol="Trainer",
        readme_template_path=Path("trainite/templates/components/trainers/decoder_trainer.md"),
    ),
}

PREPROCESSOR_SPECS = {
    "char": PreProcessorSpec(
        name="char_tokenizer",
        implementation_path=Path("trainite/preprocessors/char_tokenizer.py"),
        config_cls_path="trainite.config.preprocessors.CharTokenizerConfig",
        implementation_symbol="CharTokenizer",
        readme_template_path=Path("trainite/templates/components/preprocessors/char.md"),
    ),
}


REGISTRY = {
    "models": MODEL_SPECS,
    "datasets": DATASET_SPECS,
    "trainers": TRAINER_SPECS,
    "preprocessors": PREPROCESSOR_SPECS,
}


def get_model_config_cls(name: str) -> type[BaseModel]:
    return MODEL_SPECS[name].config_cls


def get_dataset_config_cls(name: str) -> type[BaseModel]:
    return DATASET_SPECS[name].dataset_config_cls


def get_trainer_config_cls(name: str) -> type[BaseModel]:
    return TRAINER_SPECS[name].config_cls


def get_model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name]


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[name]


def get_trainer_spec(name: str) -> TrainerSpec:
    return TRAINER_SPECS[name]


def get_preprocessor_spec(name: str) -> PreProcessorSpec:
    return PREPROCESSOR_SPECS[name]
