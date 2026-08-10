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


class ModelSpec(ComponentSpec):
    builder_symbol: str
    collate_fn_target: str | None = None


class DatasetSpec(ComponentSpec):
    builder_symbol: str
    dataset_config_cls_path: str
    preprocessor_spec_name: str | None = None

    @property
    def dataset_config_cls(self) -> type[BaseModel]:
        return get_target(self.dataset_config_cls_path)


MODEL_SPECS = {
    "basic-transformer": ModelSpec(
        name="basic_transformer",
        implementation_path=Path("trainite/models/basic_transformer.py"),
        config_cls_path="trainite.config.models.BasicTransformerModelConfig",
        implementation_symbol="BasicTransformerModel",
        builder_symbol="BasicTransformerModel",
        collate_fn_target="trainite.models.basic_transformer.CausalLMCollateFn",
        readme_template_path=Path("trainite/templates/components/models/basic_transformer.md"),
    ),
    "rope-transformer": ModelSpec(
        name="rope_transformer",
        implementation_path=Path("trainite/models/rope_transformer.py"),
        config_cls_path="trainite.config.models.RoPETransformerModelConfig",
        implementation_symbol="RoPETransformerModel",
        builder_symbol="RoPETransformerModel",
        collate_fn_target="trainite.models.rope_transformer.CausalLMCollateFn",
        readme_template_path=Path("trainite/templates/components/models/rope_transformer.md"),
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
    "counting": DatasetSpec(
        name="counting",
        implementation_path=Path("trainite/datasets/counting.py"),
        config_cls_path="trainite.config.datasets.CountingDataConfig",
        dataset_config_cls_path="trainite.config.datasets.CountingDatasetConfig",
        implementation_symbol="CountingDataset",
        builder_symbol="CountingDataset",
        readme_template_path=Path("trainite/templates/components/datasets/counting.md"),
        preprocessor_spec_name="char",
    ),
}

TRAINER_SPECS = {
    "decoder-trainer": ComponentSpec(
        name="decoder_trainer",
        implementation_path=Path("trainite/trainers/decoder_trainer.py"),
        config_cls_path="trainite.config.trainers.TrainerConfig",
        implementation_symbol="Trainer",
        readme_template_path=Path("trainite/templates/components/trainers/decoder_trainer.md"),
    ),
}

PREPROCESSOR_SPECS = {
    "char": ComponentSpec(
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
