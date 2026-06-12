from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trainite.utils import get_target


class ComponentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    implementation_path: Path
    config_cls_path: str
    implementation_symbol: str
    readme_template_path: Path | None = None
    template_replacements: list[tuple[str, str]] = []
    dependencies: list[str] = []

    @property
    def config_cls(self) -> type:
        return get_target(self.config_cls_path)


class TrainerSpec(ComponentSpec):
    project_config_cls_path: str

    @property
    def project_config_cls(self) -> type:
        from trainite.utils import get_target

        return get_target(self.project_config_cls_path)


class ModelSpec(ComponentSpec):
    builder_symbol: str
    collate_fn_target: str | None = None


class DatasetSpec(ComponentSpec):
    builder_symbol: str
    dataset_config_cls_path: str

    @property
    def dataset_config_cls(self) -> type:
        from trainite.utils import get_target

        return get_target(self.dataset_config_cls_path)


MODEL_SPECS = {
    "transformer": ModelSpec(
        name="transformer",
        implementation_path=Path("trainite/models/transformer.py"),
        config_cls_path="trainite.models.transformer.TransformerModelConfig",
        implementation_symbol="TransformerModel",
        builder_symbol="TransformerModel",
        collate_fn_target="trainite.models.transformer.CausalLMCollateFn",
        template_replacements=[
            ("trainite.utils", "utils"),
        ],
        readme_template_path=Path(
            "trainite/templates/components/models/transformer.md"
        ),
    ),
}

DATASET_SPECS = {
    "string-reverse": DatasetSpec(
        name="string_reverse",
        implementation_path=Path("trainite/datasets/string_reverse.py"),
        config_cls_path="trainite.datasets.string_reverse.StringReverseDataConfig",
        dataset_config_cls_path="trainite.datasets.string_reverse.StringReverseDatasetConfig",
        implementation_symbol="StringReverseDataset",
        builder_symbol="StringReverseDataset",
        template_replacements=[
            ("trainite.utils", "utils"),
        ],
        readme_template_path=Path(
            "trainite/templates/components/datasets/string_reverse.md"
        ),
    ),
}

TRAINER_SPECS = {
    "pretrainer": TrainerSpec(
        name="pretrainer",
        implementation_path=Path("trainite/trainers/pretrainer.py"),
        config_cls_path="trainite.trainers.pretrainer.PreTrainerConfig",
        project_config_cls_path="trainite.trainers.pretrainer.ProjectConfig",
        implementation_symbol="PreTrainer",
        template_replacements=[
            ("trainite.utils", "utils"),
        ],
        readme_template_path=Path(
            "trainite/templates/components/trainers/pretrainer.md"
        ),
    ),
}


REGISTRY = {
    "models": MODEL_SPECS,
    "datasets": DATASET_SPECS,
    "trainers": TRAINER_SPECS,
}


class LazyDict(dict):
    def __init__(self, spec_dict):
        self.spec_dict = spec_dict
        super().__init__()

    def __getitem__(self, key):
        return self.spec_dict[key].config_cls

    def __contains__(self, key):
        return key in self.spec_dict

    def keys(self):
        return self.spec_dict.keys()

    def values(self):
        return [spec.config_cls for spec in self.spec_dict.values()]

    def items(self):
        return [(name, spec.config_cls) for name, spec in self.spec_dict.items()]

    def get(self, key, default=None):
        if key in self.spec_dict:
            return self.spec_dict[key].config_cls
        return default


MODEL_CONFIGS = LazyDict(MODEL_SPECS)
DATASET_CONFIGS = LazyDict(DATASET_SPECS)
TRAINER_CONFIGS = LazyDict(TRAINER_SPECS)


def get_model_config_cls(name: str) -> type[BaseModel]:
    return MODEL_CONFIGS[name]


def get_dataset_config_cls(name: str) -> type[BaseModel]:
    return DATASET_CONFIGS[name]


def get_trainer_config_cls(name: str) -> type[BaseModel]:
    return TRAINER_CONFIGS[name]


def get_model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name]


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[name]


def get_trainer_spec(name: str) -> TrainerSpec:
    return TRAINER_SPECS[name]
