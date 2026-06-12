from trainite.config.base import (
    ComponentConfig,
    DataLoaderConfig,
    SplitConfig,
)
from trainite.utils import (
    dump_config,
    dump_yaml,
    load_config,
    load_yaml,
)
from trainite.config.registry import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    REGISTRY,
    TRAINER_CONFIGS,
    get_dataset_spec,
    get_model_spec,
    get_trainer_spec,
)


def __getattr__(name: str):
    if name == "StringReverseDatasetConfig":
        from trainite.datasets.string_reverse import StringReverseDatasetConfig

        return StringReverseDatasetConfig
    if name == "TransformerModelConfig":
        from trainite.models.transformer import TransformerModelConfig

        return TransformerModelConfig
    if name == "PreTrainerConfig":
        from trainite.trainers.pretrainer import PreTrainerConfig

        return PreTrainerConfig
    if name == "ProjectConfig":
        from trainite.trainers.pretrainer import ProjectConfig

        return ProjectConfig
    if name == "OutputConfig":
        from trainite.trainers.pretrainer import OutputConfig

        return OutputConfig
    if name == "OptimizerConfig":
        from trainite.trainers.pretrainer import OptimizerConfig

        return OptimizerConfig
    if name == "DataConfig" or name == "DataConfigBase":
        from trainite.config.base import DataConfigBase

        return DataConfigBase
    if name == "TrainerConfig":
        from trainite.trainers.pretrainer import PreTrainerConfig

        return PreTrainerConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(
        list(globals().keys())
        + [
            "StringReverseDatasetConfig",
            "TransformerModelConfig",
            "PreTrainerConfig",
            "ProjectConfig",
            "OutputConfig",
            "OptimizerConfig",
            "DataConfig",
            "DataConfigBase",
            "TrainerConfig",
        ]
    )


__all__ = [
    "ProjectConfig",
    "OutputConfig",
    "PreTrainerConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "StringReverseDatasetConfig",
    "TransformerModelConfig",
    "ComponentConfig",
    "DataConfig",
    "DataConfigBase",
    "SplitConfig",
    "DataLoaderConfig",
    "dump_config",
    "dump_yaml",
    "get_dataset_spec",
    "get_model_spec",
    "get_trainer_spec",
    "load_config",
    "load_yaml",
    "DATASET_CONFIGS",
    "MODEL_CONFIGS",
    "REGISTRY",
    "TRAINER_CONFIGS",
]
