from typing import Any, Self, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoerceFromModel(BaseModel):
    """Accepts any BaseModel instance as input by dumping it to a dict first."""

    @model_validator(mode="before")
    @classmethod
    def validate_from_any_model(cls, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump(by_alias=True)
        return data


class TargetedConfig(CoerceFromModel):
    """Config for a component addressed by a ``_target_`` import path."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(alias="_target_")


class ModelConfig(TargetedConfig):
    pass


class PreprocessorConfig(TargetedConfig):
    pass


class DatasetConfig(TargetedConfig):
    pass


class TransformConfig(TargetedConfig):
    pass


class CollateFnConfig(TargetedConfig):
    pass


class OutputConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    root: str
    run_name: str


class OptimizerConfig(TargetedConfig):
    target: str = Field(alias="_target_", default="torch.optim.AdamW")
    lr: float = Field(default=1e-3, gt=0.0)


class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    batch_size: int = Field(default=32, gt=0)
    shuffle: bool = False
    num_workers: int = Field(default=2, ge=0)
    collate_fn: CollateFnConfig | None = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    dataset: DatasetConfig
    transform: TransformConfig | None = None
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class DataConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    train: SplitConfig
    val: SplitConfig
    test: SplitConfig | None = None


class DataWithAutoSplit(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    # Option 2: Automatic splitting
    dataset: DatasetConfig
    transform: TransformConfig | None = None
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    test_ratio: float = Field(default=0.2, ge=0.0, lt=0.5)
    val_ratio: float = Field(default=0.1, gt=0.0, lt=0.3)

    @model_validator(mode="after")
    def check_split_ratios(self) -> Self:
        train_ratio = 1.0 - self.test_ratio - self.val_ratio
        if train_ratio <= 0.5:
            raise ValueError("train ratio (1.0 - test_ratio - val_ratio) must be greater than 0.5")
        if train_ratio > 0.9:
            raise ValueError("train ratio (1.0 - test_ratio - val_ratio) must be less than or equal to 0.9")
        return self


class TrainerConfig(CoerceFromModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    early_stopping_patience: int | None = Field(default=3, gt=0)
    # Inference logging
    inference_every_epochs: int | None = Field(default=None, gt=0)
    inference_num_samples: int = Field(default=5, gt=0)
    max_inference_new_tokens: int = Field(default=16, gt=0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    preprocessor: PreprocessorConfig | None = None
    model: ModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfigBase | DataWithAutoSplit
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    output: OutputConfig
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    seed: int = 42
    device: str | None = None
