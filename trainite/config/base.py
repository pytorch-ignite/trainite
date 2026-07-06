# __COLLATE_IMPORT__
from typing import Self, Literal, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class TargetedConfig(BaseModel):
    """Used only for DataLoaderConfig.collate_fn."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(alias="_target_")

    @model_validator(mode="before")
    @classmethod
    def coerce_from_model(cls, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump(by_alias=True)
        return data


class OutputConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    root: str
    run_name: str


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(alias="_target_", default="torch.optim.AdamW")
    lr: float = Field(default=1e-3, gt=0.0)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    early_stopping_patience: int | None = Field(default=3, gt=0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)


class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    batch_size: int = Field(default=32, gt=0)
    shuffle: bool = False
    num_workers: int = Field(default=2, ge=0)
    collate_fn: TargetedConfig | None = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    dataset: BaseModel
    transform: BaseModel | None = None
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class DataConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    train: SplitConfig
    val: SplitConfig
    test: SplitConfig | None = None


class DataWithAutoSplit(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    dataset: BaseModel
    transform: BaseModel | None = None
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


class ProjectConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    preprocessor: BaseModel | None = None
    model: BaseModel
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfigBase | DataWithAutoSplit
    trainer: BaseModel = Field(default_factory=TrainerConfig)
    output: OutputConfig
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    seed: int = 42
    device: str | None = None
