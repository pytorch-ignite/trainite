from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class OutputConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    root: str
    run_name: str


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(alias="_target_")


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(alias="_target_", default="torch.optim.AdamW")
    lr: float = Field(default=1e-3, gt=0.0)


class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    batch_size: int = Field(default=32, gt=0)
    shuffle: bool = False
    num_workers: int = Field(default=2, ge=0)
    collate_fn: ComponentConfig | None = None


class SplitConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    dataset: ComponentConfig
    transform: ComponentConfig | None = None
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class DataConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    train: SplitConfig
    val: SplitConfig
    test: SplitConfig | None = None


class DataWithAutoSplit(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    # Option 2: Automatic splitting
    dataset: ComponentConfig
    transform: ComponentConfig | None = None
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
