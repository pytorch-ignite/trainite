from collections.abc import Callable
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
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
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class DataConfigBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    # Option 1: Explicit splits
    train: SplitConfig | None = None
    val: SplitConfig | None = None
    test: SplitConfig | None = None

    # Option 2: Automatic splitting
    dataset: ComponentConfig | None = None
    dataloader: DataLoaderConfig | None = None
    train_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    val_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)

    @model_serializer(mode="wrap")
    def serialize_model(
        self, handler: Callable[[Any], dict[str, Any]]
    ) -> dict[str, Any]:
        res = handler(self)
        if isinstance(res, dict):
            return {k: v for k, v in res.items() if v is not None}
        return res

    @model_validator(mode="after")
    def validate_options(self) -> "DataConfigBase":
        option1_fields = {"train", "val", "test"}
        option2_fields = {"dataset", "train_ratio", "val_ratio", "dataloader"}

        present_option1 = {f for f in option1_fields if getattr(self, f) is not None}
        present_option2 = {f for f in option2_fields if getattr(self, f) is not None}

        if present_option1 and present_option2:
            if "dataset" in present_option2 or "dataloader" in present_option2:
                raise ValueError(
                    "Cannot provide train/val/test levels when 'dataset' or 'dataloader' is provided at the data level"
                )
            raise ValueError(
                "Cannot provide train_ratio or val_ratio at the data level when explicit splits are used"
            )

        if not present_option1 and not present_option2:
            raise ValueError(
                "Must provide either explicit splits (train) or automatic splitting (dataset)"
            )

        if present_option1 and "train" not in present_option1:
            raise ValueError("Explicit splits mode requires at least the 'train' split")

        if present_option2 and "dataset" not in present_option2:
            raise ValueError("Automatic splitting mode requires the 'dataset' field")

        if self.dataset is not None:
            train_ratio = self.train_ratio if self.train_ratio is not None else 1.0
            val_ratio = self.val_ratio if self.val_ratio is not None else 0.0

            if train_ratio + val_ratio > 1.0:
                raise ValueError(
                    f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) exceeds 1.0"
                )

        return self
