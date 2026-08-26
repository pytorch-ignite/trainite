from typing import Literal, Self
from pydantic import Field, model_validator
from trainite.config.base import DatasetConfig, TransformConfig, DataWithAutoSplit, DataLoaderConfig


class PromptCompletionTransformConfig(TransformConfig):
    target: Literal[
        "trainite.datasets.string_reverse.PromptCompletionTransform",
        "dataset_impl.string_reverse.PromptCompletionTransform",
    ] = Field(
        default="trainite.datasets.string_reverse.PromptCompletionTransform",
        alias="_target_",
    )
    ignore_index: int = -100


class StringReverseDatasetConfig(DatasetConfig):
    target: Literal[
        "trainite.datasets.string_reverse.StringReverseDataset",
        "dataset_impl.string_reverse.StringReverseDataset",
    ] = Field(
        default="trainite.datasets.string_reverse.StringReverseDataset",
        alias="_target_",
    )
    per_seq_size: int = Field(default=256, gt=0)
    charset: str | None = "@alpha"
    min_seq_len: int | None = Field(default=1, gt=0)
    max_seq_len: int | None = Field(default=16, gt=0)
    seq_len: int | None = Field(default=None, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if self.seq_len is not None and (self.min_seq_len is not None or self.max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if self.seq_len is None:
            if self.min_seq_len is None or self.max_seq_len is None:
                raise ValueError("Must specify either seq_len or both min_seq_len and max_seq_len.")
            if self.min_seq_len > self.max_seq_len:
                raise ValueError("min_seq_len must be less than or equal to max_seq_len.")
        return self


class StringReverseDataConfig(DataWithAutoSplit):
    dataset: StringReverseDatasetConfig | None = Field(  # type: ignore[assignment]
        default_factory=StringReverseDatasetConfig
    )
    transform: PromptCompletionTransformConfig | None = Field(default_factory=PromptCompletionTransformConfig)
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    dataloader: DataLoaderConfig = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
        )
    )


class CountingTransformConfig(TransformConfig):
    target: Literal[
        "trainite.datasets.counting.CountingTransform",
        "dataset_impl.counting.CountingTransform",
    ] = Field(
        default="trainite.datasets.counting.CountingTransform",
        alias="_target_",
    )
    ignore_index: int = -100


class CountingDatasetConfig(DatasetConfig):
    target: Literal[
        "trainite.datasets.counting.CountingDataset",
        "dataset_impl.counting.CountingDataset",
    ] = Field(
        default="trainite.datasets.counting.CountingDataset",
        alias="_target_",
    )
    total_size: int = Field(default=100, gt=0)
    k: int = Field(default=3, gt=0)
    min_seq_len: int | None = Field(default=10, gt=0)
    max_seq_len: int | None = Field(default=20, gt=0)
    seq_len: int | None = Field(default=None, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if self.seq_len is not None and (self.min_seq_len is not None or self.max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if self.seq_len is None:
            if self.min_seq_len is None or self.max_seq_len is None:
                raise ValueError("Must specify either seq_len or both min_seq_len and max_seq_len.")
            if self.min_seq_len > self.max_seq_len:
                raise ValueError("min_seq_len must be less than or equal to max_seq_len.")
        return self


class CountingDataConfig(DataWithAutoSplit):
    dataset: CountingDatasetConfig | None = Field(  # type: ignore[assignment]
        default_factory=CountingDatasetConfig
    )
    transform: CountingTransformConfig | None = Field(default_factory=CountingTransformConfig)
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    dataloader: DataLoaderConfig = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
        )
    )


class HuggingFaceTransformConfig(TransformConfig):
    target: Literal[
        "trainite.datasets.hugging_face.HuggingFaceTransform",
        "dataset_impl.hugging_face.HuggingFaceTransform",
    ] = Field(
        default="trainite.datasets.hugging_face.HuggingFaceTransform",
        alias="_target_",
    )
    max_length: int = Field(default=128, gt=1)
    ignore_index: int = -100


class HuggingFaceDatasetConfig(DatasetConfig):
    target: Literal["datasets.load_dataset"] = Field(default="datasets.load_dataset", alias="_target_")
    path: str = Field(default="namespace/dataset-name", min_length=1)
    name: str | None = None
    split: str = Field(default="train", min_length=1)
    revision: str | None = None


class HuggingFaceDataConfig(DataWithAutoSplit):
    dataset: HuggingFaceDatasetConfig = Field(default_factory=HuggingFaceDatasetConfig)  # type: ignore[assignment]
    transform: HuggingFaceTransformConfig = Field(default_factory=HuggingFaceTransformConfig)
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    dataloader: DataLoaderConfig = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
        )
    )
