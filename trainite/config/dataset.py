from pydantic import Field, ConfigDict, model_validator

from trainite.config.base import ComponentConfig, DataConfig, DataLoaderConfig


class StringReverseDatasetConfig(ComponentConfig):
    model_config = ConfigDict(validate_assignment=True)
    target: str = Field(
        default="trainite.datasets.string_reverse.build_string_reverse_dataset",
        alias="_target_",
    )
    per_seq_size: int = Field(default=256, gt=0)
    charset: str | None = "@alpha"
    min_seq_len: int | None = Field(default=1, gt=0)
    max_seq_len: int | None = Field(default=16, gt=0)
    seq_len: int | None = Field(default=None, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_lengths(self) -> "StringReverseDatasetConfig":
        if self.seq_len is not None and (
            self.min_seq_len is not None or self.max_seq_len is not None
        ):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if self.seq_len is None:
            if self.min_seq_len is None or self.max_seq_len is None:
                raise ValueError(
                    "Must specify either seq_len or both min_seq_len and max_seq_len."
                )
            if self.min_seq_len > self.max_seq_len:
                raise ValueError(
                    "min_seq_len must be less than or equal to max_seq_len."
                )
        return self


class StringReverseDataConfig(DataConfig):
    dataset: ComponentConfig | None = Field(default_factory=StringReverseDatasetConfig)
    train_ratio: float | None = 0.8
    val_ratio: float | None = 0.1
    dataloader: DataLoaderConfig | None = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            collate_fn=ComponentConfig(
                _target_="trainite.datasets.string_reverse.collate_fn"
            ),
        )
    )
