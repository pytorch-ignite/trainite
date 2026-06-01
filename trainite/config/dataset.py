from pydantic import Field

from trainite.config.base import ComponentConfig, DataConfig, DataLoaderConfig


class StringReverseDatasetConfig(ComponentConfig):
    target: str = Field(
        default="trainite.datasets.string_reverse.build_string_reverse_dataset",
        alias="_target_",
    )
    per_seq_size: int = 256
    charset: str | None = "@alpha"
    min_seq_len: int | None = 1
    max_seq_len: int | None = 16
    seq_len: int | None = None
    seed: int = 42


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
