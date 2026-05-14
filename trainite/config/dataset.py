from pydantic import Field

from trainite.config.base import ComponentConfig


class StringReverseDatasetConfig(ComponentConfig):
    target: str = Field(
        default="trainite.datasets.string_reverse.build_string_reverse_dataloaders",
        alias="_target_",
    )
    vocab_size: int = 32
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    seq_len: int = 16
    num_workers: int = 0
