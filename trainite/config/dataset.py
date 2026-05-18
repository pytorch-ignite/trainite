from pydantic import Field

from trainite.config.base import ComponentConfig


class StringReverseDatasetConfig(ComponentConfig):
    target: str = Field(
        default="trainite.datasets.string_reverse.build_string_reverse_dataloaders",
        alias="_target_",
    )
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    vocab_size: int = 26
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    min_seq_len: int = 1
    max_seq_len: int = 16
    fixed_length: bool = True
    num_workers: int = 2
    seed: int = 7
