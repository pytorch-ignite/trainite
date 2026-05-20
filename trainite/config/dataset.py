from pydantic import Field
from trainite.config.base import ComponentConfig


class StringReverseDatasetConfig(ComponentConfig):
    target: str = Field(
        default="trainite.datasets.string_reverse.build_string_reverse_dataset",
        alias="_target_",
    )
    size: int = 256
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    min_seq_len: int = 1
    max_seq_len: int = 16
    fixed_length: bool = True
    seed: int = 7
