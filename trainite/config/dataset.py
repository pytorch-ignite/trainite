from __future__ import annotations

from pydantic import BaseModel, Field


class StringReverseDatasetConfig(BaseModel):
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
    seed: int = 7


DATASET_CONFIGS: dict[str, type[BaseModel]] = {
    "string-reverse": StringReverseDatasetConfig,
}
