from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from config import StringReverseDatasetConfig


class StringReverseDataset(Dataset):
    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randint(
            low=0,
            high=vocab_size,
            size=(size, seq_len),
            generator=generator,
        )
        self.labels = torch.flip(self.inputs, dims=[1])

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }


def build_dataloaders(
    config: StringReverseDatasetConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = StringReverseDataset(
        size=config.train_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        seed=config.seed,
    )
    val_dataset = StringReverseDataset(
        size=config.val_size,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        seed=config.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
