import torch
from torch.utils.data import DataLoader, Dataset


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


def build_string_reverse_dataloaders(
    vocab_size: int = 32,
    train_size: int = 256,
    val_size: int = 64,
    batch_size: int = 32,
    seq_len: int = 16,
    num_workers: int = 0,
    seed: int = 7,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = StringReverseDataset(
        size=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    val_dataset = StringReverseDataset(
        size=val_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
