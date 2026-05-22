import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


ALPHABET_PRESETS = {
    "@alpha": "abcdefghijklmnopqrstuvwxyz",
    "@digits": "0123456789",
    "@alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
}


class StringReverseDataset(Dataset):
    def __init__(
        self,
        size: int,
        min_seq_len: int,
        max_seq_len: int,
        seed: int,
        fixed_length: bool = True,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    ) -> None:
        self.alphabet = ALPHABET_PRESETS.get(alphabet, alphabet)
        vocab_size = len(self.alphabet)
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.alphabet)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.alphabet)}

        generator = torch.Generator().manual_seed(seed)

        if fixed_length:
            self.inputs = torch.randint(
                low=1,
                high=vocab_size + 1,
                size=(size, max_seq_len),
                generator=generator,
            )
            self.labels = torch.flip(self.inputs, dims=[1])
        else:
            self.inputs = []
            for _ in range(size):
                length = torch.randint(
                    low=min_seq_len,
                    high=max_seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()

                seq = torch.randint(
                    low=1,
                    high=vocab_size + 1,
                    size=(length,),
                    generator=generator,
                )
                self.inputs.append(seq)

            self.labels = [torch.flip(seq, dims=[0]) for seq in self.inputs]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }

    def decode(self, ids: torch.Tensor) -> str:
        return "".join([self.id_to_char[idx.item()] for idx in ids if idx.item() != 0])


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
    }


def build_string_reverse_dataloaders(
    train_size: int = 256,
    val_size: int = 64,
    batch_size: int = 32,
    min_seq_len: int = 1,
    max_seq_len: int = 16,
    fixed_length: bool = True,
    alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    num_workers: int = 0,
    seed: int = 7,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = StringReverseDataset(
        size=train_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
        fixed_length=fixed_length,
        alphabet=alphabet,
    )
    val_dataset = StringReverseDataset(
        size=val_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed + 1,
        fixed_length=fixed_length,
        alphabet=alphabet,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader
