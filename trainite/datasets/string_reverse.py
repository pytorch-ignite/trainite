import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


ALPHABET_PRESETS = {
    "@alpha": "abcdefghijklmnopqrstuvwxyz",
    "@digits": "0123456789",
    "@alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
}


class CharTokenizer:
    """A simple character-level tokenizer.

    Maps each character in the alphabet to a unique integer ID (1-indexed).
    ID 0 is reserved as the padding token.
    """

    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        self.alphabet = ALPHABET_PRESETS.get(alphabet, alphabet)
        self.pad_token_id = 0
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.alphabet)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (excludes padding token)."""
        return len(self.alphabet)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert a list of token IDs back to a string, skipping padding tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            self.id_to_char[i] for i in ids if i != self.pad_token_id
        )


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
        self.tokenizer = CharTokenizer(alphabet)
        self.vocab_size = self.tokenizer.vocab_size

        generator = torch.Generator().manual_seed(seed)

        if fixed_length:
            self.inputs = torch.randint(
                low=1,
                high=self.vocab_size + 1,
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
                    high=self.vocab_size + 1,
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
        return self.tokenizer.decode(ids)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
    }


def build_string_reverse_dataset(
    size: int = 256,
    min_seq_len: int = 1,
    max_seq_len: int = 16,
    fixed_length: bool = True,
    alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    seed: int = 7,
    **kwargs,
) -> StringReverseDataset:
    return StringReverseDataset(
        size=size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
        fixed_length=fixed_length,
        alphabet=alphabet,
    )
