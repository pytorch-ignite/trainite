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

    Maps each character in the alphabet to a unique integer ID.
    ID 0 is reserved as the padding token.
    ID 1 is reserved as the BOS token.
    ID 2 is reserved as the EOS token.
    """

    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        self.alphabet = ALPHABET_PRESETS.get(alphabet, alphabet)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        self.char_to_id = {c: i + 3 for i, c in enumerate(self.alphabet)}
        self.id_to_char = {i + 3: c for i, c in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(self.alphabet) + 3

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert a list of token IDs back to a string, skipping padding and special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special_token_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return "".join(
            self.id_to_char[i] for i in ids if i not in special_token_ids and i in self.id_to_char
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

        self.inputs = []
        self.labels = []

        for _ in range(size):
            if fixed_length:
                length = max_seq_len
            else:
                length = torch.randint(
                    low=min_seq_len,
                    high=max_seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()

            seq = torch.randint(
                low=3,
                high=self.vocab_size,
                size=(length,),
                generator=generator,
            )
            
            reversed_seq = torch.flip(seq, dims=[0])
            
            bos_t = torch.tensor([self.tokenizer.bos_token_id])
            eos_t = torch.tensor([self.tokenizer.eos_token_id])
            
            full_seq = torch.cat([bos_t, seq, eos_t, reversed_seq, eos_t])
            
            input_ids = full_seq[:-1]
            target_labels = full_seq[1:].clone()
            
            prompt_len = len(seq) + 2
            target_labels[:prompt_len - 1] = -100
            
            self.inputs.append(input_ids)
            self.labels.append(target_labels)

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
