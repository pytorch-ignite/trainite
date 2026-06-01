import string
import warnings

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Hardcoded universal vocabulary: all printable ASCII characters
UNIVERSAL_VOCAB = string.ascii_letters + string.digits + string.punctuation + " "

CHARSET_PRESETS = {
    "@universal": UNIVERSAL_VOCAB,
    "@alpha": string.ascii_letters,
    "@digits": string.digits,
    "@alphanumeric": string.ascii_letters + string.digits,
}


class CharTokenizer:
    """A simple character-level tokenizer with a hardcoded universal vocabulary.

    Maps each character in UNIVERSAL_VOCAB to a unique integer ID.
    ID 0 is reserved as the padding token.
    ID 1 is reserved as the BOS token.
    ID 2 is reserved as the EOS token.
    ID 3 is reserved as the UNK token.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.char_to_id: dict[str, int] = {
            c: i + 4 for i, c in enumerate(UNIVERSAL_VOCAB)
        }
        self.id_to_char: dict[int, str] = {
            i + 4: c for i, c in enumerate(UNIVERSAL_VOCAB)
        }

        self.special_tokens: dict[int, str] = {
            self.pad_token_id: "<pad>",
            self.bos_token_id: "<bos>",
            self.eos_token_id: "<eos>",
            self.unk_token_id: "<unk>",
        }

        for k, v in self.special_tokens.items():
            self.id_to_char[k] = v

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(UNIVERSAL_VOCAB) + 4

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs, mapping unrecognized characters to UNK."""
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        decoded_chars = []
        for i in ids:
            if i == ignore_index:
                continue
            if i in self.special_tokens:
                if not skip_special_tokens or i == self.unk_token_id:
                    decoded_chars.append(self.special_tokens[i])
            elif i in self.id_to_char:
                decoded_chars.append(self.id_to_char[i])
            else:
                decoded_chars.append(self.special_tokens[self.unk_token_id])
        return "".join(decoded_chars)


class StringReverseDataset(Dataset):
    """Generates unique random strings and their reversals for autoregressive training.

    Each sample is packed as:
        <bos> seq <eos> reversed_seq <eos>

    With teacher forcing labels that mask the prompt portion (-100).
    """

    def __init__(
        self,
        per_seq_size: int,
        min_seq_len: int | None = None,
        max_seq_len: int | None = None,
        seq_len: int | None = None,
        charset: str | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer: CharTokenizer = CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        if charset is None:
            chars = UNIVERSAL_VOCAB
        elif charset in CHARSET_PRESETS:
            chars = CHARSET_PRESETS[charset]
        else:
            chars = charset

        self.valid_token_ids = [
            self.tokenizer.char_to_id[c]
            for c in chars
            if c in self.tokenizer.char_to_id
        ]

        if not self.valid_token_ids:
            raise ValueError(f"Charset '{charset}' resulted in empty token IDs.")

        if seq_len is not None and (min_seq_len is not None or max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if seq_len is not None:
            lengths = [seq_len]
        elif min_seq_len is None or max_seq_len is None:
            raise ValueError(
                "Must specify either seq_len or both min_seq_len and max_seq_len."
            )
        else:
            lengths = list(range(min_seq_len, max_seq_len + 1))

        generator = torch.Generator().manual_seed(seed)
        self.valid_token_ids_tensor = torch.tensor(self.valid_token_ids)
        vocab_size = len(self.valid_token_ids)

        self.inputs = []
        self.labels = []

        # Generate per_seq_size unique sequences for each length bucket
        for length in lengths:
            unique_sequences: set[tuple[int, ...]] = set()
            max_possible_combinations = vocab_size**length
            target = min(per_seq_size, max_possible_combinations)

            # Safety cap to avoid infinite loops when the space is small
            max_attempts = target * 20

            while (
                len(unique_sequences) < target and len(unique_sequences) < max_attempts
            ):
                indices = torch.randint(
                    low=0,
                    high=len(self.valid_token_ids),
                    size=(length,),
                    generator=generator,
                )
                seq_tuple = tuple(self.valid_token_ids_tensor[indices].tolist())
                unique_sequences.add(seq_tuple)

            if len(unique_sequences) < per_seq_size:
                warnings.warn(
                    f"Requested {per_seq_size} unique sequences for seq_len={length} "
                    f"but only {len(unique_sequences)} could be generated "
                    f"(max possible: {max_possible_combinations}).",
                    stacklevel=2,
                )

            # Pack each unique sequence into the autoregressive format
            for seq_tuple in unique_sequences:
                seq = torch.tensor(seq_tuple)
                reversed_seq = torch.flip(seq, dims=[0])

                bos_t = torch.tensor([self.tokenizer.bos_token_id])
                eos_t = torch.tensor([self.tokenizer.eos_token_id])

                full_seq = torch.cat([bos_t, seq, eos_t, reversed_seq, eos_t])

                input_ids = full_seq[:-1]
                target_labels = full_seq[1:].clone()

                prompt_len = len(seq) + 2
                target_labels[: prompt_len - 1] = -100

                self.inputs.append(input_ids)
                self.labels.append(target_labels)

        # Shuffle so variable-length samples are evenly distributed across batches
        final_shuffle_gen = torch.Generator().manual_seed(seed + 1)
        shuffle_indices = torch.randperm(
            len(self.inputs), generator=final_shuffle_gen
        ).tolist()

        self.inputs = [self.inputs[i] for i in shuffle_indices]
        self.labels = [self.labels[i] for i in shuffle_indices]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }

    def decode(
        self,
        ids: torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens, ignore_index=ignore_index
        )


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
    per_seq_size: int = 256,
    min_seq_len: int | None = None,
    max_seq_len: int | None = None,
    seq_len: int | None = None,
    charset: str | None = None,
    seed: int = 42,
    **kwargs,
) -> StringReverseDataset:
    return StringReverseDataset(
        per_seq_size=per_seq_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
        seq_len=seq_len,
        charset=charset,
    )
