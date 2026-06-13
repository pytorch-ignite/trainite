import random
import string
import warnings

import torch
from pydantic import ConfigDict, Field, model_validator
from torch.utils.data import Dataset

from trainite.config.base import ComponentConfig, DataConfigBase, DataLoaderConfig

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
    ID 0 is reserved as the <PAD> token.
    ID 1 is reserved as the <BOS> token.
    ID 2 is reserved as the <SEP> token.
    ID 3 is reserved as the <EOS> token.
    ID 4 is reserved as the <UNK> token.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 3
        self.unk_token_id = 4

        self.char_to_id: dict[str, int] = {
            c: i + 5 for i, c in enumerate(UNIVERSAL_VOCAB)
        }
        self.id_to_char: dict[int, str] = {
            i + 5: c for i, c in enumerate(UNIVERSAL_VOCAB)
        }

        self.special_tokens: dict[int, str] = {
            self.pad_token_id: "<PAD>",
            self.bos_token_id: "<BOS>",
            self.sep_token_id: "<SEP>",
            self.eos_token_id: "<EOS>",
            self.unk_token_id: "<UNK>",
        }

        for k, v in self.special_tokens.items():
            self.id_to_char[k] = v

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(UNIVERSAL_VOCAB) + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs, mapping unrecognized characters to UNK."""
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: List or tensor of token IDs to decode.
            skip_special_tokens: If True, skips printing special tokens like <bos> and <eos>.
            ignore_index: The token ID used for loss masking. These are silently ignored.
        """
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
    """Generates unique random strings and their reversals.

    Each sample is returned as a dictionary containing:
        - 'source_text': the original random string
        - 'target_text': the reversed string
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

        rng = random.Random(seed)
        vocab_size = len(self.valid_token_ids)

        self.source_texts = []
        self.target_texts = []

        # Generate per_seq_size unique sequences for each length bucket
        for length in lengths:
            unique_sequences: set[str] = set()
            max_possible_combinations = vocab_size**length
            target = min(per_seq_size, max_possible_combinations)

            # Safety cap to avoid infinite loops when the space is small
            attempts = 0
            max_attempts = target * 20

            while len(unique_sequences) < target and attempts < max_attempts:
                seq = "".join(rng.choices(chars, k=length))
                unique_sequences.add(seq)
                attempts += 1

            if len(unique_sequences) < per_seq_size:
                warnings.warn(
                    f"Requested {per_seq_size} unique sequences for seq_len={length} "
                    f"but only {len(unique_sequences)} could be generated "
                    f"(max possible: {max_possible_combinations}).",
                    stacklevel=2,
                )

            # Pack each unique sequence into source and target formats
            for seq in unique_sequences:
                reversed_seq = seq[::-1]
                self.source_texts.append(seq)
                self.target_texts.append(reversed_seq)

        # Shuffle so variable-length samples are evenly distributed across batches
        final_shuffle_gen = random.Random(seed)
        shuffle_indices = list(range(len(self.source_texts)))
        final_shuffle_gen.shuffle(shuffle_indices)

        self.source_texts = [self.source_texts[i] for i in shuffle_indices]
        self.target_texts = [self.target_texts[i] for i in shuffle_indices]

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "source_text": self.source_texts[index],
            "target_text": self.target_texts[index],
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


class StringReverseDatasetConfig(ComponentConfig):
    model_config = ConfigDict(validate_assignment=True)
    target: str = Field(
        default="trainite.datasets.string_reverse.StringReverseDataset",
        alias="_target_",
    )
    per_seq_size: int = Field(default=256, gt=0)
    charset: str | None = "@alpha"
    min_seq_len: int | None = Field(default=1, gt=0)
    max_seq_len: int | None = Field(default=16, gt=0)
    seq_len: int | None = Field(default=None, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def validate_lengths(self) -> "StringReverseDatasetConfig":
        if self.seq_len is not None and (
            self.min_seq_len is not None or self.max_seq_len is not None
        ):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if self.seq_len is None:
            if self.min_seq_len is None or self.max_seq_len is None:
                raise ValueError(
                    "Must specify either seq_len or both min_seq_len and max_seq_len."
                )
            if self.min_seq_len > self.max_seq_len:
                raise ValueError(
                    "min_seq_len must be less than or equal to max_seq_len."
                )
        return self


class StringReverseDataConfig(DataConfigBase):
    dataset: StringReverseDatasetConfig | None = Field(  # type: ignore[assignment]
        default_factory=StringReverseDatasetConfig
    )
    train_ratio: float | None = 0.8
    val_ratio: float | None = 0.1
    dataloader: DataLoaderConfig | None = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            collate_fn=None,
        )
    )
