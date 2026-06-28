import random
import string
import warnings
from typing import Any

import torch
from pydantic import ConfigDict, Field, model_validator
from torch.utils.data import Dataset

from config import ComponentConfig, DataLoaderConfig, DataWithAutoSplit

# Hardcoded universal vocabulary: all printable ASCII characters
UNIVERSAL_VOCAB = string.ascii_letters + string.digits + string.punctuation + " "

CHARSET_PRESETS = {
    "@universal": UNIVERSAL_VOCAB,
    "@alpha": string.ascii_letters,
    "@digits": string.digits,
    "@alphanumeric": string.ascii_letters + string.digits,
}


class StringReverseDataset(Dataset):
    """Generates unique random strings and their reversals.

    Each sample is returned as a dictionary containing:
        - 'source': the original random string
        - 'target': the reversed string
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
        if charset is None:
            chars = UNIVERSAL_VOCAB
        elif charset in CHARSET_PRESETS:
            chars = CHARSET_PRESETS[charset]
        else:
            chars = charset

        self.chars = chars

        if not self.chars:
            raise ValueError(f"Charset '{charset}' resulted in empty characters.")

        if seq_len is not None and (min_seq_len is not None or max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if seq_len is not None:
            lengths = [seq_len]
        elif min_seq_len is None or max_seq_len is None:
            raise ValueError("Must specify either seq_len or both min_seq_len and max_seq_len.")
        else:
            lengths = list(range(min_seq_len, max_seq_len + 1))

        num_chars = len(self.chars)

        self.source_texts, self.target_texts = self.generate_unique_sequences(
            lengths=lengths,
            num_chars=num_chars,
            per_seq_size=per_seq_size,
            seed=seed,
        )

        # Shuffle so variable-length samples are evenly distributed across batches
        final_shuffle_gen = random.Random(seed)
        shuffle_indices = list(range(len(self.source_texts)))
        final_shuffle_gen.shuffle(shuffle_indices)

        self.source_texts = [self.source_texts[i] for i in shuffle_indices]
        self.target_texts = [self.target_texts[i] for i in shuffle_indices]

    def generate_unique_sequences(
        self, lengths: list[int], num_chars: int, per_seq_size: int, seed: int = 42
    ) -> tuple[list[str], list[str]]:
        # Generate per_seq_size unique sequences for each length bucket
        source_texts = []
        target_texts = []
        rng = random.Random(seed)
        for length in lengths:
            unique_sequences: set[str] = set()
            max_possible_combinations = num_chars**length
            target = min(per_seq_size, max_possible_combinations)

            # Safety cap to avoid infinite loops when the space is small
            attempts = 0
            max_attempts = target * 20

            while len(unique_sequences) < target and attempts < max_attempts:
                seq = "".join(rng.choices(self.chars, k=length))
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
                source_texts.append(seq)
                target_texts.append(reversed_seq)

        return source_texts, target_texts

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "source": self.source_texts[index],
            "target": self.target_texts[index],
        }


class PromptCompletionTransform:
    def __init__(self, tokenizer: Any, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def build_prompt(self, sample: dict[str, str]) -> list[int]:
        """Token ids for the generation prompt: [bos] + source + [sep] (no target)."""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        source_tokens = self.tokenizer(sample["source"], add_special_tokens=False)["input_ids"]
        return [bos] + source_tokens + [sep]

    def __call__(self, sample: dict[str, str]) -> dict[str, torch.Tensor]:
        source = sample["source"]
        target = sample["target"]

        eos = self.tokenizer.eos_token_id

        prompt_ids = self.build_prompt(sample)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        combined_input_ids = prompt_ids + target_tokens + [eos]

        input_ids = torch.tensor(combined_input_ids[:-1], dtype=torch.long)
        labels = torch.tensor(combined_input_ids[1:], dtype=torch.long)

        # Mask the prompt portion of the (shifted) labels. labels = combined[1:],
        # so the prompt spans len(prompt_ids) - 1 leading positions.
        labels[: len(prompt_ids) - 1] = self.ignore_index

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": source,
            "target": target,
        }


class PromptCompletionTransformConfig(ComponentConfig):
    model_config = ConfigDict(validate_assignment=True)
    target: str = Field(
        default="datasets.string_reverse.PromptCompletionTransform",
        alias="_target_",
    )
    ignore_index: int = -100


class StringReverseDatasetConfig(ComponentConfig):
    model_config = ConfigDict(validate_assignment=True)
    target: str = Field(
        default="datasets.string_reverse.StringReverseDataset",
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
        if self.seq_len is not None and (self.min_seq_len is not None or self.max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if self.seq_len is None:
            if self.min_seq_len is None or self.max_seq_len is None:
                raise ValueError("Must specify either seq_len or both min_seq_len and max_seq_len.")
            if self.min_seq_len > self.max_seq_len:
                raise ValueError("min_seq_len must be less than or equal to max_seq_len.")
        return self


class StringReverseDataConfig(DataWithAutoSplit):
    dataset: StringReverseDatasetConfig | None = Field(  # type: ignore[assignment]
        default_factory=StringReverseDatasetConfig
    )
    transform: PromptCompletionTransformConfig | None = Field(default_factory=PromptCompletionTransformConfig)
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    dataloader: DataLoaderConfig = Field(
        default_factory=lambda: DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            collate_fn=None,
        )
    )
