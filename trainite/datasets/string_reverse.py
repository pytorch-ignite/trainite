import random
import string
import warnings
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

CHARSET_PRESETS = {
    "@universal": string.ascii_letters + string.digits + string.punctuation + " ",
    "@alpha": string.ascii_letters,
    "@alpha_lowercase": string.ascii_lowercase,
    "@alpha_uppercase": string.ascii_uppercase,
    "@digits": string.digits,
    "@alphanumeric": string.ascii_letters + string.digits,
    "@punctuation": string.punctuation,
    "@alphanumeric_lowercase": string.ascii_lowercase + string.digits,
    "@alphanumeric_uppercase": string.ascii_uppercase + string.digits,
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
        charset = "@universal" if charset is None else charset
        if charset in CHARSET_PRESETS:
            chars = CHARSET_PRESETS[charset]
        elif charset.startswith("@"):
            raise ValueError(f"Unknown charset preset: {charset}")
        else:
            chars = charset

        if not chars:
            raise ValueError("Charset cannot be empty.")
        if len(set(chars)) != len(chars):
            raise ValueError("Charset must not contain duplicate characters.")

        self.chars = chars

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


class DatapointModel(BaseModel):
    """Contract for a causal-LM transformed item: training tensors + the eval prompt.

    Convention: every causal-LM dataset transform returns this shape. The collate
    fn batches the `train_*`/`attention_mask` fields; the trainer's inference loop
    reads `eval_input_ids`/`source`/`target` directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class PromptCompletionTransform:
    """Converts a raw ``StringReverseDataset`` sample into training tensors.

    The string-reversal task is a sequence-to-sequence (seq2seq) task framed as
    causal language modelling (CLM): the model receives a prompt consisting of the
    source string and is trained to generate the target (reversed) string.

    Token layout (combined = [BOS] source [SEP] target [EOS]):

        input_ids : [BOS] source [SEP] target
                    (= combined[:-1])
        labels    :  -100  ...  -100   target [EOS]
                    (= combined[1:], with [BOS]+source+[SEP] masked to ignore_index)

    The model is trained to predict the full target string followed by EOS,
    given the prompt [BOS] source [SEP].  All prompt positions — including the
    trailing SEP — are masked so the loss only covers the target tokens.

    During evaluation (inference) only the prompt is fed to the model:
        eval_input_ids : [BOS] source [SEP]
    """

    def __init__(self, tokenizer: Any, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def build_prompt(self, sample: dict[str, str]) -> list[int]:
        """Token ids for the generation prompt: [bos] + source + [sep] (no target)."""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        source_tokens = self.tokenizer(sample["source"], add_special_tokens=False)["input_ids"]
        return [bos] + source_tokens + [sep]

    def __call__(self, sample: dict[str, str]) -> DatapointModel:
        source = sample["source"]
        target = sample["target"]

        eos = self.tokenizer.eos_token_id

        prompt_ids = self.build_prompt(sample)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Full sequence: prompt + target tokens + EOS
        combined_input_ids = prompt_ids + target_tokens + [eos]

        # input_ids is the sequence shifted right by 1 (teacher-forcing style)
        input_ids = torch.tensor(combined_input_ids[:-1], dtype=torch.long)
        # labels is the sequence shifted left by 1 (next-token prediction targets)
        labels = torch.tensor(combined_input_ids[1:], dtype=torch.long)

        # Mask the prompt portion of the (shifted) labels. labels = combined[1:],
        # so the prompt spans len(prompt_ids) - 1 leading positions.
        labels[: len(prompt_ids) - 1] = self.ignore_index

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=attention_mask,
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        )
