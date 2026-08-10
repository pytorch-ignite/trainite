import random
import warnings
from typing import Any
import math

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset


class CountingDataset(Dataset):
    """Generates unique alternating strings in L_k and their prefix-classification targets.

    Each sample is returned as a dictionary containing:
        - 'source': the generated string of 'a's and 'b's (e.g., 'aaabbb')
        - 'target': the target string of '0's and '1's (e.g., '000111')
    """

    def __init__(
        self,
        total_size: int,
        k: int,
        min_seq_len: int | None = None,
        max_seq_len: int | None = None,
        seq_len: int | None = None,
        seed: int = 42,
    ) -> None:
        self.k = k

        if seq_len is not None and (min_seq_len is not None or max_seq_len is not None):
            raise ValueError("Cannot specify both seq_len and min_seq_len/max_seq_len.")

        if seq_len is not None:
            min_len = seq_len
            max_len = seq_len
        elif min_seq_len is None or max_seq_len is None:
            raise ValueError("Must specify either seq_len or both min_seq_len and max_seq_len.")
        else:
            min_len = min_seq_len
            max_len = max_seq_len

        self.source_texts, self.target_texts = self.generate_unique_sequences(
            total_size=total_size,
            min_len=min_len,
            max_len=max_len,
            k=k,
            seed=seed,
        )

        # Shuffle so variable-length samples are evenly distributed across batches
        final_shuffle_gen = random.Random(seed)
        shuffle_indices = list(range(len(self.source_texts)))
        final_shuffle_gen.shuffle(shuffle_indices)

        self.source_texts = [self.source_texts[i] for i in shuffle_indices]
        self.target_texts = [self.target_texts[i] for i in shuffle_indices]

    def generate_ab(self, n: int, k: int, rng: random.Random) -> tuple[str, str]:
        """
        L_1 = a^+
        L_{k+1} = L_k b^+ if k odd, L_{k+1} = L_k a^+ if k even
        """
        if k == 1:
            return "a" * n, "1" * n

        # Select k-1 distinct switching indices from 1 to n-1
        switching_indices = set(rng.sample(range(1, n), k - 1))

        # Target has '0' before the last switch, and '1' after the last switch
        last_switch = max(switching_indices)
        string_tgt = "0" * last_switch + "1" * (n - last_switch)

        string_src = ""
        gen_a = True
        for i in range(n):
            if i in switching_indices:
                gen_a = not gen_a
            string_src += "a" if gen_a else "b"

        return string_src, string_tgt

    def generate_unique_sequences(
        self, total_size: int, min_len: int, max_len: int, k: int, seed: int = 42
    ) -> tuple[list[str], list[str]]:
        source_texts = []
        target_texts = []
        rng = random.Random(seed)

        # We can't generate if max length is less than k
        if max_len < k:
            return [], []

        unique_sequences: set[str] = set()
        seq_map: dict[str, str] = {}

        # Pre-calculate possible combinations to avoid infinite loops on small ranges
        total_possible = sum(math.comb(seq_len - 1, k - 1) for seq_len in range(min_len, max_len + 1) if seq_len >= k)
        target_size = min(total_size, total_possible)

        # Safety cap to avoid infinite loops
        attempts = 0
        max_attempts = target_size * 50

        while len(unique_sequences) < target_size and attempts < max_attempts:
            length = rng.randint(min_len, max_len)
            if length < k:
                attempts += 1
                continue
            src, tgt = self.generate_ab(length, k, rng)
            if src not in unique_sequences:
                unique_sequences.add(src)
                seq_map[src] = tgt
            attempts += 1

        if len(unique_sequences) < target_size:
            warnings.warn(
                f"Requested {target_size} unique sequences but only {len(unique_sequences)} could be generated.",
                stacklevel=2,
            )

        for seq in unique_sequences:
            source_texts.append(seq)
            target_texts.append(seq_map[seq])

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


class CountingTransform:
    def __init__(self, tokenizer: Any, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, sample: dict[str, str]) -> DatapointModel:
        source = sample["source"]
        target = sample["target"]

        # Map source string to token IDs using tokenizer
        source_tokens = self.tokenizer(source, add_special_tokens=False)["input_ids"]

        # Prepend BOS token
        bos = self.tokenizer.bos_token_id
        input_ids = [bos] + source_tokens

        # Targets are binary class labels (0 and 1)
        # Prepend 0 for the BOS position
        target_tokens = [int(char) for char in target]
        label_ids = [0] + target_tokens

        train_input_ids = torch.tensor(input_ids, dtype=torch.long)
        train_label_ids = torch.tensor(label_ids, dtype=torch.long)
        attention_mask = torch.ones(len(train_input_ids), dtype=torch.long)

        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=train_input_ids,
            train_label_ids=train_label_ids,
            attention_mask=attention_mask,
            eval_input_ids=train_input_ids,
        )
