from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Tokenized WikiText sample used by Trainite's causal LM pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class WikiTextTransform:
    """Convert a WikiText sample into a causal language-modeling datapoint.

    WikiText samples contain a single ``text`` field. The complete text is
    used as the autoregressive training sequence.

    Special tokens are added by the tokenizer.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        text = str(sample["text"])

        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        token_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )
        labels = torch.tensor(
            token_ids[1:],
            dtype=torch.long,
        )
        train_attention_mask = torch.tensor(
            attention_mask[:-1],
            dtype=torch.long,
        )

        return DatapointModel(
            source=text,
            target=text,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=train_attention_mask,
            eval_input_ids=input_ids,
        )
