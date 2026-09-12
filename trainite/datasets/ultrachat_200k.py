from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Tokenized UltraChat 200k sample used by Trainite's causal LM pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class UltraChat200kTransform:
    """Convert an UltraChat 200k sample into a causal language-modeling datapoint.

    Multi-turn conversation messages are formatted into a standard text prompt.
    The formatted conversation is used as the autoregressive training sequence.

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

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> str:
        return "".join(
            f"{str(message.get('role', 'user')).strip().capitalize()}: {str(message.get('content', '')).strip()}\n"
            for message in messages
        )

    def __call__(self, sample: dict[str, Any]) -> DatapointModel:
        messages = sample.get("messages")

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message in messages must be a dict")

        conversation = self._render_messages(messages)

        tokenized = self.tokenizer(
            conversation,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        token_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        train_input_ids = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )
        train_label_ids = torch.tensor(
            token_ids[1:],
            dtype=torch.long,
        )
        train_attention_mask = torch.tensor(
            attention_mask[:-1],
            dtype=torch.long,
        )

        if messages and str(messages[-1].get("role", "")).strip().lower() == "assistant":
            eval_messages = messages[:-1]
            target = str(messages[-1].get("content", "")).strip()
        else:
            eval_messages = messages
            target = ""

        eval_prompt = f"{self._render_messages(eval_messages)}Assistant: "

        eval_tokenized = self.tokenizer(
            eval_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        return DatapointModel(
            source=eval_prompt,
            target=target,
            train_input_ids=train_input_ids,
            train_label_ids=train_label_ids,
            attention_mask=train_attention_mask,
            eval_input_ids=torch.tensor(
                eval_tokenized["input_ids"],
                dtype=torch.long,
            ),
        )
