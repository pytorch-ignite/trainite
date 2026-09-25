from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


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


class UltraChat200kTransform:
    """Converts an UltraChat 200k multi-turn conversation into training tensors.

    UltraChat 200k samples contain a ``messages`` list of role/content dictionaries
    representing multi-turn dialogues. Each turn is rendered into standard plaintext
    as ``User: <content>\\n`` or ``Assistant: <content>\\n``.

    Token layout & supervised loss masking:
        In conversational SFT (Supervised Fine-Tuning), the model is trained to predict
        assistant responses given the conversational history, without computing loss on
        user queries or role headers.

        input_ids : all conversation tokens except the last one (= token_ids[:-1])
        labels    : full conversation tokens shifted left by 1 (= token_ids[1:]),
                    where all user message tokens and role headers are masked with ``ignore_index`` (-100).
                    Only assistant response tokens are active loss targets.

    During evaluation (inference):
        The evaluation prompt preserves conversational context up to the final user turn and
        ends with ``Assistant:``,
        with ``target`` containing the reference assistant response:
            eval_input_ids : tokenized prompt ending with "Assistant:"
    """

    def __init__(self, tokenizer: Any, max_length: int = 128, ignore_index: int = -100) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.tokenizer: Any = tokenizer
        self.max_length: int = max_length
        self.ignore_index: int = ignore_index

    @staticmethod
    def _render_message(message: dict[str, Any]) -> str:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        return f"{role}: {content}\n"

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        return "".join(self._render_message(m) for m in messages)

    def build_eval_prompt(self, messages: list[dict[str, Any]]) -> tuple[str, str]:
        """Extract generation prompt and reference target from conversation history."""
        if messages and str(messages[-1].get("role", "")).strip().lower() == "assistant":
            eval_messages = messages[:-1]
            target = str(messages[-1].get("content", "")).strip()
        else:
            eval_messages = messages
            target = ""

        eval_prompt = f"{self._render_messages(eval_messages)}Assistant:"
        return eval_prompt, target

    def __call__(self, sample: dict[str, Any]) -> DatapointModel:
        messages = sample.get("messages")

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        if not messages:
            raise ValueError("messages must not be empty")

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

        # input_ids is the sequence shifted right by 1 (teacher-forcing style)
        train_input_ids = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )
        # labels is the sequence shifted left by 1 (next-token prediction targets)
        # Start with all positions masked to ignore_index (-100)
        train_label_ids = torch.full(
            (len(token_ids) - 1,),
            self.ignore_index,
            dtype=torch.long,
        )
        train_attention_mask = torch.tensor(
            attention_mask[:-1],
            dtype=torch.long,
        )

        # Unmask only assistant response tokens so loss is computed solely on assistant replies.
        # Determine whether the tokenizer prepended a special token (e.g. BOS).
        first_msg_tokens = self.tokenizer(self._render_message(messages[0]), add_special_tokens=False)["input_ids"]
        curr_offset = 1 if (first_msg_tokens and len(token_ids) > 0 and token_ids[0] != first_msg_tokens[0]) else 0

        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            rendered = self._render_message(message)
            msg_tokens = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            msg_len = len(msg_tokens)

            if role == "assistant":
                header = f"{str(message.get('role', 'assistant')).strip().capitalize()}:"
                header_tokens = self.tokenizer(header, add_special_tokens=False)["input_ids"]
                header_len = len(header_tokens)

                # Unmask tokens belonging to the assistant's content
                for idx in range(curr_offset + header_len, curr_offset + msg_len):
                    label_idx = idx - 1
                    if 0 <= label_idx < len(train_label_ids):
                        train_label_ids[label_idx] = token_ids[idx]

            curr_offset += msg_len
            if curr_offset >= len(token_ids):
                break

        eval_prompt, target = self.build_eval_prompt(messages)

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
