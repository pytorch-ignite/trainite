from typing import Any

import pytest
import torch

from trainite.datasets.ultrachat_200k import UltraChat200kTransform


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int = 128,
    ) -> dict[str, Any]:
        self.calls.append(text)
        words = text.strip().split()
        ids = [1] + [10 + i for i in range(len(words))] + [2]
        attention_mask = [1] * len(ids)

        if truncation and max_length is not None:
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }


def test_ultrachat_transform():
    tokenizer = FakeTokenizer()
    transform = UltraChat200kTransform(tokenizer=tokenizer, max_length=16)

    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "I am good."},
        ]
    }

    datapoint = transform(sample)

    assert datapoint.source == ("User: hello\nAssistant: world\nUser: how are you?\nAssistant: ")
    assert datapoint.target == "I am good."

    assert datapoint.train_input_ids.dtype == torch.long
    assert datapoint.train_label_ids.dtype == torch.long
    assert datapoint.attention_mask.dtype == torch.long
    assert datapoint.eval_input_ids.dtype == torch.long

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=torch.long),
    )
    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 2], dtype=torch.long),
    )
    assert torch.equal(
        datapoint.attention_mask,
        torch.tensor([1] * 13, dtype=torch.long),
    )
    assert torch.equal(
        datapoint.eval_input_ids,
        torch.tensor([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2], dtype=torch.long),
    )

    # Full conversation was tokenized for training
    assert tokenizer.calls[0] == ("User: hello\nAssistant: world\nUser: how are you?\nAssistant: I am good.\n")


def test_ultrachat_transform_user_ended_conversation():
    tokenizer = FakeTokenizer()
    transform = UltraChat200kTransform(tokenizer=tokenizer, max_length=16)

    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "final question"},
        ]
    }

    datapoint = transform(sample)

    assert datapoint.source == ("User: hello\nAssistant: world\nUser: final question\nAssistant: ")
    assert datapoint.target == ""


def test_ultrachat_transform_truncates_to_max_length():
    tokenizer = FakeTokenizer()
    transform = UltraChat200kTransform(tokenizer=tokenizer, max_length=3)

    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }

    datapoint = transform(sample)

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10], dtype=torch.long),
    )
    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11], dtype=torch.long),
    )
    assert torch.equal(
        datapoint.attention_mask,
        torch.tensor([1, 1], dtype=torch.long),
    )


def test_ultrachat_transform_rejects_small_max_length():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="at least 2"):
        UltraChat200kTransform(tokenizer=tokenizer, max_length=1)


def test_ultrachat_transform_rejects_invalid_messages():
    tokenizer = FakeTokenizer()
    transform = UltraChat200kTransform(tokenizer=tokenizer, max_length=16)

    with pytest.raises(ValueError, match="messages must be a list"):
        transform({"messages": "not a list"})

    with pytest.raises(ValueError, match="each message in messages must be a dict"):
        transform({"messages": [{"role": "user", "content": "hi"}, "invalid"]})
