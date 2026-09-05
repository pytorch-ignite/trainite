import pytest
import torch

from trainite.datasets.wikitext import WikiTextTransform


class FakeTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
    ):
        # The tokenizer is responsible for adding special tokens.
        ids = [1, 10, 11, 12, 2]
        attention_mask = [1, 1, 1, 1, 1]

        if max_length is not None:
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }


def test_wikitext_transform():
    tokenizer = FakeTokenizer()
    transform = WikiTextTransform(tokenizer=tokenizer, max_length=8)

    datapoint = transform({"text": "hello world"})

    assert datapoint.source == "hello world"
    assert datapoint.target == "hello world"

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10, 11, 12]),
    )
    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11, 12, 2]),
    )
    assert torch.equal(
        datapoint.attention_mask,
        torch.tensor([1, 1, 1, 1]),
    )
    assert torch.equal(
        datapoint.eval_input_ids,
        torch.tensor([1, 10, 11, 12]),
    )


def test_wikitext_transform_truncates_to_max_length():
    tokenizer = FakeTokenizer()
    transform = WikiTextTransform(tokenizer=tokenizer, max_length=3)

    datapoint = transform({"text": "hello world"})

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10]),
    )
    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11]),
    )
    assert torch.equal(
        datapoint.attention_mask,
        torch.tensor([1, 1]),
    )
    assert torch.equal(
        datapoint.eval_input_ids,
        torch.tensor([1, 10]),
    )


def test_wikitext_transform_rejects_small_max_length():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="at least 2"):
        WikiTextTransform(tokenizer=tokenizer, max_length=1)
