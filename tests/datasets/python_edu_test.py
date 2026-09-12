import gzip
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest
import torch
from datasets import Dataset as HFDataset

from trainite.datasets.python_edu import PythonEduDataset, PythonEduTransform, _download_blob


class FakeTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
    ):
        ids = [1, 10, 11, 12, 2]
        attention_mask = [1, 1, 1, 1, 1]

        if max_length is not None:
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }


def test_python_edu_transform():
    tokenizer = FakeTokenizer()
    transform = PythonEduTransform(tokenizer=tokenizer, max_length=8)

    datapoint = transform({"text": "def add(a, b):\n    return a + b"})

    assert datapoint.source == "def add(a, b):\n    return a + b"
    assert datapoint.target == "def add(a, b):\n    return a + b"

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


def test_python_edu_transform_truncates_to_max_length():
    tokenizer = FakeTokenizer()
    transform = PythonEduTransform(tokenizer=tokenizer, max_length=3)

    datapoint = transform({"text": "def add(a, b):\n    return a + b"})

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


def test_python_edu_transform_rejects_small_max_length():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="at least 2"):
        PythonEduTransform(tokenizer=tokenizer, max_length=1)


def _mock_response(content: bytes):
    response = MagicMock()
    response.read.return_value = content
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_download_blob_success():
    compressed = gzip.compress(b"print('hello')")

    with patch(
        "trainite.datasets.python_edu.urlopen",
        return_value=_mock_response(compressed),
    ) as mock_urlopen:
        result = _download_blob({"blob_id": "abc123"})

    assert result == {"text": "print('hello')", "download_ok": True}
    mock_urlopen.assert_called_once()
    called_url = mock_urlopen.call_args.args[0]
    assert called_url == "https://softwareheritage.s3.amazonaws.com/content/abc123"


def test_download_blob_http_error():
    with patch(
        "trainite.datasets.python_edu.urlopen",
        side_effect=HTTPError(url="x", code=404, msg="not found", hdrs=None, fp=None),
    ):
        result = _download_blob({"blob_id": "missing"})

    assert result == {"text": "", "download_ok": False}


def test_python_edu_dataset_filters_by_min_int_score():
    fake_ds = HFDataset.from_dict(
        {
            "blob_id": ["a", "b", "c", "d"],
            "int_score": [5, 2, 4, 3],
        }
    )

    def fake_download(example):
        return {"text": f"content-{example['blob_id']}", "download_ok": True}

    with (
        patch("trainite.datasets.python_edu.load_dataset", new=lambda *a, **k: fake_ds),
        patch("trainite.datasets.python_edu._download_blob", new=fake_download),
    ):
        dataset = PythonEduDataset(split="train", min_int_score=4, max_samples=None)

    assert len(dataset) == 2
    assert {dataset[i]["blob_id"] for i in range(len(dataset))} == {"a", "c"}


def test_python_edu_dataset_caps_max_samples():
    fake_ds = HFDataset.from_dict(
        {
            "blob_id": ["a", "b", "c"],
            "int_score": [5, 5, 5],
        }
    ).to_iterable_dataset()

    def fake_download(example):
        return {"text": f"content-{example['blob_id']}", "download_ok": True}

    with (
        patch("trainite.datasets.python_edu.load_dataset", new=lambda *a, **k: fake_ds),
        patch("trainite.datasets.python_edu._download_blob", new=fake_download),
    ):
        dataset = PythonEduDataset(split="train", min_int_score=0, max_samples=2)

    assert len(dataset) == 2


def test_python_edu_dataset_streaming_filters_by_min_int_score():
    fake_ds = HFDataset.from_dict(
        {
            "blob_id": ["a", "b", "c", "d"],
            "int_score": [5, 2, 4, 3],
        }
    ).to_iterable_dataset()

    def fake_download(example):
        return {"text": f"content-{example['blob_id']}", "download_ok": True}

    with (
        patch("trainite.datasets.python_edu.load_dataset", new=lambda *a, **k: fake_ds),
        patch("trainite.datasets.python_edu._download_blob", new=fake_download),
    ):
        dataset = PythonEduDataset(split="train", min_int_score=4, max_samples=10)

    assert len(dataset) == 2
    assert {dataset[i]["blob_id"] for i in range(len(dataset))} == {"a", "c"}


def test_python_edu_dataset_filters_out_failed_downloads():
    fake_ds = HFDataset.from_dict(
        {
            "blob_id": ["a", "b"],
            "int_score": [5, 5],
        }
    )

    def fake_download(example):
        return {"text": "ok", "download_ok": example["blob_id"] == "a"}

    with (
        patch("trainite.datasets.python_edu.load_dataset", new=lambda *a, **k: fake_ds),
        patch("trainite.datasets.python_edu._download_blob", new=fake_download),
    ):
        dataset = PythonEduDataset(split="train", min_int_score=0)

    assert len(dataset) == 1
    assert dataset[0]["blob_id"] == "a"
