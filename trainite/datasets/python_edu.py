from typing import Any

import gzip
from urllib.error import HTTPError
from urllib.request import urlopen

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

_CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"


def _download_blob(example: dict[str, Any]) -> dict[str, Any]:
    """Fetch and decode one file's content from Software Heritage's public S3 bucket.

    ``python-edu`` ships only file metadata (``blob_id``, ``repo_name``, ``path``,
    ``length_bytes``, ``score``, ``int_score``) — the source is downloaded
    separately, gzip-compressed, from a public (no-credentials) bucket.
    """
    url = _CONTENT_URL.format(blob_id=example["blob_id"])
    try:
        with urlopen(url, timeout=30) as response:
            content = gzip.decompress(response.read()).decode("utf-8", errors="ignore")
        return {"text": content, "download_ok": True}
    except HTTPError:
        return {"text": "", "download_ok": False}


class PythonEduDataset(Dataset):
    """Educational Python source files scored by HuggingFaceTB's code classifier."""

    def __init__(
        self,
        split: str = "train",
        min_int_score: int = 4,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None:
            # Streaming avoids materializing/downloading the full ~600MB
            # metadata parquet just to pull a handful of rows for a quick
            # local smoke test — rows are pulled lazily until `max_samples`
            # pass the score filter and download successfully.
            stream = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split=split, streaming=True)
            if min_int_score > 0:
                stream = stream.filter(lambda ex: ex["int_score"] >= min_int_score)
            stream = stream.map(_download_blob)
            stream = stream.filter(lambda ex: ex["download_ok"])
            rows = list(stream.take(max_samples))
            self._dataset = HFDataset.from_list(rows)
        else:
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split=split)
            if min_int_score > 0:
                ds = ds.filter(lambda ex: ex["int_score"] >= min_int_score)
            ds = ds.map(_download_blob, num_proc=1)
            ds = ds.filter(lambda ex: ex["download_ok"])
            self._dataset = ds

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


class DatapointModel(BaseModel):
    """Tokenized Python-Edu sample used by Trainite's causal LM pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class PythonEduTransform:
    """Convert a downloaded Python-Edu sample into a causal LM datapoint."""

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
