from pathlib import Path
from typing import cast

import pytest

from trainite.config.datasets import HuggingFaceDatasetConfig
from trainite.datasets.hugging_face import HuggingFaceTransform
from trainite.datasets.transformed import TransformedDataset
from trainite.shared.utils import build_dataset  # pyright: ignore[reportUnknownVariableType]


def test_load_hugging_face_dataset(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    _ = data_file.write_text('{"text": "example"}\n')

    config = HuggingFaceDatasetConfig.model_validate({"path": "json", "data_files": str(data_file)})
    dataset = cast(TransformedDataset, build_dataset(config, transform_config=None, tokenizer=object()))

    assert isinstance(dataset, TransformedDataset)
    assert dataset[0] == {"text": "example"}


def test_hugging_face_transform_requires_implementation() -> None:
    with pytest.raises(NotImplementedError, match="data/hugging_face.py"):
        _ = HuggingFaceTransform(tokenizer=object())({"text": "example"})
