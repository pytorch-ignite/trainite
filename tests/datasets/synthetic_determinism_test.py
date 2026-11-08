import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("dataset_name", ["string_reverse", "counting"])
def test_synthetic_dataset_mapping_and_splits_across_hash_seeds(dataset_name):
    script = """
import json
import sys

import torch
from torch.utils.data import random_split
from trainite.datasets.counting import CountingDataset
from trainite.datasets.string_reverse import StringReverseDataset

if sys.argv[1] == "string_reverse":
    dataset = StringReverseDataset(per_seq_size=20, seq_len=5, charset="abc", seed=42)
else:
    dataset = CountingDataset(total_size=20, k=3, min_seq_len=5, max_seq_len=8, seed=42)

samples = [dataset[i] for i in range(len(dataset))]
assert len(samples) == 20
assert len({sample["source"] for sample in samples}) == 20
assert dataset.source_texts == [sample["source"] for sample in samples]
assert dataset.target_texts == [sample["target"] for sample in samples]
if sys.argv[1] == "string_reverse":
    assert all(len(sample["source"]) == 5 for sample in samples)
    assert all(sample["target"] == sample["source"][::-1] for sample in samples)
else:
    assert len({len(sample["source"]) for sample in samples}) > 1
    for sample in samples:
        source = sample["source"]
        assert 5 <= len(source) <= 8
        assert source[0] == "a" and set(source) <= {"a", "b"}
        switches = [i for i in range(1, len(source)) if source[i] != source[i - 1]]
        assert len(switches) == 2
        last_switch = switches[-1]
        assert sample["target"] == "0" * last_switch + "1" * (len(source) - last_switch)

train, val = random_split(dataset, [16, 4], generator=torch.Generator().manual_seed(123))
train_samples = [train[i] for i in range(len(train))]
val_samples = [val[i] for i in range(len(val))]
assert {sample["source"] for sample in train_samples}.isdisjoint(
    sample["source"] for sample in val_samples
)
print(json.dumps({"samples": samples, "train": train_samples, "val": val_samples}))
"""
    results = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, dataset_name],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        results.append(json.loads(result.stdout))
    assert results[0] == results[1]
