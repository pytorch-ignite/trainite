import pytest
from torch.utils.data import DataLoader
from trainite.datasets.counting import (
    CountingDataset,
    CountingTransform,
)
from trainite.datasets.transformed import TransformedDataset
from trainite.preprocessors.char_tokenizer import CharTokenizer
from trainite.config.registry import MODEL_SPECS
from trainite.shared.utils import get_target


def test_counting_dataset():
    total_size = 20
    k = 3
    dataset = CountingDataset(
        total_size=total_size,
        k=k,
        min_seq_len=10,
        max_seq_len=15,
        seed=42,
    )

    assert len(dataset) <= total_size
    assert len(dataset) > 0
    item = dataset[0]
    assert "source" in item
    assert "target" in item
    assert isinstance(item["source"], str)
    assert isinstance(item["target"], str)

    # Assert string length matches
    assert len(item["source"]) == len(item["target"])

    # Assert elements are valid 'a'/'b' and '0'/'1'
    assert all(c in {"a", "b"} for c in item["source"])
    assert all(c in {"0", "1"} for c in item["target"])


def test_counting_dataset_fixed_lengths():
    dataset = CountingDataset(total_size=10, k=2, seq_len=8, seed=42)
    source_lengths = [len(x) for x in dataset.source_texts]
    target_lengths = [len(x) for x in dataset.target_texts]
    assert all(s == t for s, t in zip(source_lengths, target_lengths))
    assert all(_len == 8 for _len in source_lengths)


def test_build_counting_dataset_constraints():
    with pytest.raises(
        ValueError,
        match="Cannot specify both seq_len and min_seq_len/max_seq_len.",
    ):
        CountingDataset(total_size=5, k=3, seq_len=10, min_seq_len=5, max_seq_len=15)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        CountingDataset(total_size=5, k=3)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        CountingDataset(total_size=5, k=3, min_seq_len=1)


def test_counting_transform_and_collator():
    # Setup CharTokenizer
    tokenizer = CharTokenizer()

    dataset = CountingDataset(total_size=10, k=3, min_seq_len=10, max_seq_len=15, seed=42)
    transform = CountingTransform(tokenizer=tokenizer)
    transformed_dataset = TransformedDataset(dataset, transform=transform)

    # Check transformed item
    dp = transformed_dataset[0]
    assert dp.source == dataset[0]["source"]
    assert dp.target == dataset[0]["target"]

    # Train input and label should match length + 1 (for BOS prepended)
    seq_len = len(dp.source)
    assert len(dp.train_input_ids) == seq_len + 1
    assert len(dp.train_label_ids) == seq_len + 1
    assert dp.train_input_ids[0] == tokenizer.bos_token_id
    assert dp.train_label_ids[0] == 0  # Targets prepend 0 for BOS position

    # Verify collator
    model_spec = MODEL_SPECS["basic-transformer"]
    collate_fn_cls = get_target(model_spec.collate_fn_target)
    collate_fn = collate_fn_cls(tokenizer=tokenizer)

    loader = DataLoader(transformed_dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
