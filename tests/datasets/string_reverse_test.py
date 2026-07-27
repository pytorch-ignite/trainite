import pytest
import torch
from torch.utils.data import DataLoader
from trainite.config.registry import DATASET_SPECS, MODEL_SPECS
from trainite.datasets.string_reverse import (
    CHARSET_PRESETS,
    PromptCompletionTransform,
    StringReverseDataset,
)
from trainite.datasets.transformed import TransformedDataset
from trainite.preprocessors.char_tokenizer import CharTokenizer
from trainite.shared.utils import build_dataset, get_target
from trainite.config.datasets import StringReverseDatasetConfig, StringReverseDataConfig


def test_string_reverse_dataset():
    per_seq_size = 10
    max_len = 5
    dataset = StringReverseDataset(
        per_seq_size=per_seq_size,
        min_seq_len=max_len,
        max_seq_len=max_len,
        seed=42,
    )

    assert len(dataset) == per_seq_size
    item = dataset[0]
    assert "source" in item
    assert "target" in item
    assert isinstance(item["source"], str)
    assert isinstance(item["target"], str)
    assert item["source"][::-1] == item["target"]


def test_string_reverse_variable_lengths_and_presets():
    # variable lengths when min/max_seq_len are provided
    dataset = StringReverseDataset(per_seq_size=20, min_seq_len=1, max_seq_len=8, seed=42)
    source_lengths = [len(x) for x in dataset.source_texts]
    target_lengths = [len(x) for x in dataset.target_texts]
    assert len(source_lengths) == len(target_lengths)
    assert all(s == t for s, t in zip(source_lengths, target_lengths))
    assert len(set(source_lengths)) > 1
    assert all([1 <= _len <= 8 for _len in source_lengths])


def test_string_reverse_fixed_lengths_and_presets():
    # fixed length when seq_len is provided
    dataset = StringReverseDataset(per_seq_size=20, seq_len=8, seed=42)
    source_lengths = [len(x) for x in dataset.source_texts]
    target_lengths = [len(x) for x in dataset.target_texts]
    assert all(s == t for s, t in zip(source_lengths, target_lengths))
    assert all([_len == 8 for _len in source_lengths])


@pytest.mark.parametrize(
    "preset",
    ["@digits", "@alpha", "@alphanumeric", "@universal", "abc"],
)
def test_string_reverse_dataset_charset_presets(preset: str):
    dataset = StringReverseDataset(per_seq_size=10, seed=42, charset=preset, min_seq_len=1, max_seq_len=5)
    if preset in CHARSET_PRESETS:
        assert len(dataset.chars) == len(CHARSET_PRESETS[preset])
    else:
        assert len(dataset.chars) == len(preset)


@pytest.mark.parametrize(
    ("charset", "message"),
    [("@unknown", "Unknown charset preset"), ("aab", "duplicate characters"), ("", "cannot be empty")],
)
def test_string_reverse_dataset_rejects_invalid_charset(charset: str, message: str):
    with pytest.raises(ValueError, match=message):
        StringReverseDataset(per_seq_size=5, seq_len=3, charset=charset)


def test_build_string_reverse_dataset():
    dataset = StringReverseDataset(per_seq_size=5, seq_len=3)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5

    dataset = StringReverseDataset(per_seq_size=5, min_seq_len=1, max_seq_len=5)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5 * 5  # 5 per length, 5 lengths (1..5)

    with pytest.raises(
        ValueError,
        match="Cannot specify both seq_len and min_seq_len/max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, seq_len=3, min_seq_len=1, max_seq_len=5)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, min_seq_len=1)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, max_seq_len=1)


def test_config_build_string_reverse_dataset():
    spec = DATASET_SPECS["string-reverse"]
    dataset_conf = spec.config_cls()
    tokenizer = CharTokenizer()
    dataset = build_dataset(dataset_conf.dataset, dataset_conf.transform, tokenizer)
    assert isinstance(dataset, TransformedDataset)

    model_spec = MODEL_SPECS["transformer"]
    collate_fn_obj = get_target(model_spec.collate_fn_target)(tokenizer=tokenizer)

    dataloader_conf = dataset_conf.dataloader
    dataloader_kwargs = dataloader_conf.model_dump(exclude={"collate_fn"})
    loader = DataLoader(dataset, **dataloader_kwargs, collate_fn=collate_fn_obj)

    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == dataloader_conf.batch_size
    assert batch["input_ids"].shape == batch["labels"].shape


def test_string_reverse_dataset_size_capping_and_warning():
    # Vocab size is 3 ('a', 'b', 'c'), max possible for length 1 is 3.
    # Requesting per_seq_size=10 should trigger a warning and cap the size at 3.
    with pytest.warns(
        UserWarning,
        match="Requested 10 unique sequences for seq_len=1 but only 3 could be generated",
    ):
        dataset = StringReverseDataset(
            per_seq_size=10,
            min_seq_len=1,
            max_seq_len=1,
            charset="abc",
            seed=42,
        )
    assert len(dataset) == 3


def test_charset_empty_raises():
    """Empty charset should raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        StringReverseDataset(
            per_seq_size=5,
            seq_len=3,
            charset="",
        )


def test_dataset_config_min_seq_len_gt_max():
    """Config validation: min_seq_len > max_seq_len should raise."""

    with pytest.raises(ValueError, match="min_seq_len must be less than or equal to max_seq_len"):
        StringReverseDatasetConfig(min_seq_len=10, max_seq_len=5)


def test_data_config_defaults():
    """Verify StringReverseDataConfig default values."""

    config = StringReverseDataConfig()
    assert config.test_ratio == 0.1
    assert config.val_ratio == 0.1
    assert config.dataset is not None
    assert config.dataset.per_seq_size == 256
    assert config.dataset.charset == "@alpha"
    assert config.dataloader.batch_size == 32
    assert not hasattr(config.dataloader, "collate_fn")


def test_prompt_completion_transform():
    tokenizer = CharTokenizer()
    transform = PromptCompletionTransform(tokenizer=tokenizer, ignore_index=-100)

    sample = {"source": "abc", "target": "cba"}
    item = transform(sample)

    # Contract: DatapointModel carries source/target strings + train tensors + eval prompt
    assert item.source == "abc"
    assert item.target == "cba"

    # Check tensor shapes and types
    assert item.train_input_ids.dtype == torch.long
    assert item.train_label_ids.dtype == torch.long
    assert item.attention_mask.dtype == torch.long

    # Auto-regressive shift: input_ids and labels should match, shifted by 1
    assert len(item.train_input_ids) == len(item.train_label_ids)

    # First token of input is BOS
    assert item.train_input_ids[0].item() == tokenizer.bos_token_id
    # Last token of labels is EOS
    assert item.train_label_ids[-1].item() == tokenizer.eos_token_id

    # The SEP token must exist in the labels/inputs
    sep_idx = (item.train_input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
    assert sep_idx > 0

    # Ensure label masking for the prompt part (BOS + source + SEP = 1 + 3 + 1 = 5 tokens)
    # The source tokens plus the SEP token in labels must be masked.
    # labels corresponds to combined_input_ids[1:], so the first len(source_tokens) + 1 tokens are masked.
    source_tokens = tokenizer("abc", add_special_tokens=False)["input_ids"]
    masked_len = len(source_tokens) + 1
    assert (item.train_label_ids[:masked_len] == -100).all()
    # The target tokens + EOS should NOT be masked
    assert (item.train_label_ids[masked_len:] != -100).all()

    # All attention mask values should be 1
    assert item.attention_mask.eq(1).all()

    # Eval prompt is [bos] + source + [sep], no target
    assert item.eval_input_ids.tolist() == [tokenizer.bos_token_id] + source_tokens + [tokenizer.sep_token_id]
