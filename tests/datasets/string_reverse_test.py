import pytest
import torch
from torch.utils.data import DataLoader

from trainite.config.registry import get_dataset_spec
from trainite.datasets.string_reverse import (
    CHARSET_PRESETS,
    StringReverseDataset,
)
from trainite.shared.utils import get_target, instantiate
from trainite.tokenizers.char import CharTokenizer


def test_string_reverse_dataset():
    per_seq_size = 10
    max_len = 5
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(
        per_seq_size=per_seq_size,
        min_seq_len=max_len,
        max_seq_len=max_len,
        tokenizer=tokenizer,
        seed=42,
    )

    assert len(dataset) == per_seq_size
    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "attention_mask" in item

    # Check tensor dtypes
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long

    # Verify reversal logic via decode (strip BOS/SEP/EOS)
    decoded_source = tokenizer.decode(item["input_ids"].tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.decode(item["labels"].tolist(), skip_special_tokens=True)
    assert decoded_source[::-1] == decoded_labels


def test_string_reverse_variable_lengths_and_presets():
    # variable lengths when min/max_seq_len are provided
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(per_seq_size=20, min_seq_len=1, max_seq_len=8, tokenizer=tokenizer, seed=42)
    source_lengths = [len(x) for x in dataset.source_texts]
    target_lengths = [len(x) for x in dataset.target_texts]
    assert len(source_lengths) == len(target_lengths)
    assert all(s == t for s, t in zip(source_lengths, target_lengths))
    assert len(set(source_lengths)) > 1
    assert all([1 <= _len <= 8 for _len in source_lengths])


def test_string_reverse_fixed_lengths_and_presets():
    # fixed length when seq_len is provided
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(per_seq_size=20, seq_len=8, tokenizer=tokenizer, seed=42)
    source_lengths = [len(x) for x in dataset.source_texts]
    target_lengths = [len(x) for x in dataset.target_texts]
    assert all(s == t for s, t in zip(source_lengths, target_lengths))
    assert all([_len == 8 for _len in source_lengths])


@pytest.mark.parametrize(
    "preset",
    ["@digits", "@alpha", "@alphanumeric", "@universal", "abc"],
)
def test_string_reverse_dataset_charset_presets(preset: str):
    dataset = StringReverseDataset(
        per_seq_size=10, seed=42, charset=preset, min_seq_len=1, max_seq_len=5, tokenizer=CharTokenizer()
    )
    if preset in CHARSET_PRESETS:
        assert len(dataset.chars) == len(CHARSET_PRESETS[preset])
    else:
        assert len(dataset.chars) == len(preset)


def test_build_string_reverse_dataset():
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(per_seq_size=5, seq_len=3, tokenizer=tokenizer)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5

    dataset = StringReverseDataset(per_seq_size=5, min_seq_len=1, max_seq_len=5, tokenizer=tokenizer)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5 * 5  # 5 per length, 5 lengths (1..5)

    with pytest.raises(
        ValueError,
        match="Cannot specify both seq_len and min_seq_len/max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, seq_len=3, min_seq_len=1, max_seq_len=5, tokenizer=tokenizer)

    tokenizer = CharTokenizer()

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, tokenizer=tokenizer)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, min_seq_len=1, tokenizer=tokenizer)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        StringReverseDataset(per_seq_size=5, max_seq_len=1, tokenizer=tokenizer)


def test_config_build_string_reverse_dataset():
    spec = get_dataset_spec("string-reverse")
    dataset_conf = spec.config_cls()
    dataset = instantiate(dataset_conf.dataset, tokenizer=CharTokenizer())
    assert isinstance(dataset, StringReverseDataset)

    from trainite.config.registry import get_model_spec

    model_spec = get_model_spec("transformer")
    tokenizer = CharTokenizer()
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
    tokenizer = CharTokenizer()
    with pytest.warns(
        UserWarning,
        match="Requested 10 unique sequences for seq_len=1 but only 3 could be generated",
    ):
        dataset = StringReverseDataset(
            per_seq_size=10,
            min_seq_len=1,
            max_seq_len=1,
            charset="abc",
            tokenizer=tokenizer,
            seed=42,
        )
    assert len(dataset) == 3


def test_getitem_tensor_structure():
    """Verify the exact structure of __getitem__ output for a known input."""
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(
        per_seq_size=10,
        min_seq_len=2,
        max_seq_len=2,
        charset="ab",
        tokenizer=tokenizer,
        seed=42,
    )
    item = dataset[0]

    # input_ids and labels are offset by 1 (autoregressive shift)
    assert len(item["input_ids"]) == len(item["labels"])
    assert torch.equal(item["input_ids"][1:], item["labels"][:-1])
    assert len(item["attention_mask"]) == len(item["input_ids"])

    # First token is BOS
    assert item["input_ids"][0].item() == tokenizer.bos_token_id
    # Last label is EOS
    assert item["labels"][-1].item() == tokenizer.eos_token_id

    # SEP token exists between source and target in labels
    sep_positions = (item["labels"] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    assert len(sep_positions) == 1

    # All attention mask values are 1 (no padding within a sample)
    assert item["attention_mask"].eq(1).all()

    # Source text appears before SEP in input_ids
    sep_idx = (item["input_ids"] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
    source_ids = item["input_ids"][1:sep_idx].tolist()  # skip BOS
    target_ids = item["input_ids"][sep_idx + 1 :].tolist()  # after SEP
    assert source_ids == tokenizer.encode(tokenizer.decode(target_ids, skip_special_tokens=True)[::-1])


def test_get_item_inference():
    """Verify get_item_inference returns prompt-only tensors."""
    tokenizer = CharTokenizer()
    dataset = StringReverseDataset(
        per_seq_size=10,
        seq_len=5,
        tokenizer=tokenizer,
        seed=42,
    )
    inf = dataset.get_item_inference(0)

    assert "input_ids" in inf
    assert "attention_mask" in inf
    assert "source_text" in inf
    assert "target_text" in inf

    assert inf["input_ids"].dtype == torch.long
    assert inf["attention_mask"].dtype == torch.long

    # Starts with BOS, ends with SEP, no target tokens
    assert inf["input_ids"][0].item() == tokenizer.bos_token_id
    assert inf["input_ids"][-1].item() == tokenizer.sep_token_id
    assert inf["attention_mask"].eq(1).all()

    # target_text matches the reversal of source_text
    assert inf["source_text"][::-1] == inf["target_text"]

    # Length matches: BOS + source_tokens + SEP
    source_encoded = tokenizer.encode(inf["source_text"])
    assert len(inf["input_ids"]) == 1 + len(source_encoded) + 1


def test_charset_empty_raises():
    """Empty charset should raise ValueError."""
    tokenizer = CharTokenizer()
    with pytest.raises(ValueError, match="resulted in empty characters"):
        StringReverseDataset(
            per_seq_size=5,
            seq_len=3,
            charset="",
            tokenizer=tokenizer,
        )


def test_dataset_config_min_seq_len_gt_max():
    """Config validation: min_seq_len > max_seq_len should raise."""
    from trainite.datasets.string_reverse import StringReverseDatasetConfig

    with pytest.raises(ValueError, match="min_seq_len must be less than or equal to max_seq_len"):
        StringReverseDatasetConfig(min_seq_len=10, max_seq_len=5)


def test_data_config_defaults():
    """Verify StringReverseDataConfig default values."""
    from trainite.datasets.string_reverse import StringReverseDataConfig

    config = StringReverseDataConfig()
    assert config.test_ratio == 0.1
    assert config.val_ratio == 0.1
    assert config.dataset is not None
    assert config.dataset.per_seq_size == 256
    assert config.dataset.charset == "@alpha"
    assert config.dataloader.batch_size == 32
    assert config.dataloader.collate_fn is None
