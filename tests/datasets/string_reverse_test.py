import pytest
from torch.utils.data import DataLoader

from trainite.config import get_dataset_spec
from trainite.datasets.string_reverse import (
    CHARSET_PRESETS,
    UNIVERSAL_VOCAB,
    CharTokenizer,
    StringReverseDataset,
)
from trainite.utils import get_target, instantiate


def test_char_tokenizer():
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size == len(UNIVERSAL_VOCAB) + len(tokenizer.special_tokens)

    encoded = tokenizer.encode("abc")
    assert encoded == [5, 6, 7]

    decoded = tokenizer.decode(encoded)
    assert decoded == "abc"

    # Test special tokens
    assert (
        tokenizer.decode([0, 1, 2, 3, 4], skip_special_tokens=False)
        == "<PAD><BOS><SEP><EOS><UNK>"
    )
    assert tokenizer.decode([0, 1, 2, 3, 4], skip_special_tokens=True) == "<UNK>"

    # Test unk
    assert tokenizer.encode("Δ") == [
        4
    ]  # Δ is not in the universal vocab, should map to UNK token ID 3
    assert tokenizer.decode([4]) == "<UNK>"


def test_string_reverse_dataset():
    per_seq_size = 10
    max_len = 5
    dataset = StringReverseDataset(
        per_seq_size=per_seq_size, min_seq_len=max_len, max_seq_len=max_len, seed=42
    )

    assert len(dataset) == per_seq_size
    item = dataset[0]
    assert "source_text" in item
    assert "target_text" in item

    source_text = item["source_text"]
    target_text = item["target_text"]

    # seq len is 5.
    assert len(source_text) == 5
    assert len(target_text) == 5

    # Verify reversal logic
    assert source_text[::-1] == target_text


def test_string_reverse_variable_lengths_and_presets():
    # variable lengths when min/max_seq_len are provided
    dataset = StringReverseDataset(
        per_seq_size=20, min_seq_len=1, max_seq_len=8, seed=42
    )
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
    dataset = StringReverseDataset(
        per_seq_size=10, seed=42, charset=preset, min_seq_len=1, max_seq_len=5
    )
    if preset in CHARSET_PRESETS:
        assert len(dataset.valid_token_ids) == len(CHARSET_PRESETS[preset])
    else:
        assert len(dataset.valid_token_ids) == len(preset)


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
    spec = get_dataset_spec("string-reverse")
    dataset_conf = spec.config_cls()
    dataset = instantiate(dataset_conf.dataset)
    assert isinstance(dataset, StringReverseDataset)

    from trainite.config.registry import get_model_spec

    model_spec = get_model_spec("transformer")
    collate_fn_obj = get_target(model_spec.collate_fn_target)(
        tokenizer=dataset.tokenizer
    )

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
            per_seq_size=10, min_seq_len=1, max_seq_len=1, charset="abc", seed=42
        )
    assert len(dataset) == 3
