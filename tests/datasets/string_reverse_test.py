import pytest
import torch
from torch.utils.data import DataLoader

from trainite.config import DataLoaderConfig, get_dataset_spec
from trainite.datasets.string_reverse import (
    CHARSET_PRESETS,
    UNIVERSAL_VOCAB,
    CharTokenizer,
    StringReverseDataset,
    build_string_reverse_dataset,
    collate_fn,
)
from trainite.utils import get_target, instantiate


def test_char_tokenizer():
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size == len(UNIVERSAL_VOCAB) + 4  # +4 for special tokens

    encoded = tokenizer.encode("abc")
    assert encoded == [4, 5, 6]

    decoded = tokenizer.decode(encoded)
    assert decoded == "abc"

    # Test special tokens
    assert (
        tokenizer.decode([0, 1, 2, 3], skip_special_tokens=False)
        == "<pad><bos><eos><unk>"
    )
    assert tokenizer.decode([0, 1, 2, 3], skip_special_tokens=True) == "<unk>"

    # Test unk
    assert tokenizer.encode("Δ") == [
        3
    ]  # Δ is not in the universal vocab, should map to UNK token ID 3
    assert tokenizer.decode([3]) == "<unk>"


def test_string_reverse_dataset():
    size = 10
    max_len = 5
    dataset = StringReverseDataset(
        size=size, min_seq_len=max_len, max_seq_len=max_len, seed=42
    )

    assert len(dataset) == size
    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item

    input_ids = item["input_ids"]
    labels = item["labels"]

    # Length: BOS + seq + EOS + reversed_seq + EOS - 1 (for labels shift)
    # seq len is 5.
    # full_seq: [BOS, c1, c2, c3, c4, c5, EOS, c5, c4, c3, c2, c1, EOS] -> length 13
    # input_ids: full_seq[:-1] -> length 12
    # labels: full_seq[1:] -> length 12
    assert len(input_ids) == 12
    assert len(labels) == 12

    # Verify reversal logic
    # input_ids: [BOS, c1, c2, c3, c4, c5, EOS, c5, c4, c3, c2, c1]
    # labels:    [c1,  c2, c3, c4, c5, EOS, c5, c4, c3, c2, c1, EOS]
    # prompt_len = 5 + 2 = 7 ([BOS, c1, c2, c3, c4, c5, EOS])
    # labels[:prompt_len-1] should be -100 (indices 0 to 5)
    assert (labels[:6] == -100).all()
    assert labels[6].item() != -100  # First char of reversed seq


def test_string_reverse_variable_lengths_and_presets():
    # variable lengths when min/max_seq_len are provided
    dataset = StringReverseDataset(size=20, min_seq_len=1, max_seq_len=8, seed=42)
    input_lengths = [len(x) for x in dataset.inputs]
    label_lengths = [len(x) for x in dataset.labels]
    assert input_lengths == label_lengths
    assert len(set(input_lengths)) > 1
    assert all(
        [4 <= _len <= 18 for _len in input_lengths]
    )  # (BOS + seq(8) + EOS + reversed_seq(8) + EOS) - 1 (labels shift)


def test_string_reverse_fixed_lengths_and_presets():
    # fixed length when seq_len is provided
    dataset = StringReverseDataset(size=20, seq_len=8, seed=42)
    input_lengths = [len(x) for x in dataset.inputs]
    label_lengths = [len(x) for x in dataset.labels]
    assert input_lengths == label_lengths
    assert all(
        [
            _len == 18 for _len in input_lengths
        ]  # (BOS + seq(8) + EOS + reversed_seq(8) + EOS) - 1 (labels shift)
    )


@pytest.mark.parametrize(
    "preset",
    ["@digits", "@alpha", "@alphanumeric", "@universal", "abc"],
)
def test_string_reverse_dataset_charset_presets(preset: str):
    dataset = StringReverseDataset(
        size=10, seed=42, charset=preset, min_seq_len=1, max_seq_len=5
    )
    if preset in CHARSET_PRESETS:
        assert len(dataset.valid_token_ids) == len(CHARSET_PRESETS[preset])
    else:
        assert len(dataset.valid_token_ids) == len(preset)


def test_collate_fn():
    tokenizer = CharTokenizer()
    encoded1 = torch.tensor(tokenizer.encode("abc"))
    encoded2 = torch.tensor(tokenizer.encode("d"))

    data1 = {"input_ids": encoded1, "labels": encoded1.flip(0)}
    data2 = {"input_ids": encoded2, "labels": encoded2.flip(0)}

    batch = [data1, data2]

    collated = collate_fn(batch)
    assert "input_ids" in collated
    assert "labels" in collated

    assert collated["input_ids"].ndim == 2
    assert collated["labels"].ndim == 2
    assert collated["input_ids"].shape == collated["labels"].shape


def test_build_string_reverse_dataset():
    dataset = build_string_reverse_dataset(size=5, seq_len=3)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5

    dataset = build_string_reverse_dataset(size=5, min_seq_len=1, max_seq_len=5)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == 5

    with pytest.raises(
        ValueError,
        match="Cannot specify both seq_len and min_seq_len/max_seq_len.",
    ):
        build_string_reverse_dataset(size=5, seq_len=3, min_seq_len=1, max_seq_len=5)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        build_string_reverse_dataset(size=5)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        build_string_reverse_dataset(size=5, min_seq_len=1)

    with pytest.raises(
        ValueError,
        match="Must specify either seq_len or both min_seq_len and max_seq_len.",
    ):
        build_string_reverse_dataset(size=5, max_seq_len=1)


def test_config_build_string_reverse_dataset():
    spec = get_dataset_spec("string-reverse")
    dataset_conf = spec.config_cls()
    dataset = instantiate(dataset_conf)
    assert isinstance(dataset, StringReverseDataset)
    assert len(dataset) == dataset_conf.size

    collate_fn_obj = None
    if spec.collate_fn_symbol:
        module_path = str(spec.implementation_path.with_suffix("")).replace("/", ".")
        collate_fn_obj = get_target(f"{module_path}.{spec.collate_fn_symbol}")

    dataloader_conf = DataLoaderConfig(batch_size=4)
    dataloader_kwargs = dataloader_conf.model_dump(exclude={"collate_fn"})
    loader = DataLoader(dataset, **dataloader_kwargs, collate_fn=collate_fn_obj)

    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 4
