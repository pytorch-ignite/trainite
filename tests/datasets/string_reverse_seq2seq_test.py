import pytest
import torch

from trainite.datasets.string_reverse_seq2seq import (
    CHARSET_PRESETS,
    UNIVERSAL_VOCAB,
    CharTokenizer,
    StringReverseDataset,
    build_string_reverse_dataset,
    collate_fn,
)


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


def test_string_reverse_seq2seq_dataset():
    size = 10
    max_len = 5
    dataset = StringReverseDataset(
        size=size, min_seq_len=max_len, max_seq_len=max_len, seed=42
    )

    assert len(dataset) == size
    item = dataset[0]
    assert "encoder_input_ids" in item
    assert "decoder_input_ids" in item
    assert "decoder_labels" in item

    # Check sizes
    # encoder_input_ids length: bos + seq(5) + eos -> 7
    # decoder_input_ids length: bos + seq(5) -> 6
    # decoder_labels length: seq(5) + eos -> 6
    assert len(item["encoder_input_ids"]) == 7
    assert len(item["decoder_input_ids"]) == 6
    assert len(item["decoder_labels"]) == 6


def test_string_reverse_seq2seq_variable_lengths():
    # variable lengths when min/max_seq_len are provided
    dataset = StringReverseDataset(size=20, min_seq_len=1, max_seq_len=8, seed=42)
    enc_lengths = [len(x) for x in dataset.encoder_input_ids]
    dec_in_lengths = [len(x) for x in dataset.decoder_input_ids]
    dec_lbl_lengths = [len(x) for x in dataset.decoder_labels]

    assert len(set(enc_lengths)) > 1
    assert len(set(dec_in_lengths)) > 1
    assert len(set(dec_lbl_lengths)) > 1

    for enc_len, dec_in_len, dec_lbl_len in zip(enc_lengths, dec_in_lengths, dec_lbl_lengths):
        # enc_len: seq_len + 2
        # dec_in_len: seq_len + 1
        # dec_lbl_len: seq_len + 1
        assert enc_len == dec_in_len + 1
        assert dec_in_len == dec_lbl_len
        assert 3 <= enc_len <= 10  # seq_len is 1 to 8, so enc_len is 3 to 10


def test_string_reverse_seq2seq_fixed_lengths():
    # fixed length when seq_len is provided
    dataset = StringReverseDataset(size=20, seq_len=8, seed=42)
    enc_lengths = [len(x) for x in dataset.encoder_input_ids]
    dec_in_lengths = [len(x) for x in dataset.decoder_input_ids]
    dec_lbl_lengths = [len(x) for x in dataset.decoder_labels]

    assert all(enc_len == 10 for enc_len in enc_lengths)
    assert all(dec_in_len == 9 for dec_in_len in dec_in_lengths)
    assert all(dec_lbl_len == 9 for dec_lbl_len in dec_lbl_lengths)


@pytest.mark.parametrize(
    "preset",
    ["@digits", "@alpha", "@alphanumeric", "@universal", "abc"],
)
def test_string_reverse_seq2seq_dataset_charset_presets(preset: str):
    dataset = StringReverseDataset(
        size=10, seed=42, charset=preset, min_seq_len=1, max_seq_len=5
    )
    if preset in CHARSET_PRESETS:
        assert len(dataset.valid_token_ids) == len(CHARSET_PRESETS[preset])
    else:
        assert len(dataset.valid_token_ids) == len(preset)


def test_collate_fn():
    batch = [
        {
            "encoder_input_ids": torch.tensor([1, 4, 5, 2]),  # len 4
            "decoder_input_ids": torch.tensor([1, 5, 4]),     # len 3
            "decoder_labels": torch.tensor([5, 4, 2]),        # len 3
        },
        {
            "encoder_input_ids": torch.tensor([1, 6, 2]),     # len 3
            "decoder_input_ids": torch.tensor([1, 6]),        # len 2
            "decoder_labels": torch.tensor([6, 2]),           # len 2
        }
    ]

    collated = collate_fn(batch)
    assert collated["encoder_input_ids"].shape == (2, 4)
    assert collated["decoder_input_ids"].shape == (2, 3)
    assert collated["decoder_labels"].shape == (2, 3)

    # Label padding value should be -100
    assert collated["decoder_labels"][1, -1].item() == -100
    # Input padding value should be 0
    assert collated["decoder_input_ids"][1, -1].item() == 0


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
