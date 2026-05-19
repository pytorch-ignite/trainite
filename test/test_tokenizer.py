from __future__ import annotations

import torch

from trainite.tokenizers.char_tokenizer import ALPHABET_PRESETS, CharTokenizer, build_tokenizer


def test_char_tokenizer_encode_decode_roundtrip() -> None:
    tokenizer = CharTokenizer(alphabet="abc")
    encoded = tokenizer.encode("abcba")
    decoded = tokenizer.decode(encoded)
    assert decoded == "abcba"


def test_char_tokenizer_encode_values() -> None:
    tokenizer = CharTokenizer(alphabet="abc")
    assert tokenizer.encode("abc") == [1, 2, 3]


def test_char_tokenizer_decode_skips_padding() -> None:
    tokenizer = CharTokenizer(alphabet="abc")
    # 0 is the padding token and should be skipped
    assert tokenizer.decode([1, 0, 2, 3]) == "abc"


def test_char_tokenizer_decode_tensor() -> None:
    tokenizer = CharTokenizer(alphabet="abc")
    ids = torch.tensor([1, 2, 3, 0])
    assert tokenizer.decode(ids) == "abc"


def test_char_tokenizer_vocab_size() -> None:
    tokenizer = CharTokenizer(alphabet="abc")
    assert tokenizer.vocab_size == 3


def test_char_tokenizer_pad_token_id() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.pad_token_id == 0


def test_char_tokenizer_alphabet_preset_alpha() -> None:
    tokenizer = CharTokenizer(alphabet="@alpha")
    assert tokenizer.alphabet == "abcdefghijklmnopqrstuvwxyz"
    assert tokenizer.vocab_size == 26


def test_char_tokenizer_alphabet_preset_digits() -> None:
    tokenizer = CharTokenizer(alphabet="@digits")
    assert tokenizer.alphabet == "0123456789"
    assert tokenizer.vocab_size == 10


def test_char_tokenizer_alphabet_preset_alphanumeric() -> None:
    tokenizer = CharTokenizer(alphabet="@alphanumeric")
    assert tokenizer.alphabet == "abcdefghijklmnopqrstuvwxyz0123456789"
    assert tokenizer.vocab_size == 36


def test_char_tokenizer_custom_alphabet() -> None:
    tokenizer = CharTokenizer(alphabet="xyz")
    assert tokenizer.alphabet == "xyz"
    assert tokenizer.vocab_size == 3
    assert tokenizer.encode("xyz") == [1, 2, 3]


def test_build_tokenizer() -> None:
    tokenizer = build_tokenizer(alphabet="@alpha")
    assert isinstance(tokenizer, CharTokenizer)
    assert tokenizer.vocab_size == 26
