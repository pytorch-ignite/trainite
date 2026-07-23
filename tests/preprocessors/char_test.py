import torch
from trainite.preprocessors import CharTokenizer, CharTokenizerConfig
import string


def test_char_tokenizer_vocab_size():
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size == len(string.printable) + len(tokenizer.special_tokens)


def test_char_tokenizer_encode():
    tokenizer = CharTokenizer()
    encoded = tokenizer.encode("abc")
    assert encoded == [15, 16, 17]


def test_char_tokenizer_decode():
    tokenizer = CharTokenizer()
    encoded = tokenizer.encode("abc")
    decoded = tokenizer.decode(encoded)
    assert decoded == "abc"


def test_char_tokenizer_special_tokens():
    tokenizer = CharTokenizer()
    # With special tokens visible
    assert tokenizer.decode([0, 1, 2, 3, 4], skip_special_tokens=False) == "<PAD><BOS><SEP><EOS><UNK>"
    # With special tokens skipped (UNK is always shown)
    assert tokenizer.decode([0, 1, 2, 3, 4], skip_special_tokens=True) == "<UNK>"


def test_char_tokenizer_unk():
    tokenizer = CharTokenizer()
    # Δ is not in the vocab, should map to UNK token ID 4
    assert tokenizer.encode("Δ") == [4]
    assert tokenizer.decode([4]) == "<UNK>"


def test_char_tokenizer_call_single():
    tokenizer = CharTokenizer()
    result = tokenizer("hello", add_special_tokens=True)
    assert "input_ids" in result
    assert "attention_mask" in result
    # <BOS> h e l l o <EOS> = 7 tokens
    assert len(result["input_ids"]) == 7
    assert len(result["attention_mask"]) == 7
    assert result["input_ids"][0] == tokenizer.bos_token_id
    assert result["input_ids"][-1] == tokenizer.eos_token_id
    assert all(m == 1 for m in result["attention_mask"])


def test_char_tokenizer_call_batch():
    tokenizer = CharTokenizer()
    result = tokenizer(["hi", "hello"], add_special_tokens=True, padding=True)
    assert isinstance(result["input_ids"], list)
    assert isinstance(result["attention_mask"], list)
    # Both sequences should be the same length (max of padded lengths)
    assert len(result["input_ids"][0]) == len(result["input_ids"][1])
    assert len(result["attention_mask"][0]) == len(result["attention_mask"][1])
    # Shorter "hi" should have padding on the left
    padded_len = len(result["input_ids"][0])
    assert result["attention_mask"][0][: padded_len - 4] == [0] * (padded_len - 4)


def test_char_tokenizer_call_return_tensors():
    tokenizer = CharTokenizer()
    result = tokenizer("abc", return_tensors="pt")
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].ndim == 2  # Batched as (1, seq_len)


def test_char_tokenizer_call_truncation():
    tokenizer = CharTokenizer()
    result = tokenizer("hello world this is a long string", add_special_tokens=True, truncation=True, max_length=5)
    # Should be truncated to max_length (including special tokens)
    assert len(result["input_ids"]) == 5


def test_char_tokenizer_call_no_special_tokens():
    tokenizer = CharTokenizer()
    result = tokenizer("abc", add_special_tokens=False)
    assert len(result["input_ids"]) == 3
    assert result["input_ids"] == [15, 16, 17]


def test_char_tokenizer_call_max_length_padding():
    tokenizer = CharTokenizer()
    result = tokenizer("abc", add_special_tokens=True, padding="max_length", max_length=10)
    # <BOS> a b c <EOS> = 5 tokens, padded to 10
    assert len(result["input_ids"]) == 10
    # Padding should be on the left
    assert result["input_ids"][:5] == [tokenizer.pad_token_id] * 5
    assert result["attention_mask"][:5] == [0] * 5


def test_char_tokenizer_config():
    config = CharTokenizerConfig()
    assert config.target == "trainite.preprocessors.char_tokenizer.CharTokenizer"
