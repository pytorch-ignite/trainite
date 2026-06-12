from unittest import mock

import pytest
import torch

from trainite.config import get_model_spec
from trainite.datasets.string_reverse import CharTokenizer
from trainite.models.transformer import (
    Attention,
    CausalLMCollateFn,
    RotaryEmbedding,
    TransformerBlock,
    TransformerModel,
)
from trainite.utils import instantiate


def test_rotary_embedding():
    dim = 8
    max_seq_len = 16
    rope = RotaryEmbedding(dim, max_seq_len)

    # Test inside cache boundary
    seq_len = 10
    x = torch.zeros(1, 2, seq_len, dim)
    cos, sin = rope(x, seq_len)
    assert cos.shape == (1, 1, seq_len, dim)
    assert sin.shape == (1, 1, seq_len, dim)
    assert cos.abs().max() <= 1.0
    assert sin.abs().max() <= 1.0

    # Test outside cache boundary (fallback dynamic computation)
    seq_len_large = 24
    x_large = torch.zeros(1, 2, seq_len_large, dim)
    cos_l, sin_l = rope(x_large, seq_len_large)
    assert cos_l.shape == (1, 1, seq_len_large, dim)
    assert sin_l.shape == (1, 1, seq_len_large, dim)
    assert cos_l.abs().max() <= 1.0
    assert sin_l.abs().max() <= 1.0

    # Test odd dimension throws ValueError
    with pytest.raises(ValueError, match="RotaryEmbedding dimension.*must be even"):
        RotaryEmbedding(7, max_seq_len)


def test_attention():
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads
    attn = Attention(embed_dim, num_heads)
    x = torch.randn(2, 10, embed_dim)

    rope = RotaryEmbedding(head_dim, 16)
    cos, sin = rope(x, seq_len=10)

    # Test causal attention (default when padding_mask is None)
    attn.eval()
    output, context = attn(x, cos, sin)
    assert output.shape == x.shape

    # Test with padding mask
    # padding_mask shape (B, 1, 1, S)
    padding_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
    padding_mask[:, :, :, -2:] = False  # Mask last 2 tokens
    output, context = attn(x, cos, sin, padding_mask=padding_mask)
    assert output.shape == x.shape

    # Non-divisible embed_dim should raise an assertion error
    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads."):
        Attention(15, 2)


def test_transformer_block():
    d_model = 16
    num_heads = 2
    feedforward_dim = 32
    head_dim = d_model // num_heads
    block = TransformerBlock(d_model, num_heads, feedforward_dim)
    x = torch.randn(2, 10, d_model)

    rope = RotaryEmbedding(head_dim, 16)
    cos, sin = rope(x, seq_len=10)

    output = block(x, cos, sin)
    assert output.shape == x.shape


def test_attention_context_and_dropout_behavior():
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads
    # use noticeable dropout so behavior is observable
    attn = Attention(embed_dim, num_heads, dropout=0.5)
    x = torch.randn(2, 10, embed_dim)

    rope = RotaryEmbedding(head_dim, 16)
    cos, sin = rope(x, seq_len=10)

    # context shape should match (B, S, C)
    output, context = attn(x, cos, sin)
    assert context.shape == x.shape

    # Dropout should be disabled in eval mode (deterministic outputs across different seeds)
    attn.eval()
    torch.manual_seed(0)
    out1, _ = attn(x, cos, sin)
    torch.manual_seed(1)
    out2, _ = attn(x, cos, sin)
    assert torch.allclose(out1, out2)

    # In train mode with different seeds outputs are expected to differ due to dropout randomness
    attn.train()
    torch.manual_seed(0)
    out3, _ = attn(x, cos, sin)
    torch.manual_seed(1)
    out4, _ = attn(x, cos, sin)
    # It's extremely unlikely these are exactly equal when dropout is active
    assert not torch.allclose(out3, out4)


def test_transformer_model():
    vocab_size = 10
    hidden_size = 16
    model = TransformerModel(vocab_size=vocab_size, hidden_size=hidden_size)

    input_ids = torch.randint(1, vocab_size, (2, 10))
    model.eval()
    output = model(input_ids)
    # output shape (B, S, vocab_size)
    assert output.shape == (2, 10, vocab_size)

    # Test with padding
    input_ids[0, -2:] = 0  # padding_idx=0
    output = model(input_ids)
    assert output.shape == (2, 10, vocab_size)


def test_build_transformer_model():
    model = TransformerModel(vocab_size=10, hidden_size=16)
    assert isinstance(model, TransformerModel)


def test_build_transformer_model_from_spec():
    spec = get_model_spec("transformer")
    model_conf = spec.config_cls()
    model = instantiate(model_conf)
    assert isinstance(model, TransformerModel)


def test_transformer_model_generate():
    tokenizer = CharTokenizer()
    hidden_size = 16
    model = TransformerModel(vocab_size=tokenizer.vocab_size, hidden_size=hidden_size)
    model.eval()

    # Test with string prompt and tokenizer
    with mock.patch.object(model, "forward") as mock_forward:

        def mock_forward_fn(x):
            logits = torch.zeros(1, x.shape[1], tokenizer.vocab_size)
            logits[0, -1, 7] = 10.0
            return logits

        mock_forward.side_effect = mock_forward_fn

        generated = model.generate(["ab"], max_new_tokens=1, tokenizer=tokenizer)
        assert isinstance(generated, list)
        assert generated[0] == "c"

    # Test with eos_token_id early exit
    with mock.patch.object(model, "forward") as mock_forward:

        def mock_forward_fn(x):
            logits = torch.zeros(1, x.shape[1], tokenizer.vocab_size)
            logits[0, -1, tokenizer.eos_token_id] = 10.0
            return logits

        mock_forward.side_effect = mock_forward_fn

        generated = model.generate(
            ["ab"],
            max_new_tokens=10,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
        )
        assert generated[0] == ""

    # Test with eos_token_id early exit (default tokenizer fallback)
    with mock.patch.object(model, "forward") as mock_forward:

        def mock_forward_fn(x):
            logits = torch.zeros(1, x.shape[1], tokenizer.vocab_size)
            logits[0, -1, tokenizer.eos_token_id] = 10.0
            return logits

        mock_forward.side_effect = mock_forward_fn

        generated = model.generate(
            ["ab"],
            max_new_tokens=10,
            tokenizer=tokenizer,
        )
        assert generated[0] == ""


def test_causal_lm_collate_fn():
    tokenizer = CharTokenizer()
    collate = CausalLMCollateFn(tokenizer=tokenizer, pad_token_id=0, ignore_index=-100)

    data1 = {"source_text": "abc", "target_text": "cba"}
    data2 = {"source_text": "d", "target_text": "d"}

    batch = [data1, data2]

    collated = collate(batch)
    assert "input_ids" in collated
    assert "labels" in collated

    assert collated["input_ids"].ndim == 2
    assert collated["labels"].ndim == 2
    assert collated["input_ids"].shape == collated["labels"].shape

    # Max sequence length:
    # "abc" (3) -> src is <bos> abc <eos> (5 tokens). target is cba <eos> (4 tokens). total full_seq = 9. input_ids = 8. labels = 8.
    # "d" (1) -> src is <bos> d <eos> (3 tokens). target is d <eos> (2 tokens). total full_seq = 5. input_ids = 4. labels = 4.
    # So max length is 8.
    assert collated["input_ids"].shape == (2, 8)
    assert (collated["input_ids"][1, :4] == 0).all()
