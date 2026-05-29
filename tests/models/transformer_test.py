import pytest
import torch

from trainite.config import get_model_spec
from trainite.models.transformer import (
    Attention,
    PositionalEncoding,
    TransformerBlock,
    TransformerModel,
    build_transformer_model,
)
from trainite.utils import instantiate


def test_positional_encoding():
    d_model = 16
    max_len = 32
    pe = PositionalEncoding(d_model, max_len)
    x = torch.zeros(1, 10, d_model)
    output = pe(x)
    assert output.shape == x.shape
    assert not torch.allclose(output, x)

    d_model = 15
    max_len = 32
    with pytest.raises(
        ValueError, match="d_model must be even for sinusoidal positional encoding."
    ):
        PositionalEncoding(d_model, max_len)


def test_attention():
    embed_dim = 16
    num_heads = 2
    attn = Attention(embed_dim, num_heads)
    x = torch.randn(2, 10, embed_dim)

    # Test causal attention (default when padding_mask is None)
    attn.eval()
    output, context = attn(x)
    assert output.shape == x.shape

    # Test with padding mask
    # padding_mask shape (B, 1, 1, S)
    padding_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
    padding_mask[:, :, :, -2:] = False  # Mask last 2 tokens
    output, context = attn(x, padding_mask=padding_mask)
    assert output.shape == x.shape

    # Non-divisible embed_dim should raise an assertion error
    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads."):
        Attention(15, 2)


def test_transformer_block():
    d_model = 16
    num_heads = 2
    feedforward_dim = 32
    block = TransformerBlock(d_model, num_heads, feedforward_dim)
    x = torch.randn(2, 10, d_model)
    output = block(x)
    assert output.shape == x.shape


def test_attention_context_and_dropout_behavior():
    embed_dim = 16
    num_heads = 2
    # use noticeable dropout so behavior is observable
    attn = Attention(embed_dim, num_heads, dropout=0.5)
    x = torch.randn(2, 10, embed_dim)

    # context shape should match (B, S, C)
    output, context = attn(x)
    assert context.shape == x.shape

    # Dropout should be disabled in eval mode (deterministic outputs across different seeds)
    attn.eval()
    torch.manual_seed(0)
    out1, _ = attn(x)
    torch.manual_seed(1)
    out2, _ = attn(x)
    assert torch.allclose(out1, out2)

    # In train mode with different seeds outputs are expected to differ due to dropout randomness
    attn.train()
    torch.manual_seed(0)
    out3, _ = attn(x)
    torch.manual_seed(1)
    out4, _ = attn(x)
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
    model = build_transformer_model(vocab_size=10, hidden_size=16)
    assert isinstance(model, TransformerModel)


def test_build_transformer_model_from_spec():
    spec = get_model_spec("transformer")
    model_conf = spec.config_cls()
    model = instantiate(model_conf)
    assert isinstance(model, TransformerModel)
