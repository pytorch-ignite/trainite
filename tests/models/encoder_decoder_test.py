import pytest
import torch

from trainite.config import get_model_spec
from trainite.models.encoder_decoder import (
    PositionalEncoding,
    MultiHeadAttention,
    EncoderBlock,
    DecoderBlock,
    EncoderDecoderModel,
    build_encoder_decoder_model,
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


def test_multi_head_attention():
    embed_dim = 16
    num_heads = 2
    attn = MultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(2, 10, embed_dim)

    # Self-attention (eval mode)
    attn.eval()
    output = attn(x)
    assert output.shape == x.shape

    # Self-attention with padding mask
    padding_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
    padding_mask[:, :, :, -2:] = False
    output = attn(x, padding_mask=padding_mask)
    assert output.shape == x.shape

    # Cross-attention (memory has different sequence length)
    memory = torch.randn(2, 8, embed_dim)
    output = attn(x, memory=memory)
    assert output.shape == x.shape

    # Non-divisible embed_dim should raise ValueError
    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads."):
        MultiHeadAttention(15, 2)


def test_multi_head_attention_dropout_behavior():
    embed_dim = 16
    num_heads = 2
    attn = MultiHeadAttention(embed_dim, num_heads, dropout=0.5)
    x = torch.randn(2, 10, embed_dim)

    # Dropout should be disabled in eval mode (deterministic outputs across different seeds)
    attn.eval()
    torch.manual_seed(0)
    out1 = attn(x)
    torch.manual_seed(1)
    out2 = attn(x)
    assert torch.allclose(out1, out2)

    # In train mode with different seeds outputs are expected to differ due to dropout randomness
    attn.train()
    torch.manual_seed(0)
    out3 = attn(x)
    torch.manual_seed(1)
    out4 = attn(x)
    assert not torch.allclose(out3, out4)


def test_encoder_block():
    d_model = 16
    num_heads = 2
    feedforward_dim = 32
    block = EncoderBlock(d_model, num_heads, feedforward_dim)
    x = torch.randn(2, 10, d_model)
    output = block(x)
    assert output.shape == x.shape


def test_decoder_block():
    d_model = 16
    num_heads = 2
    feedforward_dim = 32
    block = DecoderBlock(d_model, num_heads, feedforward_dim)
    x = torch.randn(2, 10, d_model)
    memory = torch.randn(2, 8, d_model)
    output = block(x, memory)
    assert output.shape == x.shape


def test_encoder_decoder_model():
    vocab_size = 10
    hidden_size = 16
    
    model = EncoderDecoderModel(vocab_size=vocab_size, hidden_size=hidden_size)

    enc_input_ids = torch.randint(1, vocab_size, (2, 8))
    dec_input_ids = torch.randint(1, vocab_size, (2, 10))

    model.eval()
    output = model(enc_input_ids, dec_input_ids)
    assert output.shape == (2, 10, vocab_size)

    enc_input_ids[0, -2:] = 0
    dec_input_ids[0, -2:] = 0
    output = model(enc_input_ids, dec_input_ids)
    assert output.shape == (2, 10, vocab_size)

    model_custom = EncoderDecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_encoder_layers=3,
        num_decoder_layers=1,
        encoder_max_seq_len=64,
        decoder_max_seq_len=32,
    )
    assert len(model_custom.encoder_blocks) == 3
    assert len(model_custom.decoder_blocks) == 1
    
    enc_ids_custom = torch.randint(1, vocab_size, (2, 40))
    dec_ids_custom = torch.randint(1, vocab_size, (2, 20))
    model_custom.eval()
    output_custom = model_custom(enc_ids_custom, dec_ids_custom)
    assert output_custom.shape == (2, 20, vocab_size)


def test_build_encoder_decoder_model():
    model = build_encoder_decoder_model(vocab_size=10, hidden_size=16)
    assert isinstance(model, EncoderDecoderModel)


def test_build_encoder_decoder_model_from_spec():
    spec = get_model_spec("encoder-decoder")
    model_conf = spec.config_cls()
    model = instantiate(model_conf)
    assert isinstance(model, EncoderDecoderModel)
