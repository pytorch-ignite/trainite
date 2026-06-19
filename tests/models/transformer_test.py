import pytest
import torch
from trainite.config.registry import get_model_spec
from trainite.models.transformer import (
    Attention,
    CausalLMCollateFn,
    RotaryEmbedding,
    TransformerBlock,
    TransformerModel,
)
from trainite.preprocessors.char_tokenizer import CharTokenizer
from trainite.shared.utils import instantiate


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


def test_causal_lm_collate_fn():
    tokenizer = CharTokenizer()
    collate = CausalLMCollateFn(tokenizer=tokenizer, pad_token_id=0, ignore_index=-100)

    # Create pre-tokenized items as the dataset now produces
    src1 = "abc"
    tgt1 = "cba"
    src2 = "d"
    tgt2 = "d"

    def make_item(source: str, target: str) -> dict:
        source_ids = tokenizer.encode(source)
        target_ids = tokenizer.encode(target)
        combined = (
            [tokenizer.bos_token_id] + source_ids + [tokenizer.sep_token_id] + target_ids + [tokenizer.eos_token_id]
        )
        return {
            "input_ids": torch.tensor(combined[:-1], dtype=torch.long),
            "labels": torch.tensor(combined[1:], dtype=torch.long),
            "attention_mask": torch.ones(len(combined) - 1, dtype=torch.long),
        }

    batch = [make_item(src1, tgt1), make_item(src2, tgt2)]

    collated = collate(batch)
    assert "input_ids" in collated
    assert "labels" in collated

    assert collated["input_ids"].ndim == 2
    assert collated["labels"].ndim == 2
    assert collated["input_ids"].shape == collated["labels"].shape

    # Max sequence length:
    # "abc" (3) -> <bos>abc<sep>cba<eos> = 9 tokens, input_ids=8, labels=8
    # "d" (1) -> <bos>d<sep>d<eos> = 5 tokens, input_ids=4, labels=4
    # Max length = 8, shorter padded on left
    assert collated["input_ids"].shape == (2, 8)
    assert (collated["input_ids"][1, :4] == tokenizer.pad_token_id).all()
