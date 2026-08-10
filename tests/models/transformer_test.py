import pytest
import torch
from trainite.config.registry import MODEL_SPECS
from trainite.datasets.string_reverse import DatapointModel
from trainite.models.basic_transformer import (
    Attention as BasicAttention,
)
from trainite.models.basic_transformer import (
    BasicTransformerModel,
)
from trainite.models.basic_transformer import (
    CausalLMCollateFn as BasicCausalLMCollateFn,
)
from trainite.models.basic_transformer import (
    TransformerBlock as BasicTransformerBlock,
)
from trainite.models.rope_transformer import (
    Attention as RoPEAttention,
)
from trainite.models.rope_transformer import (
    CausalLMCollateFn as RoPECausalLMCollateFn,
)
from trainite.models.rope_transformer import (
    RoPETransformerModel,
    RotaryEmbedding,
)
from trainite.models.rope_transformer import (
    TransformerBlock as RoPETransformerBlock,
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


@pytest.mark.parametrize("attn_cls,uses_rope", [(BasicAttention, False), (RoPEAttention, True)])
def test_attention_forward(attn_cls, uses_rope):
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads
    attn = attn_cls(embed_dim, num_heads)
    x = torch.randn(2, 10, embed_dim)

    kwargs = {}
    if uses_rope:
        rope = RotaryEmbedding(head_dim, 16)
        cos, sin = rope(x, seq_len=10)
        kwargs = {"cos": cos, "sin": sin}

    # Test causal attention
    attn.eval()
    output, context = attn(x, **kwargs)
    assert output.shape == x.shape

    # Test with padding mask
    padding_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
    padding_mask[:, :, :, -2:] = False
    output, context = attn(x, padding_mask=padding_mask, **kwargs)
    assert output.shape == x.shape

    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads."):
        attn_cls(15, 2)


@pytest.mark.parametrize("block_cls,uses_rope", [(BasicTransformerBlock, False), (RoPETransformerBlock, True)])
def test_transformer_block_forward(block_cls, uses_rope):
    d_model = 16
    num_heads = 2
    feedforward_dim = 32
    head_dim = d_model // num_heads
    block = block_cls(d_model, num_heads, feedforward_dim)
    x = torch.randn(2, 10, d_model)

    kwargs = {}
    if uses_rope:
        rope = RotaryEmbedding(head_dim, 16)
        cos, sin = rope(x, seq_len=10)
        kwargs = {"cos": cos, "sin": sin}

    output = block(x, **kwargs)
    assert output.shape == x.shape


@pytest.mark.parametrize("attn_cls,uses_rope", [(BasicAttention, False), (RoPEAttention, True)])
def test_attention_context_and_dropout_behavior(attn_cls, uses_rope):
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads
    attn = attn_cls(embed_dim, num_heads, dropout=0.5)
    x = torch.randn(2, 10, embed_dim)

    kwargs = {}
    if uses_rope:
        rope = RotaryEmbedding(head_dim, 16)
        cos, sin = rope(x, seq_len=10)
        kwargs = {"cos": cos, "sin": sin}

    output, context = attn(x, **kwargs)
    assert context.shape == x.shape

    attn.eval()
    torch.manual_seed(0)
    out1, _ = attn(x, **kwargs)
    torch.manual_seed(1)
    out2, _ = attn(x, **kwargs)
    assert torch.allclose(out1, out2)

    attn.train()
    torch.manual_seed(0)
    out3, _ = attn(x, **kwargs)
    torch.manual_seed(1)
    out4, _ = attn(x, **kwargs)
    assert not torch.allclose(out3, out4)


@pytest.mark.parametrize("model_cls", [BasicTransformerModel, RoPETransformerModel])
def test_transformer_model_forward(model_cls):
    vocab_size = 10
    hidden_size = 16
    model = model_cls(vocab_size=vocab_size, hidden_size=hidden_size)

    input_ids = torch.randint(1, vocab_size, (2, 10))
    model.eval()
    output = model(input_ids)
    assert output.shape == (2, 10, vocab_size)

    # Test with attention mask / padding
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    attention_mask[0, :3] = 0
    input_ids[0, :3] = 0

    output = model(input_ids, attention_mask=attention_mask)
    assert output.shape == (2, 10, vocab_size)


@pytest.mark.parametrize(
    "spec_name,expected_cls",
    [("basic-transformer", BasicTransformerModel), ("rope-transformer", RoPETransformerModel)],
)
def test_build_transformer_models_from_specs(spec_name, expected_cls):
    spec = MODEL_SPECS[spec_name]
    model_conf = spec.config_cls()
    model = instantiate(model_conf)
    assert isinstance(model, expected_cls)


@pytest.mark.parametrize("collate_cls", [BasicCausalLMCollateFn, RoPECausalLMCollateFn])
def test_causal_lm_collate_fn(collate_cls):
    tokenizer = CharTokenizer()
    collate = collate_cls(tokenizer=tokenizer)

    src1 = "abc"
    tgt1 = "cba"
    src2 = "d"
    tgt2 = "d"

    def make_item(source: str, target: str) -> DatapointModel:
        source_ids = tokenizer.encode(source)
        target_ids = tokenizer.encode(target)
        combined = (
            [tokenizer.bos_token_id] + source_ids + [tokenizer.sep_token_id] + target_ids + [tokenizer.eos_token_id]
        )
        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=torch.tensor(combined[:-1], dtype=torch.long),
            train_label_ids=torch.tensor(combined[1:], dtype=torch.long),
            attention_mask=torch.ones(len(combined) - 1, dtype=torch.long),
            eval_input_ids=torch.tensor(
                [tokenizer.bos_token_id] + source_ids + [tokenizer.sep_token_id], dtype=torch.long
            ),
        )

    batch = [make_item(src1, tgt1), make_item(src2, tgt2)]

    collated = collate(batch)
    assert "input_ids" in collated
    assert "labels" in collated

    assert collated["input_ids"].ndim == 2
    assert collated["labels"].ndim == 2
    assert collated["input_ids"].shape == collated["labels"].shape
    assert collated["input_ids"].shape == (2, 8)
    assert (collated["input_ids"][1, :4] == tokenizer.pad_token_id).all()
