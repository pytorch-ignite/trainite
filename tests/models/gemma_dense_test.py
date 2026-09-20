from types import SimpleNamespace

import torch

from trainite.config.registry import MODEL_SPECS
from trainite.models.gemma4_moe import (
    CausalLMCollateFn,
    DenseMLP,
    Gemma4DenseBlock,
    Gemma4DenseModel,
    Gemma4TextAttention,
    apply_text_rope,
)
from trainite.shared.utils import instantiate


def test_dense_mlp():
    hidden_size = 32
    intermediate_size = 64
    mlp = DenseMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    x = torch.randn(2, 4, hidden_size)
    out = mlp(x)
    assert out.shape == (2, 4, hidden_size)


def test_apply_text_rope_full_and_partial():
    head_dim = 16
    x = torch.randn(2, 4, 2, head_dim)
    position_ids = torch.arange(4).unsqueeze(0).repeat(2, 1)

    # Full RoPE (100%)
    out_full = apply_text_rope(x, position_ids, theta=10000.0, rotary_fraction=1.0)
    assert out_full.shape == x.shape

    # Partial RoPE (25%)
    out_part = apply_text_rope(x, position_ids, theta=1000000.0, rotary_fraction=0.25)
    assert out_part.shape == x.shape

    # Default rotary_fraction fallback to 1.0
    out_default = apply_text_rope(x, position_ids, theta=10000.0)
    assert torch.allclose(out_full, out_default, atol=1e-6)


def test_gemma4_text_attention_forward():
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    attn = Gemma4TextAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=4,
    )
    x = torch.randn(2, 8, hidden_size)
    position_ids = torch.arange(8).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, :2] = 0

    out = attn(x, position_ids=position_ids, attention_mask=mask)
    assert out.shape == (2, 8, hidden_size)


def test_gemma4_text_attention_key_equals_value():
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    attn = Gemma4TextAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        key_equals_value=True,
    )
    assert attn.v_proj is None
    assert attn.v_norm.weight is None  # unscaled RMSNorm

    x = torch.randn(2, 4, hidden_size)
    position_ids = torch.arange(4).unsqueeze(0).repeat(2, 1)
    out = attn(x, position_ids=position_ids)
    assert out.shape == (2, 4, hidden_size)


def test_gemma4_dense_block_forward():
    hidden_size = 32
    intermediate_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    block = Gemma4DenseBlock(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        dense_intermediate_size=intermediate_size,
        head_dim=head_dim,
    )
    x = torch.randn(2, 6, hidden_size)
    position_ids = torch.arange(6).unsqueeze(0).repeat(2, 1)

    out = block(x, position_ids=position_ids)
    assert out.shape == (2, 6, hidden_size)


def test_gemma4_dense_model_forward():
    vocab_size = 50
    hidden_size = 32
    num_layers = 4
    model = Gemma4DenseModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dense_intermediate_size=64,
        sliding_ratio=2,
        final_logit_softcap=30.0,
        tie_word_embeddings=True,
    )

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask[0, :2] = 0

    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Gradients backward check
    loss = logits.sum()
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_gemma4_dense_model_untied():
    vocab_size = 50
    hidden_size = 32
    model = Gemma4DenseModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        dense_intermediate_size=64,
        tie_word_embeddings=False,
    )
    input_ids = torch.randint(0, vocab_size, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, vocab_size)


def test_gemma4_dense_model_global_key_equals_value():
    vocab_size = 50
    hidden_size = 32
    # Layer 0 is sliding, Layer 1 is global (since last layer is always global)
    model = Gemma4DenseModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        dense_intermediate_size=64,
        sliding_ratio=5,
        global_key_equals_value=True,
    )
    # Sliding layer has separate v_proj
    assert model.layers[0].attention.v_proj is not None
    # Global layer reuses key (v_proj is None)
    assert model.layers[1].attention.v_proj is None

    input_ids = torch.randint(0, vocab_size, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, vocab_size)


def test_causal_lm_collate_fn():
    tokenizer = SimpleNamespace(pad_token_id=0)
    collate_fn = CausalLMCollateFn(tokenizer)

    batch_items = [
        SimpleNamespace(
            train_input_ids=torch.tensor([1, 2, 3]),
            train_label_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
        ),
        SimpleNamespace(
            train_input_ids=torch.tensor([4, 5]),
            train_label_ids=torch.tensor([4, 5]),
            attention_mask=torch.tensor([1, 1]),
        ),
    ]

    batch = collate_fn(batch_items)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)

    # Check left-padding on second sequence
    assert batch["input_ids"][1, 0].item() == 0
    assert batch["attention_mask"][1, 0].item() == 0
    assert batch["labels"][1, 0].item() == -100


def test_gemma4_dense_instantiate_from_config():
    spec = MODEL_SPECS["gemma4-dense"]
    config_cls = spec.config_cls
    config = config_cls()

    model = instantiate(config, vocab_size=64, pad_token_id=0)
    assert isinstance(model, Gemma4DenseModel)

    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 64)


def test_gemma_dense_alias_instantiate_from_config():
    spec = MODEL_SPECS["gemma-dense"]
    config_cls = spec.config_cls
    config = config_cls()

    model = instantiate(config, vocab_size=64, pad_token_id=0)
    assert isinstance(model, Gemma4DenseModel)

    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 64)
