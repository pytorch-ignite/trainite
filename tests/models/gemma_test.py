from types import SimpleNamespace

import torch

from trainite.config.registry import MODEL_SPECS
from trainite.models.gemma import (
    CausalLMCollateFn,
    GemmaAttention,
    GemmaBlock,
    GemmaDenseModel,
    GemmaMLP,
    RMSNorm,
    apply_rope,
)
from trainite.shared.utils import instantiate


def test_rms_norm():
    dim = 32
    norm = RMSNorm(dim=dim, eps=1e-6)
    x = torch.randn(2, 4, dim)
    out = norm(x)
    assert out.shape == x.shape
    assert norm.weight is not None


def test_gemma_mlp():
    hidden_size = 32
    intermediate_size = 64
    mlp = GemmaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    x = torch.randn(2, 4, hidden_size)
    out = mlp(x)
    assert out.shape == (2, 4, hidden_size)


def test_apply_rope_full_and_partial():
    head_dim = 16
    rotary_half = 8
    x = torch.randn(2, 4, 2, head_dim)
    positions = torch.arange(4).unsqueeze(0).repeat(2, 1)

    # Full RoPE (100%)
    out_full = apply_rope(x, positions, base_theta=10000.0, rope_proportion=1.0)
    assert out_full.shape == x.shape

    # Partial RoPE (25%)
    out_part = apply_rope(x, positions, base_theta=1000000.0, rope_proportion=0.25)
    assert out_part.shape == x.shape

    # None fallback to 1.0
    out_none = apply_rope(x, positions, base_theta=10000.0, rope_proportion=None)
    assert torch.allclose(out_full, out_none, atol=1e-6)


def test_gemma_attention_forward():
    dim = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    attn = GemmaAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_sliding=True,
        sliding_window=4,
    )
    x = torch.randn(2, 8, dim)
    positions = torch.arange(8).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, :2] = 0

    out = attn(x, positions=positions, attention_mask=mask)
    assert out.shape == (2, 8, dim)


def test_gemma_block_forward():
    dim = 32
    intermediate_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    block = GemmaBlock(
        dim=dim,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_sliding=False,
    )
    x = torch.randn(2, 6, dim)
    positions = torch.arange(6).unsqueeze(0).repeat(2, 1)

    out = block(x, positions=positions)
    assert out.shape == (2, 6, dim)


def test_gemma_dense_model_forward():
    vocab_size = 50
    dim = 32
    num_layers = 4
    model = GemmaDenseModel(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        sliding_ratio=2,
        final_logit_softcap=30.0,
        tie_word_embeddings=True,
    )

    B, S = 2, 6
    input_ids = torch.randint(0, vocab_size, (B, S))
    mask = torch.ones(B, S, dtype=torch.long)
    mask[0, :2] = 0

    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (B, S, vocab_size)

    # Gradients backward check
    loss = logits.sum()
    loss.backward()
    assert model.embed_tokens.weight.grad is not None


def test_gemma_dense_model_untied():
    vocab_size = 50
    dim = 32
    model = GemmaDenseModel(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        intermediate_size=64,
        tie_word_embeddings=False,
    )
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


def test_gemma_instantiate_from_config():
    spec = MODEL_SPECS["gemma-dense"]
    config_cls = spec.config_cls
    config = config_cls()

    model = instantiate(config, vocab_size=64, pad_token_id=0)
    assert isinstance(model, GemmaDenseModel)

    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 64)
