from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import trainite.models.gemma4_dense as gemma4_dense
import trainite.models.gemma4_moe as gemma4_moe
from trainite.config.models import Gemma4DenseModelConfig, Gemma4MoEModelConfig
from trainite.config.registry import MODEL_SPECS
from trainite.models.gemma4_dense import Gemma4DenseBlock, Gemma4DenseModel
from trainite.models.gemma4_moe import Gemma4MoE, Gemma4TextBlock, Gemma4MoEModel
from trainite.shared.utils import build_model, instantiate


def make_dense_model(**overrides) -> Gemma4DenseModel:
    options = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "dense_intermediate_size": 12,
        "layer_types": ("sliding", "global"),
        "sliding_window": 2,
        "global_num_key_value_heads": 1,
        "global_head_dim": 6,
        "global_key_equals_value": True,
    }
    options.update(overrides)
    return Gemma4DenseModel(**options)


def make_moe_model(**overrides) -> Gemma4MoEModel:
    options = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "dense_intermediate_size": 12,
        "expert_dim": 4,
        "num_experts": 3,
        "top_k": 2,
        "layer_types": ("sliding", "global"),
        "sliding_window": 2,
        "global_num_key_value_heads": 1,
        "global_head_dim": 6,
        "global_key_equals_value": True,
    }
    options.update(overrides)
    return Gemma4MoEModel(**options)


# ---------------------------------------------------------------------------
# DenseMLP and Activation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mlp_cls", "gelu_tanh_fn"),
    [
        (gemma4_dense.DenseMLP, gemma4_dense.gelu_tanh),
        (gemma4_moe.DenseMLP, gemma4_moe.gelu_tanh),
    ],
)
def test_dense_mlp(mlp_cls, gelu_tanh_fn):
    torch.manual_seed(0)
    mlp = mlp_cls(hidden_size=8, intermediate_size=16)
    x = torch.randn(2, 3, 8)

    actual = mlp(x)
    expected = mlp.down_proj(gelu_tanh_fn(mlp.gate_proj(x)) * mlp.up_proj(x))

    assert actual.shape == (2, 3, 8)
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# RoPE Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "apply_rope_fn",
    [
        gemma4_dense.apply_text_rope,
        gemma4_moe.apply_text_rope,
    ],
)
def test_apply_text_rope_full_and_partial(apply_rope_fn):
    head_dim = 16
    x = torch.randn(2, 4, 2, head_dim)
    position_ids = torch.arange(4).unsqueeze(0).repeat(2, 1)

    # Full RoPE (100%)
    out_full = apply_rope_fn(x, position_ids, theta=10000.0, rotary_fraction=1.0)
    assert out_full.shape == x.shape

    # Partial RoPE (25%)
    out_part = apply_rope_fn(x, position_ids, theta=1000000.0, rotary_fraction=0.25)
    assert out_part.shape == x.shape

    # Default rotary_fraction fallback to 1.0
    out_default = apply_rope_fn(x, position_ids, theta=10000.0)
    assert torch.allclose(out_full, out_default, atol=1e-6)


# ---------------------------------------------------------------------------
# Attention Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attention_cls",
    [
        gemma4_dense.Gemma4TextAttention,
        gemma4_moe.Gemma4TextAttention,
    ],
)
def test_gemma4_text_attention_forward(attention_cls):
    hidden_size = 32
    attn = attention_cls(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
    )
    x = torch.randn(2, 8, hidden_size)
    position_ids = torch.arange(8).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, :2] = 0

    out = attn(x, position_ids=position_ids, attention_mask=mask)
    assert out.shape == (2, 8, hidden_size)


@pytest.mark.parametrize(
    "attention_cls",
    [
        gemma4_dense.Gemma4TextAttention,
        gemma4_moe.Gemma4TextAttention,
    ],
)
def test_gemma4_text_attention_key_equals_value(attention_cls):
    hidden_size = 32
    attn = attention_cls(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        key_equals_value=True,
    )
    assert attn.v_proj is None
    assert attn.v_norm.weight is None  # unscaled RMSNorm

    x = torch.randn(2, 4, hidden_size)
    position_ids = torch.arange(4).unsqueeze(0).repeat(2, 1)
    out = attn(x, position_ids=position_ids)
    assert out.shape == (2, 4, hidden_size)


@pytest.mark.parametrize(
    "attention_cls",
    [
        gemma4_dense.Gemma4TextAttention,
        gemma4_moe.Gemma4TextAttention,
    ],
)
def test_sliding_attention_combines_local_and_causal_masks(attention_cls):
    attention = attention_cls(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=2,
    )

    with patch(
        "torch.nn.functional.scaled_dot_product_attention",
        return_value=torch.zeros(1, 2, 4, 4),
    ) as sdpa:
        attention(torch.randn(1, 4, 8))

    mask = sdpa.call_args.kwargs["attn_mask"][0, 0]
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
            ]
        ),
    )


# ---------------------------------------------------------------------------
# MoE & Block Components
# ---------------------------------------------------------------------------


def test_gemma4_moe_routes_and_combines_top_k_experts():
    torch.manual_seed(0)
    moe = Gemma4MoE(hidden_size=8, expert_dim=4, num_experts=3, top_k=2)
    x = torch.randn(5, 8)

    actual = moe(x)

    router_input = moe.router_norm(x) * moe.router_scale * (moe.hidden_size**-0.5)
    logits = moe.router(router_input).float()
    probs = torch.softmax(logits, dim=-1)
    _, indices = torch.topk(logits, k=moe.top_k, dim=-1)
    weights = probs.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * moe.per_expert_scale[indices]

    expected = torch.zeros_like(x)
    for expert_idx in torch.unique(indices).tolist():
        match = indices == expert_idx
        token_idx, topk_pos = match.nonzero(as_tuple=True)
        gate, up = F.linear(x[token_idx], moe.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        expert_output = F.linear(gemma4_moe.gelu_tanh(gate) * up, moe.down_proj[expert_idx])
        expected.index_add_(
            0,
            token_idx,
            expert_output * weights[token_idx, topk_pos].unsqueeze(-1),
        )

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)


def test_gemma4_dense_block_forward():
    block = Gemma4DenseBlock(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        dense_intermediate_size=64,
        head_dim=8,
    )
    x = torch.randn(2, 6, 32)
    position_ids = torch.arange(6).unsqueeze(0).repeat(2, 1)

    out = block(x, position_ids=position_ids)
    assert out.shape == (2, 6, 32)


def test_gemma4_text_block_matches_attention_and_parallel_ffn_branches():
    torch.manual_seed(0)
    block = Gemma4TextBlock(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        dense_intermediate_size=12,
        expert_dim=4,
        num_experts=3,
        top_k=2,
        head_dim=4,
    )
    x = torch.randn(2, 3, 8)

    actual = block(x)

    after_attention = x + block.post_attention_norm(block.attention(block.pre_attention_norm(x)))
    dense = block.post_dense_norm(block.dense_mlp(block.pre_dense_norm(after_attention)))
    moe_input = block.pre_moe_norm(after_attention)
    moe = block.moe(moe_input.reshape(-1, 8), router_input=after_attention.reshape(-1, 8)).reshape_as(moe_input)
    expected = after_attention + block.post_feedforward_norm(dense + block.post_moe_norm(moe))
    expected = expected * block.layer_scale

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Collate Function Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collate_cls",
    [
        gemma4_dense.CausalLMCollateFn,
        gemma4_moe.CausalLMCollateFn,
    ],
)
def test_causal_lm_collate_fn(collate_cls):
    collate = collate_cls(SimpleNamespace(pad_token_id=9))
    batch = [
        SimpleNamespace(
            train_input_ids=torch.tensor([1, 2]),
            train_label_ids=torch.tensor([2, 3]),
            attention_mask=torch.tensor([1, 1]),
        ),
        SimpleNamespace(
            train_input_ids=torch.tensor([4]),
            train_label_ids=torch.tensor([5]),
            attention_mask=torch.tensor([1]),
        ),
    ]

    result = collate(batch)

    assert torch.equal(result["input_ids"], torch.tensor([[1, 2], [9, 4]]))
    assert torch.equal(result["attention_mask"], torch.tensor([[1, 1], [0, 1]]))
    assert torch.equal(result["labels"], torch.tensor([[2, 3], [-100, 5]]))


# ---------------------------------------------------------------------------
# Full Model Tests (Parametrized Across Dense & MoE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_factory", [make_dense_model, make_moe_model])
def test_model_forward_and_backward(model_factory):
    model = model_factory(
        pad_token_id=0,
        rope_scaling_factor=2.0,
        global_rope_scaling_factor=4.0,
    )
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = input_ids != 0

    logits = model(input_ids, attention_mask)
    logits.sum().backward()

    assert logits.shape == (2, 3, 32)
    assert model.token_embedding.weight.grad is not None
    assert model.layers[0].attention.sliding_window == 2
    assert model.layers[0].attention.head_dim == 4
    assert model.layers[0].attention.rope_scaling_factor == 2.0
    assert model.layers[1].attention.sliding_window is None
    assert model.layers[1].attention.head_dim == 6
    assert model.layers[1].attention.rope_scaling_factor == 4.0
    assert model.layers[1].attention.v_proj is None


@pytest.mark.parametrize("model_factory", [make_dense_model, make_moe_model])
def test_model_builds_distinct_local_and_global_attention(model_factory):
    model = model_factory()
    local = model.layers[0].attention
    global_attention = model.layers[1].attention

    assert local.q_proj.weight.shape == (8, 8)
    assert local.k_proj.weight.shape == (4, 8)
    assert local.v_proj is not None
    assert local.o_proj.weight.shape == (8, 8)
    assert local.rope_theta == 10_000
    assert local.rotary_fraction == 1.0

    assert global_attention.q_proj.weight.shape == (12, 8)
    assert global_attention.k_proj.weight.shape == (6, 8)
    assert global_attention.v_proj is None
    assert global_attention.o_proj.weight.shape == (8, 12)
    assert global_attention.rope_theta == 1_000_000
    assert global_attention.rotary_fraction == 0.25


@pytest.mark.parametrize("model_factory", [make_dense_model, make_moe_model])
def test_model_softcaps_tied_embedding_logits(model_factory):
    model = model_factory(final_logit_softcap=0.1)

    logits = model(torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, 32)
    assert logits.abs().max() <= 0.1
    assert "lm_head.weight" not in model.state_dict()


def test_gemma4_dense_model_untied():
    model = make_dense_model(tie_word_embeddings=False)
    input_ids = torch.randint(0, 32, (2, 4))
    logits = model(input_ids)

    assert logits.shape == (2, 4, 32)
    assert model.lm_head is not None


@pytest.mark.parametrize("model_factory", [make_dense_model, make_moe_model])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_types": ("global",)}, "one entry per layer"),
        ({"layer_types": ("sliding", "invalid")}, "'sliding' or 'global'"),
        ({"sliding_window": None, "layer_types": ("sliding", "global")}, "required for sliding layers"),
        ({"layer_types": ("sliding", "global"), "layer_pattern": "sg"}, "mutually exclusive"),
        ({"layer_types": None, "layer_pattern": "sx"}, "must only contain"),
        ({"layer_types": None, "layer_pattern": "sgg"}, "does not divide"),
    ],
)
def test_model_rejects_invalid_layer_configuration(model_factory, overrides, message):
    with pytest.raises(ValueError, match=message):
        model_factory(**overrides)


@pytest.mark.parametrize("model_factory", [make_dense_model, make_moe_model])
def test_model_expands_compact_layer_pattern(model_factory):
    model = model_factory(
        num_layers=12,
        layer_types=None,
        layer_pattern="sssssg",
    )

    assert model.layer_types == ("sliding",) * 5 + ("global",) + ("sliding",) * 5 + ("global",)
    assert model.layers[0].attention.sliding_window == 2
    assert model.layers[5].attention.sliding_window is None
    assert model.layers[5].attention.v_proj is None

    tiled = model_factory(num_layers=6, layer_types=None, layer_pattern="sssssg")
    assert tiled.layer_types == ("sliding",) * 5 + ("global",)


# ---------------------------------------------------------------------------
# Config & Registry Instantiation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_cls",
    [
        Gemma4DenseModelConfig,
        Gemma4MoEModelConfig,
    ],
)
def test_gemma4_config_defaults_to_sg_pattern(config_cls):
    config = config_cls()
    assert config.layer_pattern == "sg"
    assert config.rope_theta == 10_000
    assert config.global_rope_theta == 1_000_000
    assert config.rotary_fraction == 1.0
    assert config.global_rotary_fraction == 0.25
    assert config.attention_dropout == 0.0


@pytest.mark.parametrize(
    ("spec_name", "model_cls"),
    [
        ("gemma4-dense", Gemma4DenseModel),
        ("gemma4-moe", Gemma4MoEModel),
    ],
)
def test_gemma4_model_instantiate_from_config(spec_name, model_cls):
    spec = MODEL_SPECS[spec_name]
    config = spec.config_cls()

    model = instantiate(config, vocab_size=64, pad_token_id=0)
    assert isinstance(model, model_cls)

    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 64)

    built = build_model(config, "cpu", vocab_size=17, pad_token_id=3)
    assert isinstance(built, model_cls)
    assert built.token_embedding.num_embeddings == 17
    assert built.token_embedding.padding_idx == 3


@pytest.mark.parametrize("spec_name", ["gemma4-dense", "gemma4-moe"])
def test_attention_dropout_only_applies_during_training(spec_name):
    config = MODEL_SPECS[spec_name].config_cls(attention_dropout=0.25)
    model = instantiate(config, vocab_size=32)
    input_ids = torch.tensor([[1, 2, 3]])

    with patch(
        "torch.nn.functional.scaled_dot_product_attention",
        side_effect=lambda query, *args, **kwargs: torch.zeros_like(query),
    ) as sdpa:
        model(input_ids)
        assert len(sdpa.call_args_list) == config.num_layers
        assert all(call.kwargs["dropout_p"] == 0.25 for call in sdpa.call_args_list)
        sdpa.reset_mock()
        model.eval()
        model(input_ids)
        assert len(sdpa.call_args_list) == config.num_layers
        assert all(call.kwargs["dropout_p"] == 0.0 for call in sdpa.call_args_list)

    model(input_ids).sum().backward()
    assert model.token_embedding.weight.grad is not None

    for invalid in (-0.1, 1.0):
        with pytest.raises(ValueError):
            MODEL_SPECS[spec_name].config_cls(attention_dropout=invalid)
        with pytest.raises(ValueError, match="attention_dropout"):
            model.layers[0].attention.__class__(8, 2, 1, head_dim=4, attention_dropout=invalid)
