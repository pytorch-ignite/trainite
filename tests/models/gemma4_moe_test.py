from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from trainite.models.gemma4_moe import (
    DenseMLP,
    Gemma4MoE,
    Gemma4TextBlock,
    Gemma4TextModel,
    gelu_tanh,
    Gemma4TextAttention,
    CausalLMCollateFn,
)
from trainite.config.models import Gemma4MoEModelConfig
from trainite.shared.utils import build_model


def make_text_model(**overrides) -> Gemma4TextModel:
    options = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "dense_intermediate_size": 12,
        "expert_dim": 4,
        "num_experts": 3,
        "top_k": 2,
        "head_dim": 4,
        "layer_types": ("sliding", "global"),
        "sliding_window": 2,
        "global_num_key_value_heads": 1,
        "global_head_dim": 6,
        "global_rope_theta": 1_000_000,
        "global_rotary_fraction": 0.25,
        "global_key_equals_value": True,
    }
    options.update(overrides)
    return Gemma4TextModel(**options)


def test_causal_lm_collate_left_pads_batch():
    collate = CausalLMCollateFn(SimpleNamespace(pad_token_id=9))
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


def test_gemma4_config_builds_with_tokenizer_values():
    config = Gemma4MoEModelConfig(
        hidden_size=8,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        dense_intermediate_size=12,
        expert_dim=4,
        num_experts=3,
        top_k=2,
        head_dim=4,
        layer_types=("sliding", "global"),
        sliding_window=2,
        global_num_key_value_heads=1,
        global_head_dim=4,
    )

    model = build_model(config, "cpu", vocab_size=17, pad_token_id=3)

    assert model.token_embedding.num_embeddings == 17
    assert model.token_embedding.padding_idx == 3


def test_gemma4_text_attention_forward():
    attention = Gemma4TextAttention(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    hidden_states = torch.randn(2, 5, 32)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape


def test_sliding_attention_combines_local_and_causal_masks():
    attention = Gemma4TextAttention(
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


def test_dense_mlp_matches_gated_projection_formula():
    torch.manual_seed(0)
    mlp = DenseMLP(hidden_size=8, intermediate_size=16)
    x = torch.randn(2, 3, 8)

    actual = mlp(x)
    expected = mlp.down_proj(gelu_tanh(mlp.gate_proj(x)) * mlp.up_proj(x))

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)


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
        expert_output = F.linear(gelu_tanh(gate) * up, moe.down_proj[expert_idx])
        expected.index_add_(
            0,
            token_idx,
            expert_output * weights[token_idx, topk_pos].unsqueeze(-1),
        )

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)


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


def test_gemma4_text_model_returns_token_logits_and_backpropagates():
    model = make_text_model(
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


def test_text_model_builds_distinct_local_and_global_attention():
    model = make_text_model()
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
    assert model.layers[0].moe.router.weight is not model.layers[1].moe.router.weight


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_types": ("global",)}, "one entry per layer"),
        ({"layer_types": ("sliding", "invalid")}, "'sliding' or 'global'"),
        ({"sliding_window": None}, "required for sliding layers"),
    ],
)
def test_text_model_rejects_invalid_layer_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_text_model(**overrides)


def test_text_model_softcaps_tied_embedding_logits():
    model = make_text_model(final_logit_softcap=0.1)

    logits = model(torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, 32)
    assert logits.abs().max() <= 0.1
    assert "lm_head.weight" not in model.state_dict()
