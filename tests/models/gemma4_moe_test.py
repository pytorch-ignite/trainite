import torch
import torch.nn.functional as F

from trainite.models.gemma4_moe import DenseMLP, Gemma4MoE, Gemma4TextBlock, Gemma4TextModel, gelu_tanh


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
    moe = block.moe(moe_input.reshape(-1, 8)).reshape_as(moe_input)
    expected = after_attention + block.post_feedforward_norm(dense + block.post_moe_norm(moe))
    expected = expected * block.layer_scale

    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)


def test_gemma4_text_model_returns_token_logits_and_backpropagates():
    model = Gemma4TextModel(
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        dense_intermediate_size=12,
        expert_dim=4,
        num_experts=3,
        top_k=2,
        head_dim=4,
        padding_idx=0,
    )
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attention_mask = input_ids != 0

    logits = model(input_ids, attention_mask)
    logits.sum().backward()

    assert logits.shape == (2, 3, 32)
    assert model.token_embedding.weight.grad is not None
