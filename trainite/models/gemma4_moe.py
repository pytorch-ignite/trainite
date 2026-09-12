import torch
import torch.nn.functional as F
from torch import nn


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _apply_rope_rotation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    scaling_factor: float,
    head_dim: int,
    num_rotary_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the RoPE 2D rotation to pairs drawn from the two head halves.

    For each pair `(x1[i], x2[i])`, RoPE applies the rotation matrix
    `[[cos, -sin], [sin, cos]]`. Dimensions after `num_rotary_pairs` are
    returned unchanged, which implements partial rotary embeddings.
    """
    if num_rotary_pairs == 0:
        return x1, x2

    # One frequency per rotated pair. Lower dimensions rotate faster:
    # inv_freq[i] = 1 / theta^(2i / head_dim).
    freq_exponents = (2.0 / head_dim) * torch.arange(
        num_rotary_pairs,
        device=x1.device,
        dtype=torch.float32,
    )
    inv_freq = 1.0 / (theta**freq_exponents)

    # Outer product of token positions and inverse frequencies gives one angle
    # per `[batch, sequence, rotary_pair]`. Scaling > 1 slows rotation so the
    # model can represent longer contexts.
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = angles / scaling_factor

    # Insert the singleton head axis so these broadcast over every attention
    # head: `[batch, sequence, 1, rotary_pair]`.
    sin = torch.sin(angles).unsqueeze(-2)
    cos = torch.cos(angles).unsqueeze(-2)

    # Separate dimensions that receive RoPE from dimensions that pass through.
    x1_rotary = x1[..., :num_rotary_pairs]
    x2_rotary = x2[..., :num_rotary_pairs]
    x1_pass = x1[..., num_rotary_pairs:]
    x2_pass = x2[..., num_rotary_pairs:]

    # Standard 2D rotation: (a, b) -> (a cos - b sin, b cos + a sin).
    rotated_x1 = x1_rotary * cos - x2_rotary * sin
    rotated_x2 = x2_rotary * cos + x1_rotary * sin

    return (
        torch.cat([rotated_x1, x1_pass], dim=-1),
        torch.cat([rotated_x2, x2_pass], dim=-1),
    )


def apply_text_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    theta: float,
    scaling_factor: float = 1.0,
    rotary_fraction: float | None = 1.0,
) -> torch.Tensor:
    """Apply split-half RoPE to projected query or key states.

    Unlike interleaved RoPE, Gemma pairs the first and second halves of each
    attention head: dimension `i` pairs with `i + head_dim / 2`. RoPE is
    applied before attention, independently to the query and key tensors.

    Args:
        x: Tensor shaped `[batch, sequence, heads, head_dim]`.
        position_ids: Token positions shaped `[batch, sequence]`.
        theta: RoPE base frequency (`rope_theta` in model configs).
        scaling_factor: Context-extension factor (`rope_scaling.factor` in configs).
        rotary_fraction: Fraction of `head_dim` to rotate
            (`partial_rotary_factor` or `rope_proportion` in configs).
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Each rotation consumes one dimension from each half, so rotating a
    # fraction of `head_dim` means half as many 2D rotation pairs.
    rotary_fraction = 1.0 if rotary_fraction is None else rotary_fraction
    num_rotary_pairs = int((rotary_fraction * head_dim) // 2)
    num_rotary_pairs = max(0, min(num_rotary_pairs, half_dim))

    # Gemma's split-half layout pairs (x[i], x[i + head_dim / 2]).
    x1, x2 = x.split(half_dim, dim=-1)
    x1, x2 = _apply_rope_rotation(
        x1,
        x2,
        position_ids,
        theta,
        scaling_factor,
        head_dim,
        num_rotary_pairs,
    )
    return torch.cat([x1, x2], dim=-1).to(dtype=x.dtype)


def create_sliding_mask(
    positions: torch.Tensor,
    sliding_window: int,
    cache_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create the bidirectional local mask used by the official JAX port.

    Positions whose distance is smaller than `sliding_window` can attend to
    each other. Causality, if required by the attention layer, is applied
    separately. The result has shape `[batch, query_length, key_length]`.
    """
    if cache_positions is None:
        cache_positions = positions

    # Expand query and key positions into pairwise `[batch, query, key]`
    # comparisons. This supports different query and KV-cache lengths.
    cache_positions = cache_positions[:, None, :]
    positions = positions[:, :, None]
    return (cache_positions > positions - sliding_window) & (cache_positions < positions + sliding_window)


class DenseMLP(nn.Module):
    """Gemma 4's gated dense feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # Learns which expanded features should be activated.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Projects the input into the expanded feature representation.
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Compresses the gated expanded representation back to hidden_size.
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GeGLU: gate one expanded projection with the other, then compress it.
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class Gemma4MoE(nn.Module):
    """Gemma 4 top-k mixture-of-experts feed-forward block."""

    def __init__(self, hidden_size: int, expert_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.hidden_size = hidden_size
        self.top_k = top_k
        self.router_norm = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.router_scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * expert_dim, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, expert_dim))

        for weight in self.gate_up_proj:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        for weight in self.down_proj:
            nn.init.kaiming_uniform_(weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_input = self.router_norm(x) * self.router_scale * (self.hidden_size**-0.5)
        router_logits = self.router(router_input).float()
        router_probs = F.softmax(router_logits, dim=-1)
        topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1).indices
        topk_weights = router_probs.gather(-1, topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.per_expert_scale[topk_indices]

        output = torch.zeros_like(x)
        for expert_idx in torch.unique(topk_indices).tolist():
            token_indices, topk_positions = (topk_indices == expert_idx).nonzero(as_tuple=True)
            gate, up = F.linear(x[token_indices], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            expert_output = F.linear(gelu_tanh(gate) * up, self.down_proj[expert_idx])
            expert_output = expert_output * topk_weights[token_indices, topk_positions, None].to(x.dtype)
            output.index_add_(0, token_indices, expert_output)

        return output


class Gemma4TextAttention(nn.Module):
    """Checkpoint-compatible Gemma 4 text self-attention without KV caching."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        *,
        head_dim: int | None = None,
        rope_theta: float = 10_000.0,
        rotary_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        # Query asks what each token should attend to: [hidden -> query heads].
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        # Keys and values use fewer heads in Gemma's grouped-query attention.
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        # Output mixes the attended heads back into the model hidden size.
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # Gemma normalizes each attention head after projection; values have no scale.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.rope_theta = rope_theta
        self.rotary_fraction = rotary_fraction

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device).expand(batch_size, -1)

        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query = apply_text_rope(
            self.q_norm(query), position_ids, theta=self.rope_theta, rotary_fraction=self.rotary_fraction
        )
        key = apply_text_rope(
            self.k_norm(key), position_ids, theta=self.rope_theta, rotary_fraction=self.rotary_fraction
        )
        value = self.v_norm(value)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is None:
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        else:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :].bool()
                causal = torch.ones(
                    sequence_length, sequence_length, device=hidden_states.device, dtype=torch.bool
                ).tril()
                attention_mask = attention_mask & causal
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :].bool()
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, enable_gqa=True)

        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attended)


class Gemma4TextBlock(nn.Module):
    """Gemma 4 MoE decoder block: attention, then parallel dense and expert FFNs."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dense_intermediate_size: int,
        expert_dim: int,
        num_experts: int,
        top_k: int,
        *,
        head_dim: int | None = None,
        rope_theta: float = 10_000.0,
        rotary_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.attention = Gemma4TextAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rotary_fraction=rotary_fraction,
        )
        self.pre_attention_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.pre_dense_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.dense_mlp = DenseMLP(hidden_size, dense_intermediate_size)
        self.post_dense_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.pre_moe_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.moe = Gemma4MoE(hidden_size, expert_dim, num_experts, top_k)
        self.post_moe_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_feedforward_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.layer_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention(self.pre_attention_norm(hidden_states), position_ids, attention_mask)
        hidden_states = residual + self.post_attention_norm(hidden_states)

        residual = hidden_states
        dense = self.post_dense_norm(self.dense_mlp(self.pre_dense_norm(hidden_states)))
        moe_input = self.pre_moe_norm(hidden_states)
        moe = self.moe(moe_input.reshape(-1, moe_input.shape[-1])).reshape_as(moe_input)
        hidden_states = self.post_feedforward_norm(dense + self.post_moe_norm(moe))
        return (residual + hidden_states) * self.layer_scale


class Gemma4TextModel(nn.Module):
    """Minimal trainable Gemma 4 MoE language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dense_intermediate_size: int,
        expert_dim: int,
        num_experts: int,
        top_k: int,
        *,
        head_dim: int | None = None,
        rope_theta: float = 10_000.0,
        rotary_fraction: float = 1.0,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.embedding_scale = hidden_size**0.5
        self.layers = nn.ModuleList(
            Gemma4TextBlock(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                dense_intermediate_size,
                expert_dim,
                num_experts,
                top_k,
                head_dim=head_dim,
                rope_theta=rope_theta,
                rotary_fraction=rotary_fraction,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None and attention_mask is not None:
            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        hidden_states = self.token_embedding(input_ids) * self.embedding_scale
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)

        hidden_states = self.final_norm(hidden_states)
        return F.linear(hidden_states, self.token_embedding.weight)
