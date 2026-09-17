import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any
from torch.nn.utils.rnn import pad_sequence


class RMSNorm(nn.RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=with_scale)


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base_theta: float = 10000.0,
    rope_proportion: float | None = 1.0,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Applies RoPE (Rotary Positional Embedding) to the input tensor using Gemma/Google Deepmind's RoPE implementation.
    """
    if rope_proportion is None:
        rope_proportion = 1.0

    if not (0 < rope_proportion <= 1):
        raise ValueError(f"rope_proportion must be in the range (0, 1], got {rope_proportion}")

    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    rotary_half = int((rope_proportion * head_dim) // 2)

    if rotary_half == 0:
        return x

    freq_exponents = (2.0 / head_dim) * torch.arange(
        rotary_half, device=x.device, dtype=torch.float32
    )  # We use float32 to avoid overflow
    timescale = base_theta**freq_exponents

    sinusoid = (
        positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) / timescale
    )  # (batch, seq_len, rotary_half)
    sinusoid = sinusoid / scale_factor

    cos = torch.cos(sinusoid).unsqueeze(-2)
    sin = torch.sin(sinusoid).unsqueeze(-2)

    first_half, second_half = x.split(half_dim, dim=-1)

    first_rot = first_half[..., :rotary_half]
    first_pass = first_half[..., rotary_half:]

    second_rot = second_half[..., :rotary_half]
    second_pass = second_half[..., rotary_half:]

    rotated_first = first_rot * cos - second_rot * sin
    rotated_second = first_rot * sin + second_rot * cos

    out_first = torch.cat([rotated_first, first_pass], dim=-1)
    out_second = torch.cat([rotated_second, second_pass], dim=-1)

    return torch.cat([out_first, out_second], dim=-1).to(dtype=x.dtype)


class GemmaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        is_sliding: bool = False,
        sliding_window: int | None = 1024,
        rope_theta: float = 10000.0,
        rope_proportion: float = 1.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        key_equals_value: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_proportion = rope_proportion
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.key_equals_value = key_equals_value

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = None if key_equals_value else nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.v_norm = RMSNorm(head_dim, eps=rms_norm_eps, with_scale=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = k if self.v_proj is None else self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q = apply_rope(q, positions, self.rope_theta, self.rope_proportion)
        k = apply_rope(k, positions, self.rope_theta, self.rope_proportion)

        # PyTorch expects (B, S, num_heads, head_dim) but we have (B, num_heads, S, head_dim) so we transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = torch.ones((S, S), dtype=torch.bool, device=x.device).tril()

        if self.is_sliding and self.sliding_window is not None:
            sliding_mask = torch.ones((S, S), dtype=torch.bool, device=x.device).triu(
                diagonal=-(self.sliding_window - 1)
            )
            mask = mask & sliding_mask

        if attention_mask is not None:
            mask = mask & attention_mask.bool().unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=1.0, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class GemmaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        is_sliding: bool = False,
        sliding_window: int | None = 1024,
        rope_theta: float = 10000.0,
        rope_proportion: float = 1.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_post_attn_norm: bool = True,
        use_post_ffn_norm: bool = True,
        key_equals_value: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_proportion = rope_proportion
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.use_post_attn_norm = use_post_attn_norm
        self.use_post_ffn_norm = use_post_ffn_norm
        self.key_equals_value = key_equals_value

        self.pre_attn_norm = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = GemmaAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_sliding=is_sliding,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            rope_proportion=rope_proportion,
            rms_norm_eps=rms_norm_eps,
            dropout=dropout,
            key_equals_value=key_equals_value,
        )
        self.post_attn_norm = RMSNorm(dim, eps=rms_norm_eps) if use_post_attn_norm else None

        self.pre_ffn_norm = RMSNorm(dim, eps=rms_norm_eps)
        self.mlp = GemmaMLP(hidden_size=dim, intermediate_size=intermediate_size)
        self.post_ffn_norm = RMSNorm(dim, eps=rms_norm_eps) if use_post_ffn_norm else None

        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        h = self.pre_attn_norm(x)
        h = self.attn(h, positions, attention_mask)
        if self.post_attn_norm is not None:
            h = self.post_attn_norm(h)
        x = residual + h

        residual = x  # residual now holds the value of x holds the post-attention state
        h = self.pre_ffn_norm(x)
        h = self.mlp(h)
        if self.post_ffn_norm is not None:
            h = self.post_ffn_norm(h)
        x = residual + h  # x now holds: Original + Attention + MLP
        x = x * self.layer_scalar

        return x


class GemmaDenseModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        sliding_window: int = 512,
        local_rope_theta: float = 10000.0,
        global_rope_theta: float = 1000000.0,
        local_rope_proportion: float = 1.0,
        global_rope_proportion: float = 0.25,
        sliding_ratio: int = 5,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        final_logit_softcap: float | None = None,
        tie_word_embeddings: bool = True,
        global_key_equals_value: bool = True,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.local_rope_theta = local_rope_theta
        self.global_rope_theta = global_rope_theta
        self.local_rope_proportion = local_rope_proportion
        self.global_rope_proportion = global_rope_proportion
        self.sliding_ratio = sliding_ratio
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.final_logit_softcap = final_logit_softcap
        self.tie_word_embeddings = tie_word_embeddings
        self.global_key_equals_value = global_key_equals_value

        self.embed_tokens = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.embed_scale = math.sqrt(dim)
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            is_sliding = ((layer + 1) % (sliding_ratio + 1)) != 0
            if layer == num_layers - 1:
                is_sliding = False  # Gemma makes final layer global

            if is_sliding:
                rope_theta = local_rope_theta
                rope_proportion = local_rope_proportion
                window = sliding_window
                key_equals_value = False
            else:
                rope_theta = global_rope_theta
                rope_proportion = global_rope_proportion
                window = None
                key_equals_value = global_key_equals_value

            self.layers.append(
                GemmaBlock(
                    dim=dim,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_sliding=is_sliding,
                    sliding_window=window,
                    rope_theta=rope_theta,
                    rope_proportion=rope_proportion,
                    rms_norm_eps=rms_norm_eps,
                    dropout=dropout,
                    use_post_attn_norm=True,
                    use_post_ffn_norm=True,
                    key_equals_value=key_equals_value,
                )
            )

        self.final_norm = RMSNorm(dim, eps=rms_norm_eps)
        self.lm_head = None if tie_word_embeddings else nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) * self.embed_scale

        if attention_mask is not None:
            positions = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        else:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        for layer in self.layers:
            x = layer(x, positions=positions, attention_mask=attention_mask)

        x = self.final_norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        if self.final_logit_softcap:
            logits = torch.tanh(logits / self.final_logit_softcap) * self.final_logit_softcap

        return logits


class CausalLMCollateFn:
    """Collate sequences for decoder-only autoregressive training.

    Pads a batch of variable-length sequences to the same length using *left*-padding
    (padding tokens are prepended rather than appended).  Left-padding keeps all real
    tokens right-aligned so a single causal mask works correctly for the whole batch.

    The trick used here is: flip each sequence, right-pad with ``pad_sequence``, then
    flip the result back -- this is equivalent to left-padding.

    ``ignore_index`` (-100 by default) is used as the label value for padding
    positions so that PyTorch's ``CrossEntropyLoss`` ignores them during training.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad_id if pad_id is not None else 0
        # Cross-entropy ignores positions with this label value (PyTorch convention)
        self.ignore_index = -100

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in batch:
            input_ids = item.train_input_ids
            labels = item.train_label_ids
            attention_mask = item.attention_mask

            # We flip the sequences to have padding on the left, which allows us to use causal masking without modification
            input_ids_list.append(input_ids.flip(0))
            labels_list.append(labels.flip(0))
            attention_mask_list.append(attention_mask.flip(0))

        # pad_sequence right-pads; flipping before and after converts that to left-padding
        padded_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id if self.pad_token_id is not None else 0,
        ).flip(1)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).flip(1)
        # Use ignore_index so the loss function skips padded label positions
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index).flip(1)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }
