import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dimension (head_dim) must be even, got {dim}.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin buffers
        cos, sin = self._compute_embeddings(max_seq_len, device=inv_freq.device, dtype=torch.float32)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _compute_embeddings(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            return self._compute_embeddings(seq_len, device=x.device, dtype=x.dtype)

        return self.cos_cached[:, :, :seq_len].to(x.dtype), self.sin_cached[:, :, :seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ponytail: exact equivalent implementation of PyTorch docs SDPA, modified to return weights
def manual_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # Handle NaN values for fully masked rows to match PyTorch's native SDPA behavior
    attn_weight = torch.nan_to_num(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    return attn_weight @ value, attn_weight


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if not self.num_heads * self.head_dim == self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, S, C = x.shape

        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention (B, S, num_heads, head_dim) and transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ponytail: determine mask and causal flag to pass to attention implementations
        attn_mask, is_causal = None, True
        if padding_mask is not None:
            attn_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril() & padding_mask
            is_causal = False

        if output_attentions:
            context, attn_weights = manual_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=is_causal,
                training=self.training,
            )
        else:
            attn_weights = None
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        out = self.out(context)
        return self.dropout(out), attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.norm1(x)
        attn_output, attn_weights = self.attention(
            normed, cos, sin, padding_mask=padding_mask, output_attentions=output_attentions
        )
        x = x + attn_output

        normed = self.norm2(x)
        x = x + self.feedforward(normed)
        return x, attn_weights


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        cos, sin = self.rotary_emb(x, seq_len=S)
        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if not attention_mask.all().item():
                padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)
        elif (input_ids == self.embedding.padding_idx).any().item():
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)

        all_attentions = []
        for block in self.blocks:
            x, attn = block(x, cos, sin, padding_mask=padding_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attn)
        x = self.norm(x)
        logits = self.proj(x)

        if output_attentions:
            return logits, all_attentions
        return logits


class CausalLMCollateFn:
    """Collate sequences for decoder-only autoregressive training."""

    def __init__(
        self,
        tokenizer: Any,
        pad_token_id: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.ignore_index = ignore_index

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

        padded_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id if self.pad_token_id is not None else 0,
        ).flip(1)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).flip(1)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index).flip(1)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }
