import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


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
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, C = x.shape

        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention (B, S, num_heads, head_dim) and transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        cos, sin = self.rotary_emb(q, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # padding mask shape should be (B,1,1,S) to broadcast correctly with attention scores of shape (B, num_heads, S, S)
        if padding_mask is not None:
            causal_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            mask = causal_mask & padding_mask
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        out = self.out(context)
        return self.dropout(out), context


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
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
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, padding_mask=padding_mask)
        x = x + attn_output

        normed = self.norm2(x)
        x = x + self.feedforward(normed)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        if (input_ids == self.embedding.padding_idx).any():
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)
        else:
            padding_mask = None
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)
        return self.proj(x)


def build_transformer_model(
    vocab_size: int = 32,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    feedforward_dim: int = 128,
    dropout: float = 0.1,
    max_seq_len: int = 128,
    **kwargs,
) -> TransformerModel:
    return TransformerModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
