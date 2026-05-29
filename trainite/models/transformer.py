import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


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
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, C = x.shape

        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention (B, S, num_heads, head_dim) and transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
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
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
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
        x = self.pos_encoding(x)
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
