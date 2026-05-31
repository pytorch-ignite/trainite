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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if not self.num_heads * self.head_dim == self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, S, C = x.shape
        q = self.q_proj(x)

        if memory is None:
            # Self-attention
            k = self.k_proj(x)
            v = self.v_proj(x)
            S_k = S
        else:
            # Cross-attention
            k = self.k_proj(memory)
            v = self.v_proj(memory)
            S_k = memory.shape[1]

        # Reshape for multi-head attention (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        if padding_mask is not None:
            if is_causal:
                causal_mask = torch.ones(
                    S, S_k, dtype=torch.bool, device=x.device
                ).tril()
                mask = causal_mask & padding_mask
            else:
                mask = padding_mask
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
                is_causal=is_causal,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        out = self.out(context)
        return self.dropout(out)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
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
        attn_output = self.attention(normed, padding_mask=padding_mask, is_causal=False)
        x = x + attn_output

        normed = self.norm2(x)
        x = x + self.feedforward(normed)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(d_model)
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
        memory: torch.Tensor,
        self_padding_mask: torch.Tensor | None = None,
        cross_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        self_attn_out = self.self_attn(
            normed,
            padding_mask=self_padding_mask,
            is_causal=True,
        )
        x = x + self_attn_out

        normed = self.norm2(x)
        cross_attn_out = self.cross_attn(
            normed,
            memory=memory,
            padding_mask=cross_padding_mask,
            is_causal=False,
        )
        x = x + cross_attn_out

        normed = self.norm3(x)
        x = x + self.feedforward(normed)
        return x


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 64,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        encoder_max_seq_len: int = 128,
        decoder_max_seq_len: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder_pos_encoding = PositionalEncoding(
            hidden_size, encoder_max_seq_len, dropout
        )
        self.decoder_pos_encoding = PositionalEncoding(
            hidden_size, decoder_max_seq_len, dropout
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_size, num_heads, feedforward_dim, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(hidden_size, num_heads, feedforward_dim, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.proj = nn.Linear(hidden_size, vocab_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Encoder forward
        B_enc, S_enc = encoder_input_ids.shape
        x_enc = self.embedding(encoder_input_ids) * math.sqrt(
            self.embedding.embedding_dim
        )
        x_enc = self.encoder_pos_encoding(x_enc)

        if (encoder_input_ids == self.embedding.padding_idx).any():
            enc_padding_mask = (
                encoder_input_ids != self.embedding.padding_idx
            ).reshape(B_enc, 1, 1, S_enc)
        else:
            enc_padding_mask = None

        for block in self.encoder_blocks:
            x_enc = block(x_enc, padding_mask=enc_padding_mask)

        memory = x_enc

        # Decoder forward
        B_dec, S_dec = decoder_input_ids.shape
        x_dec = self.embedding(decoder_input_ids) * math.sqrt(
            self.embedding.embedding_dim
        )
        x_dec = self.decoder_pos_encoding(x_dec)

        if (decoder_input_ids == self.embedding.padding_idx).any():
            dec_padding_mask = (
                decoder_input_ids != self.embedding.padding_idx
            ).reshape(B_dec, 1, 1, S_dec)
        else:
            dec_padding_mask = None

        # Cross attention padding mask (B_dec, 1, 1, S_enc) to mask out padded tokens in encoder memory
        if (encoder_input_ids == self.embedding.padding_idx).any():
            cross_padding_mask = (
                encoder_input_ids != self.embedding.padding_idx
            ).reshape(B_dec, 1, 1, S_enc)
        else:
            cross_padding_mask = None

        x = x_dec
        for block in self.decoder_blocks:
            x = block(
                x,
                memory,
                self_padding_mask=dec_padding_mask,
                cross_padding_mask=cross_padding_mask,
            )

        x = self.norm(x)
        return self.proj(x)


def build_encoder_decoder_model(
    vocab_size: int = 32,
    hidden_size: int = 64,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    num_heads: int = 2,
    feedforward_dim: int = 128,
    dropout: float = 0.1,
    encoder_max_seq_len: int = 128,
    decoder_max_seq_len: int = 128,
    **kwargs,
) -> EncoderDecoderModel:
    return EncoderDecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        encoder_max_seq_len=encoder_max_seq_len,
        decoder_max_seq_len=decoder_max_seq_len,
    )
