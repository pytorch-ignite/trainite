import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class SinusoidalPositionalEncoding(nn.Module):
    """Encodes position information into token embeddings using fixed sine/cosine waves.

    Because a Transformer's attention mechanism treats every token the same regardless
    of its position in the sequence, we need to explicitly inject positional information.
    This module computes a unique pattern of sine and cosine values for each position,
    so the model can learn to distinguish token order.

    See "Attention Is All You Need" (Vaswani et al., 2017) for the original formulation.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")

        self.dropout = nn.Dropout(p=dropout)
        # pe[pos, :] will hold the positional encoding vector for position `pos`
        pe = torch.zeros(max_seq_len, d_model)
        # Column vector of positions [0, 1, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Scaling term so successive dimensions have geometrically increasing wavelengths
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        # Even dimensions use sine, odd dimensions use cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a non-trainable buffer so it moves with the model to GPU etc.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # Index the precomputed table with the actual position indices for this batch
        return self.dropout(self.pe[positions])


class Attention(nn.Module):
    """Causal multi-head self-attention with an optional padding mask.

    Multi-head attention splits the embedding dimension into `num_heads` smaller
    "heads" that each learn to attend to different aspects of the input.  The
    outputs are concatenated and projected back to `embed_dim`.

    Causal (auto-regressive) masking ensures that position i can only attend to
    positions 0..i, so the model cannot look ahead during training.

    Docs: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Each head operates on a slice of the full embedding
        self.head_dim = embed_dim // num_heads
        if not self.num_heads * self.head_dim == self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        # Single fused linear that produces Q, K, V in one matrix multiply
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Final projection that mixes the concatenated head outputs
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

        # When padding tokens are present we combine causal + padding masks; otherwise
        # PyTorch's built-in causal path is used (faster and uses less memory).
        if padding_mask is not None:
            # Lower-triangular causal mask prevents attending to future tokens
            causal_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            # Combined mask: True only where the position is both causal and non-padding
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
        # Merge heads back: (B, num_heads, S, head_dim) -> (B, S, embed_dim)
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        out = self.out(context)
        return self.dropout(out), context


class TransformerBlock(nn.Module):
    """A single Transformer decoder block: Pre-LN self-attention + feedforward network.

    Uses Pre-Layer Normalisation (norm before attention/FFN) which is more stable
    to train than the original Post-LN design.  Each sub-layer is wrapped with a
    residual connection so that gradients can flow through the whole stack.

    See "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020).
    """

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Normalisation applied before attention (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, dropout=dropout)
        # Normalisation applied before the feedforward network (Pre-LN)
        self.norm2 = nn.LayerNorm(d_model)
        # Position-wise FFN: expands to feedforward_dim then projects back to d_model
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
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # --- Self-attention sub-layer (Pre-LN + residual) ---
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, padding_mask=padding_mask)
        x = x + attn_output  # residual connection

        # --- Feedforward sub-layer (Pre-LN + residual) ---
        normed = self.norm2(x)
        x = x + self.feedforward(normed)  # residual connection
        return x


class BasicTransformerModel(nn.Module):
    """Decoder-only Transformer language model using sinusoidal absolute positional encoding.

    Architecture overview:
        token ids -> embedding (scaled) + positional encoding
                  -> N x TransformerBlock (causal self-attention + FFN)
                  -> LayerNorm
                  -> linear projection to vocab logits

    The sinusoidal encoding assigns each position a unique fixed pattern of sin/cos
    values (see SinusoidalPositionalEncoding).  Because the positions are *absolute*,
    left-padding shifts real-token positions and can silently hurt quality -- prefer
    the RoPETransformerModel when you need robust left-padding support.

    """

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
        # Token embedding: maps each integer token id to a dense vector
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len, dropout=dropout)
        # Stack of N decoder blocks
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
        # Final linear projection: hidden_size -> vocab_size (produces raw logits)
        self.proj = nn.Linear(hidden_size, vocab_size)
        # Final layer norm applied before the projection
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S = input_ids.shape

        # Derive position indices.  When an attention_mask is supplied (e.g. for
        # left-padded batches), we count only non-padding tokens so that each real
        # token gets its intended position rather than its slot index.
        if attention_mask is not None:
            positions = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        else:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        # Scale embedding by sqrt(d_model) to keep magnitude comparable with positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim) + self.pos_encoding(positions)

        # Build a boolean padding mask of shape (B, 1, 1, S) for the attention layers.
        # Shape (B, 1, 1, S) broadcasts over (B, num_heads, S, S) attention scores.
        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # Only create a mask when there actually are padding tokens (optimisation)
            if not attention_mask.all().item():
                padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)
        elif (input_ids == self.embedding.padding_idx).any().item():
            # Fall back to deriving the mask from pad token ids when no mask is provided
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)
        return self.proj(x)


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
