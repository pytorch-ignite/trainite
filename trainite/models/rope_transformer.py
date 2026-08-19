import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class RotaryEmbedding(nn.Module):
    """Precomputes and applies Rotary Position Embeddings (RoPE) to Q and K tensors.

    Unlike sinusoidal absolute positional encodings that are *added* to token
    embeddings once, RoPE *rotates* the query and key vectors of each attention
    head by an angle proportional to the token's position.  This encodes relative
    distances directly in the attention dot-product, making RoPE robust to
    left-padding and to sequence lengths beyond what was seen during training.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021). https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dimension (head_dim) must be even, got {dim}.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Inverse frequencies: one per pair of embedding dimensions
        # Lower-index dimensions rotate quickly (high frequency), higher-index slowly (low frequency)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin tables up to max_seq_len to avoid recomputation at runtime
        cos, sin = self._compute_embeddings(max_seq_len, device=inv_freq.device, dtype=torch.float32)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _compute_embeddings(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Position indices [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # Outer product: each position × each frequency dimension
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to fill all head_dim slots (first half sin, second half cos)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Add batch and head dimensions so tensors broadcast over (B, num_heads, S, head_dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Recompute on-the-fly if the sequence exceeds the precomputed cache length
        if seq_len > self.max_seq_len:
            return self._compute_embeddings(seq_len, device=x.device, dtype=x.dtype)

        # Slice the cache to the actual sequence length and cast to match x's dtype
        return self.cos_cached[:, :, :seq_len].to(x.dtype), self.sin_cached[:, :, :seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (negated).

    This implements the 2D rotation matrix that is at the heart of RoPE:
    given a pair (x1, x2), return (-x2, x1), which is equivalent to a 90-degree
    rotation when combined with the cos/sin scaling in apply_rotary_pos_emb.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    The rotation encodes position so that the dot-product q·k depends only on
    the *relative* distance between tokens, not their absolute positions.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Causal multi-head self-attention with Rotary Position Embeddings (RoPE).

    Multiple attention heads capture different relationships between tokens. Before
    computing attention scores, RoPE encodes relative positions by rotating the query
    and key vectors using precomputed cosine and sine values. Causal masking prevents
    each position from attending to future tokens.
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
        cos: torch.Tensor,
        sin: torch.Tensor,
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
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

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
    """A single Transformer decoder block: Pre-LN self-attention (RoPE) + feedforward network.

    Uses Pre-Layer Normalisation (norm before attention/FFN) and residual connections.
    RoPE (cos, sin) tensors are passed in from the model so they are computed once
    per forward pass and shared across all blocks.
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
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # --- Self-attention sub-layer (Pre-LN + residual) ---
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, cos, sin, padding_mask=padding_mask)
        x = x + attn_output  # residual connection

        # --- Feedforward sub-layer (Pre-LN + residual) ---
        normed = self.norm2(x)
        x = x + self.feedforward(normed)  # residual connection
        return x


class RoPETransformerModel(nn.Module):
    """Decoder-only Transformer language model using Rotary Position Embeddings (RoPE).

    Architecture overview:
        token ids -> embedding (scaled)
                  -> RoPE cos/sin computed once for the full sequence length
                  -> N x TransformerBlock (RoPE causal self-attention + FFN)
                  -> LayerNorm
                  -> linear projection to vocab logits

    Unlike sinusoidal absolute position encodings, RoPE is robust to left-padding:
    it preserves relative distances between real tokens regardless of padding width,
    and the attention mask ensures padding positions are excluded from attention.

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
        # RoPE operates on each attention head independently; dim = head_dim
        self.rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads, max_seq_len=max_seq_len)
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
        # Scale embedding by sqrt(d_model) to keep magnitude in a reasonable range
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        # Compute RoPE cos/sin tables once and share them across all blocks
        cos, sin = self.rotary_emb(x, seq_len=S)
        # Build a boolean padding mask of shape (B, 1, 1, S) for the attention layers
        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # Only create a mask when there actually are padding tokens (optimisation)
            if not attention_mask.all().item():
                padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)
        elif (input_ids == self.embedding.padding_idx).any().item():
            # Fall back to deriving the mask from pad token ids when no mask is provided
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)
        for block in self.blocks:
            x = block(x, cos, sin, padding_mask=padding_mask)
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
