import math
from typing import Protocol

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class Tokenizer(Protocol):
    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def sep_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding dimension (head_dim) must be even, got {dim}."
            )
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin buffers
        cos, sin = self._compute_embeddings(
            max_seq_len, device=inv_freq.device, dtype=torch.float32
        )
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

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            return self._compute_embeddings(seq_len, device=x.device, dtype=x.dtype)

        return self.cos_cached[:, :, :seq_len].to(x.dtype), self.sin_cached[
            :, :, :seq_len
        ].to(x.dtype)


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
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1
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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, cos, sin, padding_mask=padding_mask)
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
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads, max_seq_len=max_seq_len
        )
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
        cos, sin = self.rotary_emb(x, seq_len=S)
        if (input_ids == self.embedding.padding_idx).any():
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)
        else:
            padding_mask = None
        for block in self.blocks:
            x = block(x, cos, sin, padding_mask=padding_mask)
        x = self.norm(x)
        return self.proj(x)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        tokenizer: Tokenizer,
        bos_token_id: int | None = None,
        sep_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> list[str]:
        """Generate text from a raw text prompts.

        Args:
            prompt: Prompt text (list of str).
            max_new_tokens: Maximum number of new tokens to generate.
            tokenizer: Tokenizer to use for encoding/decoding.
            bos_token_id: Optional token ID that signals beginning-of-sequence.
            sep_token_id: Optional token ID that signals separation between prompt and generated text.
            eos_token_id: Optional token ID that signals end-of-sequence.
            pad_token_id: Optional token ID used for padding sequences.

        Returns:
            The generated text (str or list of str).
        """
        self.eval()
        bos_id = bos_token_id or getattr(tokenizer, "bos_token_id", None)
        sep_id = sep_token_id or getattr(tokenizer, "sep_token_id", None)
        eos_id = eos_token_id or getattr(tokenizer, "eos_token_id", None)
        pad_id = pad_token_id or getattr(tokenizer, "pad_token_id", 0)
        if bos_id is None or eos_id is None or sep_id is None:
            raise ValueError(
                "Tokenizer must have bos_token_id, sep_token_id, and eos_token_id defined, or they must be provided explicitly to the generate method."
            )

        encoded = []
        for text in prompts:
            ids = tokenizer.encode(text)
            ids = torch.tensor([bos_id] + ids + [sep_id], dtype=torch.long)
            encoded.append(ids.flip(0))

        param = next(self.parameters(), None)
        device = param.device if param is not None else "cpu"

        # left-pad sequences and flip back to (B, S) for processing
        input_ids = (
            pad_sequence(encoded, batch_first=True, padding_value=pad_id)
            .flip(1)
            .to(device)
        )

        generated = input_ids.clone()
        prompt_len = input_ids.size(1)

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                eos_mask = generated[:, -1:].eq(eos_token_id)
                next_token = torch.where(
                    eos_mask, torch.tensor(eos_token_id, device=device), next_token
                )
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and generated[:, -1].eq(eos_token_id).all():
                break

        new_tokens = generated[:, prompt_len:].tolist()
        decoded = [tokenizer.decode(tokens) for tokens in new_tokens]
        return decoded


class CausalLMCollateFn:
    """Collate sequences for decoder-only autoregressive training."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        pad_token_id: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        )
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        for item in batch:
            source_text = item["source_text"]
            target_text = item["target_text"]

            # Tokenize using the tokenizer
            source_ids = self.tokenizer.encode(source_text)
            target_ids = self.tokenizer.encode(target_text)

            # Format sequence: <bos> source <eos> target <eos>
            bos_t = torch.tensor([self.tokenizer.bos_token_id])
            eos_t = torch.tensor([self.tokenizer.eos_token_id])
            sep_t = torch.tensor([self.tokenizer.sep_token_id])
            src_t = torch.tensor(source_ids)
            tgt_t = torch.tensor(target_ids)

            src = torch.cat([bos_t, src_t, sep_t])
            tgt = torch.cat([tgt_t, eos_t])

            full_seq = torch.cat([src, tgt])

            input_ids = full_seq[:-1]
            labels = full_seq[1:].clone()

            # Mask the prompt portion in target labels
            labels[: len(src) - 1] = self.ignore_index

            # We flip the sequences to have padding on the left, which allows us to use causal masking without modification
            input_ids_list.append(input_ids.flip(0))
            labels_list.append(labels.flip(0))

        padded_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id if self.pad_token_id is not None else 0,
        ).flip(1)
        padded_labels = pad_sequence(
            labels_list, batch_first=True, padding_value=self.ignore_index
        ).flip(1)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
        }
