import math
import string
from typing import Any

import torch
from pydantic import Field
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from trainite.config.base import ComponentConfig

# Hardcoded universal vocabulary: all printable ASCII characters
UNIVERSAL_VOCAB = string.ascii_letters + string.digits + string.punctuation + " "

CHARSET_PRESETS = {
    "@universal": UNIVERSAL_VOCAB,
    "@alpha": string.ascii_letters,
    "@digits": string.digits,
    "@alphanumeric": string.ascii_letters + string.digits,
}


class CharTokenizer:
    """A simple character-level tokenizer with a hardcoded universal vocabulary.

    Maps each character in UNIVERSAL_VOCAB to a unique integer ID.
    ID 0 is reserved as the <PAD> token.
    ID 1 is reserved as the <BOS> token.
    ID 2 is reserved as the <SEP> token.
    ID 3 is reserved as the <EOS> token.
    ID 4 is reserved as the <UNK> token.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 3
        self.unk_token_id = 4

        self.char_to_id: dict[str, int] = {
            c: i + 5 for i, c in enumerate(UNIVERSAL_VOCAB)
        }
        self.id_to_char: dict[int, str] = {
            i + 5: c for i, c in enumerate(UNIVERSAL_VOCAB)
        }

        self.special_tokens: dict[int, str] = {
            self.pad_token_id: "<PAD>",
            self.bos_token_id: "<BOS>",
            self.sep_token_id: "<SEP>",
            self.eos_token_id: "<EOS>",
            self.unk_token_id: "<UNK>",
        }

        for k, v in self.special_tokens.items():
            self.id_to_char[k] = v

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(UNIVERSAL_VOCAB) + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs, mapping unrecognized characters to UNK."""
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: List or tensor of token IDs to decode.
            skip_special_tokens: If True, skips printing special tokens like <bos> and <eos>.
            ignore_index: The token ID used for loss masking. These are silently ignored.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        decoded_chars = []
        for i in ids:
            if i == ignore_index:
                continue
            if i in self.special_tokens:
                if not skip_special_tokens or i == self.unk_token_id:
                    decoded_chars.append(self.special_tokens[i])
            elif i in self.id_to_char:
                decoded_chars.append(self.id_to_char[i])
            else:
                decoded_chars.append(self.special_tokens[self.unk_token_id])
        return "".join(decoded_chars)

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(text, str):
            is_batched = False
            texts = [text]
        else:
            is_batched = True
            texts = text

        batch_input_ids = []
        batch_attention_mask = []

        for t in texts:
            ids = self.encode(t)
            if add_special_tokens:
                prefix = [self.bos_token_id]
                suffix = [self.eos_token_id]
                ids = prefix + ids + suffix

            if truncation and max_length is not None:
                ids = ids[:max_length]

            batch_input_ids.append(ids)
            batch_attention_mask.append([1] * len(ids))

        if padding:
            if padding == "max_length" and max_length is not None:
                target_len = max_length
            else:
                target_len = max(len(ids) for ids in batch_input_ids)

            for i in range(len(batch_input_ids)):
                current_len = len(batch_input_ids[i])
                if current_len < target_len:
                    pad_len = target_len - current_len
                    batch_input_ids[i] = [
                        self.pad_token_id
                    ] * pad_len + batch_input_ids[i]
                    batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
                elif current_len > target_len:
                    batch_input_ids[i] = batch_input_ids[i][-target_len:]
                    batch_attention_mask[i] = batch_attention_mask[i][-target_len:]

        if not is_batched:
            out = {
                "input_ids": batch_input_ids[0],
                "attention_mask": batch_attention_mask[0],
            }
        else:
            out = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
            }

        if return_tensors == "pt":
            tensor_out = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }
            return tensor_out

        return out

    def apply_chat_template(
        self,
        conversation: str,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> Any:
        if isinstance(conversation, str):
            prompt = conversation
        else:
            raise ValueError("Unsupported conversation format")

        if tokenize:
            ids = self.encode(prompt)
            prefix = [self.bos_token_id]
            suffix = [self.sep_token_id]
            return prefix + ids + suffix
        else:
            return f"<BOS>{prompt}<SEP>"


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
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        cos, sin = self.rotary_emb(x, seq_len=S)
        if attention_mask is not None:
            padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)
        elif (input_ids == self.embedding.padding_idx).any():
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
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate text token IDs from prompt input_ids.

        Expects a pre-padded tensor (B, S). The caller is responsible for
        left-padding and providing an attention mask before calling this method.

        Args:
            input_ids: Prompt token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            attention_mask: Optional attention mask of shape (batch, seq_len).
            eos_token_id: Optional token ID that signals end-of-sequence.
            pad_token_id: Optional token ID used for padding sequences.

        Returns:
            Tensor containing the full token IDs (prompt + newly generated tokens)
            of shape (batch, prompt_len + new_tokens).
        """
        self.eval()
<<<<<<< HEAD

        eos_id = eos_token_id

        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self(generated, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if eos_id is not None:
                eos_mask = generated[:, -1:].eq(eos_id)
                next_token = torch.where(eos_mask, torch.tensor(eos_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                next_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                )
                if eos_id is not None:
                    already_ended = generated[:, -2:-1].eq(eos_id)
                    next_mask = torch.where(
                        already_ended,
                        torch.tensor(0, dtype=attention_mask.dtype, device=device),
                        next_mask,
                    )
                attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            if eos_id is not None and generated[:, -1].eq(eos_id).all():
                break

        return generated


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

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        for item in batch:
            prompt = item.get("prompt")
            completion = item.get("completion")

            if prompt is None or completion is None:
                raise KeyError(
                    f"Dataset item must contain 'prompt' and 'completion' keys. Got: {list(item.keys())}"
                )

            # Tokenize using the tokenizer
            prompt_ids = self.tokenizer.encode(prompt)
            completion_ids = self.tokenizer.encode(completion)

            # Format sequence: <bos> source <eos> target <eos>
            bos_t = torch.tensor([self.tokenizer.bos_token_id])
            eos_t = torch.tensor([self.tokenizer.eos_token_id])
            sep_t = torch.tensor([self.tokenizer.sep_token_id])
            src_t = torch.tensor(prompt_ids)
            tgt_t = torch.tensor(completion_ids)

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
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index).flip(1)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
        }


class TransformerModelConfig(ComponentConfig):
    target: str = Field(default="trainite.models.transformer.TransformerModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)
