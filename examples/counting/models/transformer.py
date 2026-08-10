from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_token_id: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        out_dim = num_classes if num_classes is not None else vocab_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def generate_future_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids shape: (batch_size, seq_len)
        x_embed = self.embedding(input_ids)  # (batch_size, seq_len, hidden_size)

        # Generate causal mask for self-attention
        seq_len = input_ids.size(1)
        tgt_mask = self.generate_future_mask(seq_len, device=input_ids.device)

        # Generate dummy zero-memory tensor for cross-attention
        memory = torch.zeros_like(x_embed)

        # Build padding mask if batch size is > 1 and padding exists
        tgt_key_padding_mask = None
        if attention_mask is not None:
            if not attention_mask.all().item():
                tgt_key_padding_mask = ~attention_mask.to(torch.bool)

        # Pass through decoder
        x_decoded = self.decoder(
            tgt=x_embed, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Final projection to classes (binary targets)
        return self.proj(x_decoded)


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
