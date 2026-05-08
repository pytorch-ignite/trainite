from __future__ import annotations

from pydantic import BaseModel


class TransformerModelConfig(BaseModel):
    vocab_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    max_seq_len: int = 128


MODEL_CONFIGS: dict[str, type[BaseModel]] = {
    "transformer": TransformerModelConfig,
}
