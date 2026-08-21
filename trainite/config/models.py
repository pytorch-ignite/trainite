from typing import Literal

from pydantic import Field

from trainite.config.base import ModelConfig


class BasicTransformerModelConfig(ModelConfig):
    """Configuration for the BasicTransformerModel (absolute positional encoding)."""

    target: Literal[
        "trainite.models.basic_transformer.BasicTransformerModel",
        "models.basic_transformer.BasicTransformerModel",
    ] = Field(default="trainite.models.basic_transformer.BasicTransformerModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)


class RoPETransformerModelConfig(ModelConfig):
    """Configuration for the RoPETransformerModel (rotary positional embeddings)."""

    target: Literal[
        "trainite.models.rope_transformer.RoPETransformerModel",
        "models.rope_transformer.RoPETransformerModel",
    ] = Field(default="trainite.models.rope_transformer.RoPETransformerModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)
