from pydantic import Field

from trainite.config.base import ComponentConfig


class TransformerModelConfig(ComponentConfig):
    target: str = Field(
        default="trainite.models.transformer.build_transformer_model", alias="_target_"
    )
    vocab_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    max_seq_len: int = 128
