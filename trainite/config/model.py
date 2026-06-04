from pydantic import ConfigDict, Field

from trainite.config.base import ComponentConfig


class TransformerModelConfig(ComponentConfig):
    model_config = ConfigDict(validate_assignment=True)
    target: str = Field(
        default="trainite.models.transformer.build_transformer_model", alias="_target_"
    )
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)
