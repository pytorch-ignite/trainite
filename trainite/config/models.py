from pydantic import Field

from trainite.config.base import ModelConfig


class BasicTransformerModelConfig(ModelConfig):
    """Configuration for the BasicTransformerModel (absolute positional encoding)."""

    target: str = Field(default="trainite.models.basic_transformer.BasicTransformerModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)


class RoPETransformerModelConfig(ModelConfig):
    """Configuration for the RoPETransformerModel (rotary positional embeddings)."""

    target: str = Field(default="trainite.models.rope_transformer.RoPETransformerModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=2, gt=0)
    feedforward_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_len: int = Field(default=128, gt=0)


class Gemma4MoEModelConfig(ModelConfig):
    """Small Gemma 4 MoE configuration suitable for training from scratch."""

    target: str = Field(default="trainite.models.gemma4_moe.Gemma4TextModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_attention_heads: int = Field(default=4, gt=0)
    num_key_value_heads: int = Field(default=2, gt=0)
    dense_intermediate_size: int = Field(default=128, gt=0)
    expert_dim: int = Field(default=64, gt=0)
    num_experts: int = Field(default=4, gt=0)
    top_k: int = Field(default=2, gt=0)
    head_dim: int = Field(default=16, gt=0)
    layer_types: tuple[str, ...] = ("sliding", "global")
    sliding_window: int = Field(default=64, gt=0)
    global_num_key_value_heads: int = Field(default=2, gt=0)
    global_head_dim: int = Field(default=16, gt=0)
    global_key_equals_value: bool = True
