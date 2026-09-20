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


class Gemma4DenseModelConfig(ModelConfig):
    """Small Gemma 4 Dense configuration suitable for training from scratch."""

    target: str = Field(default="trainite.models.gemma4_moe.Gemma4DenseModel", alias="_target_")
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_attention_heads: int = Field(default=4, gt=0)
    num_key_value_heads: int = Field(default=2, gt=0)
    dense_intermediate_size: int = Field(default=128, gt=0)
    head_dim: int = Field(default=16, gt=0)
    layer_types: tuple[str, ...] | None = None
    sliding_window: int = Field(default=512, gt=0)
    sliding_ratio: int = Field(default=5, ge=1)
    rope_theta: float = Field(default=10000.0, gt=0.0)
    global_rope_theta: float = Field(default=1000000.0, gt=0.0)
    rotary_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    global_rotary_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    global_num_key_value_heads: int = Field(default=2, gt=0)
    global_head_dim: int = Field(default=16, gt=0)
    global_key_equals_value: bool = True
    tie_word_embeddings: bool = True
    final_logit_softcap: float | None = None
    max_seq_len: int = Field(default=512, gt=0)
