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


class GemmaDenseModelConfig(ModelConfig):
    """Configuration for the GemmaDenseModel (Google DeepMind Gemma 4 dense causal LM)."""

    target: str = Field(default="trainite.models.gemma.GemmaDenseModel", alias="_target_")
    dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_kv_heads: int = Field(default=2, gt=0)
    head_dim: int = Field(default=16, gt=0)
    intermediate_size: int = Field(default=128, gt=0)
    sliding_window: int = Field(default=512, gt=0)
    local_rope_theta: float = Field(default=10000.0, gt=0.0)
    global_rope_theta: float = Field(default=1000000.0, gt=0.0)
    local_rope_proportion: float = Field(default=1.0, gt=0.0, le=1.0)
    global_rope_proportion: float = Field(default=0.25, gt=0.0, le=1.0)
    sliding_ratio: int = Field(default=5, ge=1)
    rms_norm_eps: float = Field(default=1e-6, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    final_logit_softcap: float | None = Field(default=None)
    tie_word_embeddings: bool = Field(default=True)
    max_seq_len: int = Field(default=512, gt=0)
