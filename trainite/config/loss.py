from trainite.config.base import LossConfig
from pydantic import Field


class Gemma4MoELoss(LossConfig):
    target: str = Field(
        default="trainite.models.gemma4_moe.GemmaMoECausalLMLoss",
        alias="_target_",
    )
    ignore_index: int = -100
    aux_loss_coef: float = 0.01


class BasicTransformerLoss(LossConfig):
    target: str = Field(default="trainite.models.basic_transformer.CausalLMCrossEntropyLoss", alias="_target_")


class RoPETransformerLoss(LossConfig):
    target: str = Field(default="trainite.models.rope_transformer.CausalLMCrossEntropyLoss", alias="_target_")


class Gemma4DenseLoss(LossConfig):
    target: str = Field(default="trainite.models.gemma4_dense.CausalLMCrossEntropyLoss", alias="_target_")
