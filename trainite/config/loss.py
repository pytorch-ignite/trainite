from trainite.config.base import LossConfig
from pydantic import Field


class Gemma4MoELossConfig(LossConfig):
    target: str = Field(
        default="trainite.models.gemma4_moe.Gemma4MoELoss",
        alias="_target_",
    )
    ignore_index: int = -100
    aux_loss_coef: float = 0.1
