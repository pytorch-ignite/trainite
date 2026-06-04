from pydantic import Field

from trainite.config.base import TrainerConfig


class PreTrainerConfig(TrainerConfig):
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)
