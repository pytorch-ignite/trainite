from pydantic import Field

from trainite.config.base import TrainerConfig


class PreTrainerConfig(TrainerConfig):
    epochs: int = Field(default=3, gt=0)
    log_every_steps: int = Field(default=10, gt=0)
    inference_every_epochs: int | None = Field(default=None, gt=0)
    inference_num_samples: int = Field(default=5, gt=0)
    max_inference_new_tokens: int = Field(default=16, gt=0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)
