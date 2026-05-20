from trainite.config.base import TrainerConfig


class PreTrainerConfig(TrainerConfig):
    epochs: int = 3
    learning_rate: float = 1e-3
    log_every_steps: int = 10
    grad_clip_norm: float | None = None
