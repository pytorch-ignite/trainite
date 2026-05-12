from pydantic import BaseModel


class PreTrainerConfig(BaseModel):
    device: str = "cpu"
    type: str = "pre"
    epochs: int = 3
    learning_rate: float = 1e-3
    log_every_steps: int = 10
    grad_clip_norm: float | None = None


TRAINER_CONFIGS: dict[str, type[BaseModel]] = {
    "pretrainer": PreTrainerConfig,
}
