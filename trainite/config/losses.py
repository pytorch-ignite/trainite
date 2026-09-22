from pydantic import Field

from trainite.config.base import LossConfig


class CrossEntropyLossConfig(LossConfig):
    """Typed config for the default loss (copy-paste template for custom losses).

    A custom loss (e.g. MoE load-balancing, defined next to its model) follows
    the same shape: subclass LossConfig, set ``target`` to the loss
    implementation, declare its params with types/defaults. Point the model's
    ``ModelSpec.loss_config_cls_path`` at it and ``init`` emits it into
    config.yaml automatically.
    """

    target: str = Field(default="torch.nn.CrossEntropyLoss", alias="_target_")
    ignore_index: int = -100
