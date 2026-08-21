from typing import Any
import torch
from pydantic import BaseModel, ConfigDict


class DatapointModel(BaseModel):
    """Item contract consumed by Trainite's causal-LM collate function and trainer."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # Human-readable prompt and expected continuation used in inference logs.
    source: str
    target: str

    # One-dimensional, equally sized tensors used for next-token training.
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor  # Use -100 for label positions that loss should ignore.
    attention_mask: torch.Tensor  # 1 for real tokens, 0 for padding.

    # Prompt-only token IDs passed to model.generate() during inference logging.
    eval_input_ids: torch.Tensor


class HuggingFaceTransform:
    """Replace this transform with the task-specific conversion your model expects."""

    def __init__(self, tokenizer: Any, max_length: int = 128, ignore_index: int = -100) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.tokenizer: Any = tokenizer
        self.max_length: int = max_length
        self.ignore_index: int = ignore_index

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        raise NotImplementedError(
            "Implement HuggingFaceTransform.__call__ in dataset_impl/hugging_face.py for your dataset and model"
        )
