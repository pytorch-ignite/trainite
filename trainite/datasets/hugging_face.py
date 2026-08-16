class HuggingFaceTransform:
    """Replace this transform with the task-specific conversion your model expects."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer: object = tokenizer

    def __call__(self, sample: dict[str, object]) -> object:
        raise NotImplementedError(
            "Implement HuggingFaceTransform.__call__ in data/hugging_face.py for your dataset and model"
        )
