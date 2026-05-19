from pydantic import Field

from trainite.config.base import ComponentConfig


class CharTokenizerConfig(ComponentConfig):
    target: str = Field(
        default="trainite.tokenizers.char_tokenizer.build_tokenizer",
        alias="_target_",
    )
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
