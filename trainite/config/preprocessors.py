from pydantic import Field
from trainite.config.base import PreprocessorConfig


class CharTokenizerConfig(PreprocessorConfig):
    target: str = Field(
        default="trainite.preprocessors.char_tokenizer.CharTokenizer",
        alias="_target_",
    )
    charset: str = "@universal"
