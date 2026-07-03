from pydantic import Field, ConfigDict
from typing import Literal
from trainite.config.base import PreprocessorConfig


class CharTokenizerConfig(PreprocessorConfig):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: Literal[
        "trainite.preprocessors.char_tokenizer.CharTokenizer", "preprocessors.char_tokenizer.CharTokenizer"
    ] = Field(
        default="trainite.preprocessors.char_tokenizer.CharTokenizer",
        alias="_target_",
    )
