from pydantic import Field, ConfigDict
from trainite.config.base import PreprocessorConfig


class CharTokenizerConfig(PreprocessorConfig):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(
        default="trainite.preprocessors.char_tokenizer.CharTokenizer",
        alias="_target_",
    )
    charset: str = "@universal"
