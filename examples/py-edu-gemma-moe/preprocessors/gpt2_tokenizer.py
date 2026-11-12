from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_gpt2_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
