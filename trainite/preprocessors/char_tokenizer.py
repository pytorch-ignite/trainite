import string
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

# Hardcoded universal vocabulary: all printable ASCII characters
UNIVERSAL_VOCAB = string.ascii_letters + string.digits + string.punctuation + " "


class CharTokenizer:
    """A simple character-level tokenizer with a hardcoded universal vocabulary.
    Consists of all printable ASCII characters, plus special tokens for padding, beginning of sequence, separator, end of sequence, and unknown characters.

    Maps each character in UNIVERSAL_VOCAB to a unique integer ID.
    ID 0 is reserved as the <PAD> token.
    ID 1 is reserved as the <BOS> token.
    ID 2 is reserved as the <SEP> token.
    ID 3 is reserved as the <EOS> token.
    ID 4 is reserved as the <UNK> token.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 3
        self.unk_token_id = 4

        self.char_to_id: dict[str, int] = {c: i + 5 for i, c in enumerate(UNIVERSAL_VOCAB)}
        self.id_to_char: dict[int, str] = {i + 5: c for i, c in enumerate(UNIVERSAL_VOCAB)}

        self.special_tokens: dict[int, str] = {
            self.pad_token_id: "<PAD>",
            self.bos_token_id: "<BOS>",
            self.sep_token_id: "<SEP>",
            self.eos_token_id: "<EOS>",
            self.unk_token_id: "<UNK>",
        }

        for k, v in self.special_tokens.items():
            self.id_to_char[k] = v

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(UNIVERSAL_VOCAB) + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs, mapping unrecognized characters to UNK."""
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: List or tensor of token IDs to decode.
            skip_special_tokens: If True, skips printing special tokens like <bos> and <eos>.
            ignore_index: The token ID used for loss masking. These are silently ignored.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        decoded_chars = []
        for i in ids:
            if i == ignore_index:
                continue
            if i in self.special_tokens:
                if not skip_special_tokens or i == self.unk_token_id:
                    decoded_chars.append(self.special_tokens[i])
            elif i in self.id_to_char:
                decoded_chars.append(self.id_to_char[i])
            else:
                decoded_chars.append(self.special_tokens[self.unk_token_id])
        return "".join(decoded_chars)

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        if isinstance(text, str):
            is_batched = False
            texts = [text]
        else:
            is_batched = True
            texts = text

        batch_input_ids = []
        batch_attention_mask = []

        for t in texts:
            ids = self.encode(t)
            if add_special_tokens:
                prefix = [self.bos_token_id]
                suffix = [self.eos_token_id]
                ids = prefix + ids + suffix

            if truncation and max_length is not None:
                ids = ids[:max_length]

            batch_input_ids.append(ids)
            batch_attention_mask.append([1] * len(ids))

        if padding:
            if padding == "max_length" and max_length is not None:
                target_len = max_length
            else:
                target_len = max(len(ids) for ids in batch_input_ids)

            for i in range(len(batch_input_ids)):
                current_len = len(batch_input_ids[i])
                if current_len < target_len:
                    pad_len = target_len - current_len
                    batch_input_ids[i] = [self.pad_token_id] * pad_len + batch_input_ids[i]
                    batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
                elif current_len > target_len:
                    batch_input_ids[i] = batch_input_ids[i][-target_len:]
                    batch_attention_mask[i] = batch_attention_mask[i][-target_len:]

        if not is_batched:
            out = {
                "input_ids": batch_input_ids[0],
                "attention_mask": batch_attention_mask[0],
            }
        else:
            out = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
            }

        if return_tensors == "pt":
            tensor_out = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }
            return tensor_out

        return out


class CharTokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    target: str = Field(
        default="trainite.preprocessors.char_tokenizer.CharTokenizer",
        alias="_target_",
    )
