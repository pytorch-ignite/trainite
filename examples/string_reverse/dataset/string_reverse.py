import string

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Hardcoded universal vocabulary: all printable ASCII characters
UNIVERSAL_VOCAB = string.ascii_letters + string.digits + string.punctuation + " "


class CharTokenizer:
    """A simple character-level tokenizer with a hardcoded universal vocabulary.

    Maps each character in UNIVERSAL_VOCAB to a unique integer ID.
    ID 0 is reserved as the padding token.
    ID 1 is reserved as the BOS token.
    ID 2 is reserved as the EOS token.
    ID 3 is reserved as the UNK token.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.char_to_id = {c: i + 4 for i, c in enumerate(UNIVERSAL_VOCAB)}
        self.id_to_char = {i + 4: c for i, c in enumerate(UNIVERSAL_VOCAB)}

        self.special_tokens = {
            self.pad_token_id: "<pad>",
            self.bos_token_id: "<bos>",
            self.eos_token_id: "<eos>",
            self.unk_token_id: "<unk>",
        }

        for k, v in self.special_tokens.items():
            self.id_to_char[k] = v

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (includes special tokens)."""
        return len(UNIVERSAL_VOCAB) + 4

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


class StringReverseDataset(Dataset):
    def __init__(
        self,
        size: int,
        min_seq_len: int,
        max_seq_len: int,
        seed: int,
        seq_len: int | None = None,
        charset: str | None = None,
    ) -> None:
        self.tokenizer = CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # If charset is not provided, use the full universal vocab
        if charset is None or charset == "@universal":
            chars = UNIVERSAL_VOCAB
        # Map charset presets or raw strings to token IDs
        elif charset == "@alpha":
            chars = string.ascii_letters
        elif charset == "@digits":
            chars = string.digits
        elif charset == "@alphanumeric":
            chars = string.ascii_letters + string.digits
        else:
            chars = charset

        self.valid_token_ids = [
            self.tokenizer.char_to_id[c]
            for c in chars
            if c in self.tokenizer.char_to_id
        ]

        if not self.valid_token_ids:
            raise ValueError(f"Charset '{charset}' resulted in empty token IDs.")

        generator = torch.Generator().manual_seed(seed)
        self.valid_token_ids_tensor = torch.tensor(self.valid_token_ids)

        self.inputs = []
        self.labels = []

        for _ in range(size):
            if seq_len is not None and (min_seq_len or max_seq_len):
                raise ValueError(
                    "Cannot specify both seq_len and min_seq_len/max_seq_len."
                )
            if seq_len is not None:
                length = seq_len
            elif min_seq_len is None or max_seq_len is None:
                raise ValueError(
                    "Must specify either seq_len or both min_seq_len and max_seq_len."
                )
            else:
                length = torch.randint(
                    low=min_seq_len,
                    high=max_seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()

            # Sample from the valid token IDs
            indices = torch.randint(
                low=0,
                high=len(self.valid_token_ids),
                size=(length,),
                generator=generator,
            )
            seq = self.valid_token_ids_tensor[indices]

            reversed_seq = torch.flip(seq, dims=[0])

            bos_t = torch.tensor([self.tokenizer.bos_token_id])
            eos_t = torch.tensor([self.tokenizer.eos_token_id])

            full_seq = torch.cat([bos_t, seq, eos_t, reversed_seq, eos_t])

            input_ids = full_seq[:-1]
            target_labels = full_seq[1:].clone()

            prompt_len = len(seq) + 2
            target_labels[: prompt_len - 1] = -100

            self.inputs.append(input_ids)
            self.labels.append(target_labels)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }

    def decode(
        self,
        ids: torch.Tensor,
        skip_special_tokens: bool = True,
        ignore_index: int = -100,
    ) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens, ignore_index=ignore_index
        )


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
    }


def build_string_reverse_dataset(
    size: int = 256,
    min_seq_len: int = 1,
    max_seq_len: int = 16,
    seq_len: int | None = None,
    charset: str | None = None,
    seed: int = 7,
    **kwargs,
) -> StringReverseDataset:
    # Handle 'alphabet' if passed from old configs for backward compatibility or just ignore
    if "alphabet" in kwargs and charset is None:
        charset = kwargs["alphabet"]

    return StringReverseDataset(
        size=size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        seed=seed,
        seq_len=seq_len,
        charset=charset,
    )
