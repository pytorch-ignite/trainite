import torch


ALPHABET_PRESETS = {
    "@alpha": "abcdefghijklmnopqrstuvwxyz",
    "@digits": "0123456789",
    "@alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
}


class CharTokenizer:
    """A simple character-level tokenizer.

    Maps each character in the alphabet to a unique integer ID (1-indexed).
    ID 0 is reserved as the padding token.
    """

    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        self.alphabet = ALPHABET_PRESETS.get(alphabet, alphabet)
        self.pad_token_id = 0
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.alphabet)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary (excludes padding token)."""
        return len(self.alphabet)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert a list of token IDs back to a string, skipping padding tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            self.id_to_char[i] for i in ids if i != self.pad_token_id
        )


def build_tokenizer(
    alphabet: str = "abcdefghijklmnopqrstuvwxyz", **kwargs
) -> CharTokenizer:
    return CharTokenizer(alphabet=alphabet)
