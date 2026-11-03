"""Whole-file corpus: one causal-LM example per Python file.

The function-level pipeline (:mod:`dataset_impl.python_edu`) predicts a single body
from its signature. That forces every example to be self-contained, and the
self-containment check discards about a third of all top-level functions, because
they call a helper or read a constant defined elsewhere in the same file. Those are
exactly the functions that compose, so the filter removes the most useful signal.

This module keeps each file intact instead. The model sees imports, module
docstrings, helpers, classes and top-level code in their real order, and the loss
covers every token. At inference the task is unchanged: hand the model an import
block plus a signature and it continues, because that is a prefix of a file.

Differences from the function-level corpus, measured on 20k real source rows:

    example count      ~6.3M files instead of ~4.2M functions
    tokens per pass    ~3.6x the function-level budget at max_length 1024
    function bodies    only ~15% of file characters are inside def/class bodies;
                       the rest is imports, top-level code, comments and blank lines
    lengths            file median 152 tokens, p90 653, p99 2547; 94.9% fit in 1024
"""

from typing import Any

import torch
from datasets import Dataset, Features, Value, load_dataset

from dataset_impl.python_edu import (
    _ALLOWED_IMPORTS,
    _import_roots,
    DatapointModel,
    parse_source,
)


def extract_file(text: str) -> str | None:
    """Return the file source if it is eligible, otherwise ``None``.

    A file is eligible when it parses and every import is absolute and resolves to
    the standard library or the third-party allowlist. The same allowlist as the
    function-level corpus, so the two are comparable.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        tree = parse_source(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    imported_roots = _import_roots(tree)
    if imported_roots is None or not imported_roots <= _ALLOWED_IMPORTS:
        return None
    return text


def load_file_dataset(
    path: str,
    tokenizer: Any,
    max_length: int = 1024,
    split: str = "train",
    dataset_size: int | None = None,
    min_length: int = 16,
):
    """Index eligible files; the transform only tokenizes.

    Files longer than ``max_length`` are dropped rather than truncated, so no
    target ends mid-statement. ``min_length`` skips trivial files whose whole
    content is an import or a single constant.
    """
    if dataset_size is not None and (type(dataset_size) is not int or dataset_size < 1):
        raise ValueError("dataset_size must be a positive integer or None")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if min_length < 2:
        raise ValueError("min_length must be at least 2")
    if min_length > max_length:
        raise ValueError("min_length cannot exceed max_length")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    raw_dataset = load_dataset(path, split=split)

    def generate():
        count = 0
        for sample in raw_dataset:
            text = extract_file(sample["text"])
            if text is None:
                continue
            # +1 for the EOS the transform appends.
            length = len(tokenizer.encode(text, add_special_tokens=False)) + 1
            if length < min_length or length > max_length:
                continue
            yield {"text": text}
            count += 1
            if dataset_size is not None and count >= dataset_size:
                return

    # Keep source/tokenizer settings and output schema in the cache key. The
    # "file-level" marker guarantees this never collides with a function cache.
    source_fingerprint = getattr(raw_dataset, "_fingerprint", None) or "unknown"
    tokenizer_fingerprint = "|".join(
        str(getattr(tokenizer, attribute, ""))
        for attribute in ("name_or_path", "vocab_size", "eos_token_id", "pad_token_id")
    )
    fingerprint = (
        f"{path}|{split}|{dataset_size}|{max_length}|{source_fingerprint}|{tokenizer_fingerprint}|"
        f"file-level|stdlib-allowlist-v1|min-{min_length}"
    )
    dataset = Dataset.from_generator(
        generate,
        features=Features({"text": Value("string")}),
        fingerprint=fingerprint,
    )
    if len(dataset) == 0:
        raise ValueError("No usable files; inspect max_length and the import allowlist")
    return dataset


class FileCausalLMTransform:
    """Tokenize one file and predict every next token.

    Unlike the function-level transform there is no prompt mask: the loss covers
    the entire file plus the trailing EOS. That is what teaches the model to
    continue a file from any point, including from an import block plus a
    signature. The EOS is a real target too, so the model learns where a file ends.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Dataset rows must provide a non-empty text string")
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id")
        token_ids = text_ids + [eos]
        if len(token_ids) > self.max_length:
            raise ValueError("Example exceeds max_length; use the same dataset and transform token limit")
        if len(token_ids) < 2:
            raise ValueError("File must contain at least one token")

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        # The benchmark and sample-generation paths need a prompt to continue.
        # Use the first half of the file: imports plus the beginning of the code.
        split = max(1, len(input_ids) // 2)
        eval_ids = input_ids[:split].clone()
        return DatapointModel(
            source=self.tokenizer.decode(eval_ids.tolist(), skip_special_tokens=True),
            target="",
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=torch.ones_like(input_ids),
            eval_input_ids=eval_ids,
        )


# Re-exported so the config and tests can import the model type from here.
__all__ = ["FileCausalLMTransform", "extract_file", "load_file_dataset"]
