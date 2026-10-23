import ast
import re
import warnings
from typing import Any
import torch
from pydantic import BaseModel, ConfigDict

_SOURCE_LINE_RE = re.compile(r"\r\n|\r|\n")


def split_source_lines(text: str) -> list[str]:
    """Split on LF/CRLF/CR, matching CPython's line numbering.

    ``str.splitlines`` also treats Unicode separators such as U+2028 and U+2029 as
    line breaks, while Python's parser does not. Using it here would misalign AST
    line numbers and truncate functions whose docstrings contain those characters.
    """
    lines = []
    start = 0
    for match in _SOURCE_LINE_RE.finditer(text):
        lines.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


def parse_source(text: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def function_spans(text: str) -> list[tuple[int, int, int]]:
    """Index eligible top-level functions as start/docstring-end/function-end lines."""
    try:
        tree = parse_source(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    spans = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not ast.get_docstring(node) or len(node.body) < 2:
            continue
        doc = node.body[0]
        # shortcut: omit decorated functions and inline suites; add context-aware extraction if needed.
        if node.decorator_list or node.body[1].lineno <= doc.end_lineno:
            continue
        spans.append((node.lineno - 1, doc.end_lineno, node.end_lineno))
    return spans


def import_prefix(text: str) -> str:
    """Preserve top-level import statements without executing or resolving them."""
    try:
        tree = parse_source(text)
    except (SyntaxError, ValueError, RecursionError):
        return ""
    imports = [
        ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return "\n".join(imports) + "\n\n" if imports else ""


def load_function_dataset(
    path: str,
    tokenizer: Any,
    max_length: int = 512,
    split: str = "train",
    dataset_size: int | None = None,
):
    """Index raw functions; the transform owns prompt/body formatting and tokenization."""
    from datasets import Dataset, Features, Value, load_dataset

    if dataset_size is not None and (type(dataset_size) is not int or dataset_size < 1):
        raise ValueError("dataset_size must be a positive integer or None")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    raw_dataset = load_dataset(path, split=split)

    def generate():
        count = 0
        for sample in raw_dataset:
            text = sample["text"]
            if not isinstance(text, str):
                continue
            lines = split_source_lines(text)
            imports = import_prefix(text)
            for start, doc_end, end in function_spans(text):
                # Match the transform's separately encoded prompt/body lengths, including EOS.
                prompt_length = len(tokenizer.encode(imports + "".join(lines[start:doc_end]), add_special_tokens=False))
                body_length = len(tokenizer.encode("".join(lines[doc_end:end]), add_special_tokens=False))
                if not prompt_length or not body_length or prompt_length + body_length + 1 > max_length:
                    continue
                yield {"text": imports + "".join(lines[start:end])}
                count += 1
                if dataset_size is not None and count >= dataset_size:
                    return

    # The generator closes over the source dataset and filtering options.  Letting
    # `datasets` cache it by generator code alone can silently reuse a stale split.
    # Include the source fingerprint and filtering/tokenizer settings in the cache key.
    source_fingerprint = getattr(raw_dataset, "_fingerprint", None) or "unknown"
    tokenizer_fingerprint = "|".join(
        str(getattr(tokenizer, attribute, ""))
        for attribute in ("name_or_path", "vocab_size", "eos_token_id", "pad_token_id")
    )
    fingerprint = f"{path}|{split}|{dataset_size}|{max_length}|{source_fingerprint}|{tokenizer_fingerprint}"
    dataset = Dataset.from_generator(
        generate,
        features=Features({"text": Value("string")}),
        fingerprint=fingerprint,
    )
    if len(dataset) == 0:
        raise ValueError("No usable functions; inspect docstrings and max_length")
    return dataset


class DatapointModel(BaseModel):
    """Tokenized function-completion example."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class PythonEduTransform:
    """Train on function bodies conditioned on imports, signatures and docstrings."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample: dict[str, object]) -> DatapointModel:
        text = sample["text"]
        if not isinstance(text, str):
            raise ValueError("text must be a Python source string")
        spans = function_spans(text)
        if len(spans) != 1:
            raise ValueError("Expected exactly one eligible documented function")
        start, doc_end, end = spans[0]
        lines = split_source_lines(text)
        prompt = import_prefix(text) + "".join(lines[start:doc_end])
        completion = "".join(lines[doc_end:end])
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        body_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not prompt_ids or not body_ids:
            raise ValueError("prompt and completion must both contain tokens")
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id")
        token_ids = prompt_ids + body_ids + [eos]
        if len(token_ids) > self.max_length:
            raise ValueError("Example exceeds max_length; use the same dataset and transform token limit")
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        labels[: len(prompt_ids) - 1] = -100
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        return DatapointModel(
            source=prompt,
            target=completion,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=torch.ones_like(input_ids),
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        )
