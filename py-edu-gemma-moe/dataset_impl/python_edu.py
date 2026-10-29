import ast
import re
import symtable
import sys
import warnings
from typing import Any

import torch
from datasets import Dataset, Features, Value, load_dataset
from pydantic import BaseModel, ConfigDict

_SOURCE_LINE_RE = re.compile(r"\r\n|\r|\n")
_ALLOWED_IMPORTS = frozenset(sys.stdlib_module_names) | {
    "PIL",
    "bs4",
    "cv2",
    "keras",
    "matplotlib",
    "mpl_toolkits",
    "networkx",
    "nltk",
    "numpy",
    "pandas",
    "pygame",
    "requests",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorflow",
    "torch",
}


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


def _import_roots(tree: ast.Module) -> set[str] | None:
    """Return imported root modules, or None if any import is relative."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return None
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _symtable(text: str) -> symtable.SymbolTable | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return symtable.symtable(text, "<source>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        return None


def _module_names(text: str) -> set[str] | None:
    """Return module-level names that are not imports."""
    table = _symtable(text)
    if table is None:
        return None
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if (symbol.is_assigned() or symbol.is_namespace()) and not symbol.is_imported()
    }


def _referenced_globals(table: symtable.SymbolTable) -> set[str]:
    """Return module-level names read by a scope or any nested scope."""
    names = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_global() and symbol.is_referenced()}
    for child in table.get_children():
        names |= _referenced_globals(child)
    return names


def function_spans(text: str, tree: ast.Module | None = None) -> list[tuple[int, int, int]]:
    """Index eligible top-level functions as start/docstring-end/function-end lines."""
    if tree is None:
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


def import_prefix(text: str, tree: ast.Module | None = None) -> str:
    """Preserve top-level import statements without executing or resolving them."""
    if tree is None:
        try:
            tree = parse_source(text)
        except (SyntaxError, ValueError, RecursionError):
            return ""
    imports = [
        ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return "\n".join(imports) + "\n\n" if imports else ""


def extract_functions(text: str) -> list[dict[str, str]]:
    """Extract validated, self-contained prompt/completion pairs from one raw source row."""
    if not isinstance(text, str):
        return []
    try:
        tree = parse_source(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    imported_roots = _import_roots(tree)
    if imported_roots is None or not imported_roots <= _ALLOWED_IMPORTS:
        return []
    module_names = _module_names(text)
    if module_names is None:
        return []
    lines = split_source_lines(text)
    imports = import_prefix(text, tree)
    examples = []
    for start, doc_end, end in function_spans(text, tree):
        # Include lines consumed by a trailing backslash continuation.
        while end < len(lines) and lines[end - 1].rstrip("\r\n").endswith("\\"):
            end += 1
        prompt = imports + "".join(lines[start:doc_end])
        completion = "".join(lines[doc_end:end])
        if not prompt or not completion:
            continue

        combined = prompt + completion
        try:
            combined_tree = parse_source(combined)
        except (SyntaxError, ValueError, RecursionError):
            continue
        functions = [node for node in combined_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1 or len(function_spans(combined, combined_tree)) != 1:
            continue
        table = _symtable(combined)
        if table is None:
            continue
        imported = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_imported()}
        if _referenced_globals(table) & (module_names - imported - {functions[0].name}):
            continue
        examples.append({"prompt": prompt, "completion": completion})
    return examples


def load_function_dataset(
    path: str,
    tokenizer: Any,
    max_length: int = 512,
    split: str = "train",
    dataset_size: int | None = None,
):
    """Index raw functions; the transform only tokenizes precomputed pairs."""
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
            for example in extract_functions(sample["text"]):
                prompt_length = len(tokenizer.encode(example["prompt"], add_special_tokens=False))
                body_length = len(tokenizer.encode(example["completion"], add_special_tokens=False))
                if not prompt_length or not body_length or prompt_length + body_length + 1 > max_length:
                    continue
                yield example
                count += 1
                if dataset_size is not None and count >= dataset_size:
                    return

    # Keep source/tokenizer settings and output schema in the cache key.
    source_fingerprint = getattr(raw_dataset, "_fingerprint", None) or "unknown"
    tokenizer_fingerprint = "|".join(
        str(getattr(tokenizer, attribute, ""))
        for attribute in ("name_or_path", "vocab_size", "eos_token_id", "pad_token_id")
    )
    fingerprint = (
        f"{path}|{split}|{dataset_size}|{max_length}|{source_fingerprint}|{tokenizer_fingerprint}|"
        "prompt-completion|self-contained|stdlib-allowlist-v1"
    )
    dataset = Dataset.from_generator(
        generate,
        features=Features({"prompt": Value("string"), "completion": Value("string")}),
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
        prompt = sample.get("prompt")
        completion = sample.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError("Dataset rows must provide prompt and completion strings")
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
