"""Helpers for the EvalPlus HumanEval+ mini evaluation."""

import ast
import json
from pathlib import Path

PASS = "pass"


def strip_docstring(prompt: str) -> str:
    """Remove a HumanEval prompt's docstring, keeping imports and the signature.

    Used to measure how much the model depends on the docstring. The returned
    prompt ends at the signature, so generation starts where the body belongs.
    Prompts without a parseable docstring are returned unchanged.
    """
    try:
        tree = ast.parse(prompt)
    except (SyntaxError, ValueError, RecursionError):
        return prompt
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        doc = node.body[0]
        if not isinstance(doc, ast.Expr):
            continue
        if not (isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str)):
            continue
        lines = prompt.splitlines(keepends=True)
        return "".join(lines[: doc.lineno - 1] + lines[doc.end_lineno :])
    return prompt


def summarize_results(result_path: Path) -> dict[str, float | int]:
    """Aggregate one-sample-per-task EvalPlus results into pass@1 rates."""
    evaluations = json.loads(Path(result_path).read_text())["eval"]
    total = len(evaluations)
    base_passed = sum(row[0]["base_status"] == PASS for row in evaluations.values())
    plus_passed = sum(row[0]["base_status"] == PASS and row[0]["plus_status"] == PASS for row in evaluations.values())
    return {
        "total": total,
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "base_pass_at_1": base_passed / total if total else 0.0,
        "plus_pass_at_1": plus_passed / total if total else 0.0,
    }
