"""Helpers for the EvalPlus HumanEval+ mini evaluation."""

import json
from pathlib import Path

PASS = "pass"


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
