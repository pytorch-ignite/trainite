"""Offline HumanEval result aggregation checks: python -m unittest test_humaneval.py."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.humaneval import strip_docstring, summarize_results


class HumanEvalStripDocstringTest(unittest.TestCase):
    def test_removes_docstring_and_keeps_imports_and_signature(self):
        prompt = (
            "from typing import List\n"
            "\n"
            "\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """Check whether any two numbers are closer than the threshold.\n'
            "\n"
            "    >>> has_close_elements([1.0, 2.0], 0.5)\n"
            "    True\n"
            '    """\n'
        )
        self.assertEqual(
            strip_docstring(prompt),
            "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n",
        )

    def test_leaves_prompts_without_docstring_or_parseable_code_alone(self):
        documented_body = "def f():\n    x = 1\n    return x\n"
        self.assertEqual(strip_docstring(documented_body), documented_body)
        not_python = "this is not python"
        self.assertEqual(strip_docstring(not_python), not_python)


class HumanEvalSummaryTest(unittest.TestCase):
    def test_counts_base_and_plus_passes(self):
        evaluations = {
            "eval": {
                "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
                "HumanEval/1": [{"base_status": "pass", "plus_status": "fail"}],
                "HumanEval/2": [{"base_status": "fail", "plus_status": "fail"}],
            }
        }
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "results.json"
            path.write_text(json.dumps(evaluations))
            summary = summarize_results(path)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["base_passed"], 2)
        self.assertEqual(summary["plus_passed"], 1)
        self.assertAlmostEqual(summary["base_pass_at_1"], 2 / 3)
        self.assertAlmostEqual(summary["plus_pass_at_1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
