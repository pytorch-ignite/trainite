"""Offline HumanEval result aggregation checks: python -m unittest test_humaneval.py."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.humaneval import summarize_results


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
