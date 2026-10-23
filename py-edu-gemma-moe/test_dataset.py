"""Offline checks: python -m unittest test_dataset.py."""

import unittest
import uuid
import warnings

import torch

from dataset_impl.python_edu import PythonEduTransform
from models.gemma4_moe import CausalLMCollateFn
from dataset_impl.python_edu import function_spans, load_function_dataset, parse_source
from unittest.mock import patch
from datasets import Dataset


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]


class DatasetTest(unittest.TestCase):
    def test_source_warning_suppression(self):
        source = 'def f():\n    """A path: ' + chr(92) + 'P"""\n    return 1\n'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(len(function_spans(source)), 1)
            parse_source(source)
            self.assertEqual(caught, [])
            # The local filter must not silence warnings outside source parsing.
            warnings.warn("invalid escape sequence", SyntaxWarning)
            self.assertEqual(len(caught), 1)
        with self.assertRaises(SyntaxError):
            parse_source("def broken(")

        def other_warning(text):
            warnings.warn("invalid octal escape sequence", SyntaxWarning)
            warnings.warn("invalid decimal literal", SyntaxWarning)
            warnings.warn("unrelated warning", UserWarning)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("dataset_impl.python_edu.ast.parse", side_effect=other_warning):
                parse_source(source)
            self.assertEqual(len(caught), 1)
            self.assertIs(caught[0].category, UserWarning)

    def test_extraction(self):
        code = 'async def add(\n    a, b,\n):\n    """Add two numbers.\n\n    Preserve values.\n    """\n    # Keep indentation.\n    return a + b\n\ndef undocumented():\n    return 1\n'
        rows = [PythonEduTransform(Tokenizer(), 512)({"text": code})]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].source.startswith("async def add("))
        self.assertTrue(rows[0].source.endswith('    """\n'))
        self.assertEqual(rows[0].target, "    # Keep indentation.\n    return a + b\n")
        for invalid in [
            "not valid python!",
            'def f(): """Doc."""; return 1',
            'def f():\n    """Doc only."""\n',
            '@decorator\ndef f():\n    """Doc."""\n    return 1\n',
            'class C:\n    def f(self):\n        """Doc."""\n        return 1\n',
        ]:
            self.assertEqual(function_spans(invalid), [])

    def test_unicode_line_separator_in_docstring(self):
        source = (
            "import random\n\n"
            "def sample():\n"
            '    """First line.\u2028Second line.\n\n'
            "    More documentation.\n"
            '    """\n'
            "    value = 1\n"
            "    return value\n"
        )
        transform = PythonEduTransform(Tokenizer(), 512)
        item = transform({"text": source})
        self.assertIn("\u2028", item.source)
        self.assertTrue(item.source.endswith('    """\n'))
        self.assertEqual(item.target, "    value = 1\n    return value\n")

        raw = Dataset.from_dict({"text": [source]})
        with patch("datasets.load_dataset", return_value=raw):
            dataset = load_function_dataset(f"python-edu-{uuid.uuid4()}", Tokenizer(), 512)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(transform(dataset[0]).target, item.target)

    def test_masking_and_collation(self):
        tokenizer = Tokenizer()
        transform = PythonEduTransform(tokenizer, max_length=256)
        row = {"prompt": 'def f():\n    """Return one."""\n', "completion": "    return 1\n"}
        item = transform({"text": row["prompt"] + row["completion"]})
        prompt_ids = tokenizer.encode(row["prompt"])
        body_ids = tokenizer.encode(row["completion"])
        n = len(prompt_ids)
        self.assertEqual(item.eval_input_ids.tolist(), prompt_ids)
        self.assertTrue((item.train_label_ids[: n - 1] == -100).all())
        self.assertEqual(item.train_label_ids[n - 1 :].tolist(), body_ids + [0])
        batch = CausalLMCollateFn(tokenizer=tokenizer)([item, transform({"text": row["prompt"] + "    return 123\n"})])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        logits = torch.randn(*batch["labels"].shape, 512, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for bad in [{"text": ""}, {"text": row["prompt"]}, {"text": 42}]:
            with self.assertRaises(ValueError):
                transform(bad)
        with self.assertRaises(ValueError):
            PythonEduTransform(tokenizer, max_length=2)({"text": row["prompt"] + row["completion"]})

    def test_dataset_loader(self):
        samples = [f'def f{i}():\n    """Return {i}."""\n    return {i}\n' for i in range(100)]
        raw = Dataset.from_dict({"text": samples + ["invalid python!"], "metadata": ["unused"] * 101})
        with patch("datasets.load_dataset", return_value=raw) as load:
            dataset = load_function_dataset("existing-python-edu", Tokenizer(), 256)
            load.assert_called_once_with("existing-python-edu", split="train")
            self.assertEqual(len(dataset), 100)
            self.assertEqual(dataset.column_names, ["text"])
            for row in dataset:
                PythonEduTransform(Tokenizer(), 256)(row)
            with self.assertRaises(ValueError):
                load_function_dataset("existing-python-edu", Tokenizer(), 2)

    def test_loader_filters_without_transform(self):
        short = 'def f():\n    """Return one."""\n    return 1\n'
        long = 'def g():\n    """Return text."""\n    return "' + "x" * 500 + '"\n'
        raw = Dataset.from_dict({"text": [short, long]})
        prompt = short[: short.index("    return")]
        body = short[len(prompt) :]
        limit = len(Tokenizer().encode(prompt)) + len(Tokenizer().encode(body)) + 1
        with (
            patch("datasets.load_dataset", return_value=raw),
            patch(
                "dataset_impl.python_edu.PythonEduTransform", new=lambda *a, **k: self.fail("Loader called transform")
            ),
        ):
            dataset = load_function_dataset("python-edu", Tokenizer(), max_length=limit)
            self.assertEqual(dataset["text"], [short])
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", Tokenizer(), max_length=limit - 1)
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", Tokenizer(), max_length=1)
            tokenizer = Tokenizer()
            tokenizer.eos_token_id = None
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", tokenizer)

    def test_import_context(self):
        imports = "import numpy as np\nfrom math import (\n    sin,\n    cos,\n)\n"
        function = 'def sig(x):\n    """Compute sigmoid."""\n    return 1 / (1 + np.exp(-x))\n'
        # Other source statements must not become part of the prompt.
        source = imports + "\nGLOBAL = 7\n\n" + function + "\nimport os\n"
        prefix = imports + "import os\n\n"
        raw = Dataset.from_dict({"text": [source]})
        transform = PythonEduTransform(Tokenizer(), 512)
        item = transform({"text": source})
        expected_prompt = prefix + function[: function.index("    return")]
        self.assertEqual(item.source, expected_prompt)
        self.assertEqual(item.target, "    return 1 / (1 + np.exp(-x))\n")
        prompt_ids = Tokenizer().encode(expected_prompt)
        self.assertEqual(item.eval_input_ids.tolist(), prompt_ids)
        self.assertTrue((item.train_label_ids[: len(prompt_ids) - 1] == -100).all())
        limit = len(prompt_ids) + len(Tokenizer().encode(item.target)) + 1
        with patch("datasets.load_dataset", return_value=raw):
            dataset = load_function_dataset("python-edu", Tokenizer(), max_length=limit)
            self.assertEqual(dataset[0]["text"], prefix + function)
            self.assertEqual(transform(dataset[0]).source, expected_prompt)
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", Tokenizer(), max_length=limit - 1)

    def test_dataset_size(self):
        raw = Dataset.from_dict(
            {
                "text": [
                    "invalid python!",
                    'def oversized():\n    """Long."""\n    return "' + "x" * 600 + '"\n',
                    'def a():\n    """A."""\n    return 1\n\ndef b():\n    """B."""\n    return 2\n',
                    'def c():\n    """C."""\n    return 3\n',
                ]
            }
        )
        with patch("datasets.load_dataset", return_value=raw):
            limited = load_function_dataset("python-edu", Tokenizer(), dataset_size=1)
            self.assertEqual(len(limited), 1)
            self.assertTrue(limited[0]["text"].startswith("def a"))
            for size, expected in ((None, 3), (2, 2), (100, 3)):
                subset = load_function_dataset("python-edu", Tokenizer(), dataset_size=size)
                self.assertEqual(len(subset), expected)
                if size == 2:
                    self.assertTrue(subset[1]["text"].startswith("def b"))
        with patch("datasets.load_dataset") as load:
            for size in (0, -1, True, 1.5, "10"):
                with self.assertRaises(ValueError):
                    load_function_dataset("python-edu", Tokenizer(), dataset_size=size)
            load.assert_not_called()

    def test_dataset_size_stops_processing(self):
        from dataset_impl.python_edu import function_spans as original_spans

        raw = Dataset.from_dict(
            {
                "text": [
                    "invalid python!",
                    'def a():\n    """A."""\n    return 1\n',
                    'def b():\n    """B."""\n    return 2\n',
                ]
            }
        )
        visited = []

        def tracked_spans(text):
            visited.append(text)
            if text.startswith("def b"):
                raise AssertionError("Read past requested usable example count")
            return original_spans(text)

        with (
            patch("datasets.load_dataset", return_value=raw),
            patch("dataset_impl.python_edu.function_spans", new=tracked_spans),
        ):
            # A fresh cache key keeps this test about generator early-stopping,
            # not about whether an earlier run already cached the same rows.
            dataset = load_function_dataset(f"python-edu-{uuid.uuid4()}", Tokenizer(), dataset_size=1)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(visited), 2)

    def test_config_dataloaders(self):
        from pathlib import Path
        from config import ProjectConfig
        from utils import load_config, build_dataloaders

        config = load_config(Path(__file__).with_name("config.yaml"), ProjectConfig)
        raw = Dataset.from_dict({"text": [f'def f{i}():\n    """Return {i}."""\n    return {i}\n' for i in range(100)]})
        config.data.dataloader.num_workers = 0
        config.data.dataset.dataset_size = 50
        with patch("datasets.load_dataset", return_value=raw):
            loaders = build_dataloaders(config.data, Tokenizer(), 42, config.model.collate_fn_target)
            self.assertEqual([len(loader.dataset) for loader in loaders], [40, 5, 5])
            load_indices = [set(loader.dataset.indices) for loader in loaders]
            self.assertFalse(load_indices[0] & load_indices[1])
            self.assertFalse(load_indices[0] & load_indices[2])
            self.assertFalse(load_indices[1] & load_indices[2])
            for loader in loaders:
                batch = next(iter(loader))
                self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
                self.assertTrue((batch["labels"] != -100).any())


if __name__ == "__main__":
    unittest.main()
