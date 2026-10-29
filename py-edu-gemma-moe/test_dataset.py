"""Offline checks: python -m unittest test_dataset.py."""

import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset

from config import ProjectConfig
from dataset_impl.python_edu import (
    PythonEduTransform,
    extract_functions,
    function_spans,
    load_function_dataset,
    parse_source,
)
from models.gemma4_moe import CausalLMCollateFn
from utils import build_dataloaders, load_config


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
        examples = extract_functions(code)
        self.assertEqual(len(examples), 1)
        self.assertTrue(examples[0]["prompt"].startswith("async def add("))
        self.assertTrue(examples[0]["prompt"].endswith('    """\n'))
        self.assertEqual(examples[0]["completion"], "    # Keep indentation.\n    return a + b\n")
        item = PythonEduTransform(Tokenizer(), 512)(examples[0])
        self.assertTrue(item.source.startswith("async def add("))
        self.assertEqual(item.target, examples[0]["completion"])
        for invalid in [
            "not valid python!",
            'def f(): """Doc."""; return 1',
            'def f():\n    """Doc only."""\n',
            '@decorator\ndef f():\n    """Doc."""\n    return 1\n',
            'class C:\n    def f(self):\n        """Doc."""\n        return 1\n',
        ]:
            self.assertEqual(extract_functions(invalid), [])

    def test_extraction_keeps_only_self_contained_functions(self):
        source = (
            "import math\n\n"
            "LIMIT = 10\n\n"
            "def helper(x):\n"
            '    """Double x."""\n'
            "    return x * 2\n\n"
            "def uses_helper(x):\n"
            '    """Double x, capped."""\n'
            "    return min(helper(x), LIMIT)\n\n"
            "def uses_constant(x):\n"
            '    """Cap x."""\n'
            "    return min(x, LIMIT)\n\n"
            "def uses_nested_helper(x):\n"
            '    """Double x through a closure."""\n'
            "    def inner():\n"
            "        return helper(x)\n"
            "    return inner()\n\n"
            "def shadows_constant(x):\n"
            '    """Cap x with a local."""\n'
            "    LIMIT = 5\n"
            "    return min(x, LIMIT)\n\n"
            "def uses_import(x):\n"
            '    """Floor x."""\n'
            "    return math.floor(x)\n"
        )
        prompts = [example["prompt"] for example in extract_functions(source)]
        self.assertEqual(len(prompts), 3)
        self.assertTrue(any("def helper(" in prompt for prompt in prompts))
        self.assertTrue(any("def shadows_constant(" in prompt for prompt in prompts))
        self.assertTrue(any("def uses_import(" in prompt for prompt in prompts))
        self.assertFalse(any("def uses_helper(" in prompt for prompt in prompts))
        self.assertFalse(any("def uses_constant(" in prompt for prompt in prompts))
        self.assertFalse(any("def uses_nested_helper(" in prompt for prompt in prompts))

    def test_extraction_rejects_relative_imports(self):
        source = 'from .helper import helper\n\ndef f(x):\n    """Call helper."""\n    return helper(x)\n'
        self.assertEqual(extract_functions(source), [])

    def test_extraction_allows_only_allowlisted_imports(self):
        source = 'import numpy as np\n\ndef f(x):\n    """Use numpy."""\n    return np.abs(x)\n'
        self.assertEqual(len(extract_functions(source)), 1)
        for imports in (
            "import helpers",
            "from app.db import Base",
            "from . import helper",
            "from .numpy import array",
        ):
            with self.subTest(imports=imports):
                source = f'{imports}\n\ndef f(x):\n    """Use a helper."""\n    return x\n'
                self.assertEqual(extract_functions(source), [])

    def test_extraction_rejects_signature_dependencies(self):
        source = (
            "def helper():\n"
            '    """Return one."""\n'
            "    return 1\n\n"
            "class Marker:\n"
            "    pass\n\n"
            "def uses_default(x=helper()):\n"
            '    """Use a helper in a default."""\n'
            "    return x\n\n"
            "def uses_annotation(x: Marker) -> Marker:\n"
            '    """Use a class in annotations."""\n'
            "    return x\n\n"
            "def independent(x: int) -> int:\n"
            '    """Use only builtins."""\n'
            "    return x + 1\n"
        )
        prompts = [example["prompt"] for example in extract_functions(source)]
        self.assertEqual(len(prompts), 2)
        self.assertTrue(any("def helper(" in prompt for prompt in prompts))
        self.assertTrue(any("def independent(" in prompt for prompt in prompts))
        self.assertFalse(any("def uses_default(" in prompt for prompt in prompts))
        self.assertFalse(any("def uses_annotation(" in prompt for prompt in prompts))

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
        examples = extract_functions(source)
        self.assertEqual(len(examples), 1)
        item = transform(examples[0])
        self.assertIn("\u2028", item.source)
        self.assertTrue(item.source.endswith('    """\n'))
        self.assertEqual(item.target, "    value = 1\n    return value\n")

        raw = Dataset.from_dict({"text": [source]})
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw):
            dataset = load_function_dataset(f"python-edu-{uuid.uuid4()}", Tokenizer(), 512)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(transform(dataset[0]).target, item.target)

    def test_masking_and_collation(self):
        tokenizer = Tokenizer()
        transform = PythonEduTransform(tokenizer, max_length=256)
        row = {"prompt": 'def f():\n    """Return one."""\n', "completion": "    return 1\n"}
        item = transform(row)
        prompt_ids = tokenizer.encode(row["prompt"])
        body_ids = tokenizer.encode(row["completion"])
        n = len(prompt_ids)
        self.assertEqual(item.eval_input_ids.tolist(), prompt_ids)
        self.assertTrue((item.train_label_ids[: n - 1] == -100).all())
        self.assertEqual(item.train_label_ids[n - 1 :].tolist(), body_ids + [0])
        batch = CausalLMCollateFn(tokenizer=tokenizer)([item, transform({**row, "completion": "    return 123\n"})])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        logits = torch.randn(*batch["labels"].shape, 512, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for bad in [{}, {"prompt": ""}, {"completion": ""}, {"prompt": row["prompt"], "completion": 42}]:
            with self.assertRaises(ValueError):
                transform(bad)
        with self.assertRaises(ValueError):
            PythonEduTransform(tokenizer, max_length=2)(row)

    def test_dataset_loader(self):
        samples = [f'def f{i}():\n    """Return {i}."""\n    return {i}\n' for i in range(100)]
        raw = Dataset.from_dict({"text": samples + ["invalid python!"], "metadata": ["unused"] * 101})
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw) as load:
            dataset = load_function_dataset("existing-python-edu", Tokenizer(), 256)
            load.assert_called_once_with("existing-python-edu", split="train")
            self.assertEqual(len(dataset), 100)
            self.assertEqual(dataset.column_names, ["prompt", "completion"])
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
            patch("dataset_impl.python_edu.load_dataset", return_value=raw),
            patch(
                "dataset_impl.python_edu.PythonEduTransform", new=lambda *a, **k: self.fail("Loader called transform")
            ),
        ):
            dataset = load_function_dataset("python-edu", Tokenizer(), max_length=limit)
            self.assertEqual(dataset["prompt"], [prompt])
            self.assertEqual(dataset["completion"], [body])
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", Tokenizer(), max_length=limit - 1)
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", Tokenizer(), max_length=1)
            tokenizer = Tokenizer()
            tokenizer.eos_token_id = None
            with self.assertRaises(ValueError):
                load_function_dataset("python-edu", tokenizer)

    def test_loader_keeps_problematic_backslash_datapoint(self):
        problematic = (
            "def add(num1, num2):\r\n"
            '    """Returns num1 plus num2. """\r\n'
            "    return num1 + num2\r\n"
            "\r\n"
            "def div(num1, num2):\r\n"
            '    """Returns num1 divided by num2. """\r\n'
            "    return num1 / num2\\\r\n"
            "\r\n"
        )
        raw = Dataset.from_dict({"text": [problematic]})
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw):
            dataset = load_function_dataset(f"python-edu-{uuid.uuid4()}", Tokenizer(), 256)
        self.assertEqual(len(dataset), 2)
        self.assertIn("return num1 / num2\\", dataset[1]["completion"])
        PythonEduTransform(Tokenizer(), 256)(dataset[1])

    def test_import_context(self):
        imports = "import numpy as np\nfrom math import (\n    sin,\n    cos,\n)\n"
        function = 'def sig(x):\n    """Compute sigmoid."""\n    return 1 / (1 + np.exp(-x))\n'
        # Other source statements must not become part of the prompt.
        source = imports + "\nGLOBAL = 7\n\n" + function + "\nimport os\n"
        prefix = imports + "import os\n\n"
        raw = Dataset.from_dict({"text": [source]})
        examples = extract_functions(source)
        self.assertEqual(len(examples), 1)
        expected_prompt = prefix + function[: function.index("    return")]
        self.assertEqual(examples[0]["prompt"], expected_prompt)
        self.assertEqual(examples[0]["completion"], "    return 1 / (1 + np.exp(-x))\n")
        transform = PythonEduTransform(Tokenizer(), 512)
        item = transform(examples[0])
        self.assertEqual(item.source, expected_prompt)
        self.assertEqual(item.target, "    return 1 / (1 + np.exp(-x))\n")
        prompt_ids = Tokenizer().encode(expected_prompt)
        self.assertEqual(item.eval_input_ids.tolist(), prompt_ids)
        self.assertTrue((item.train_label_ids[: len(prompt_ids) - 1] == -100).all())
        limit = len(prompt_ids) + len(Tokenizer().encode(item.target)) + 1
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw):
            dataset = load_function_dataset("python-edu", Tokenizer(), max_length=limit)
            self.assertEqual(dataset[0]["prompt"], expected_prompt)
            self.assertEqual(dataset[0]["completion"], item.target)
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
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw):
            limited = load_function_dataset("python-edu", Tokenizer(), dataset_size=1)
            self.assertEqual(len(limited), 1)
            self.assertTrue(limited[0]["prompt"].startswith("def a"))
            for size, expected in ((None, 3), (2, 2), (100, 3)):
                subset = load_function_dataset("python-edu", Tokenizer(), dataset_size=size)
                self.assertEqual(len(subset), expected)
                if size == 2:
                    self.assertTrue(subset[1]["prompt"].startswith("def b"))
        with patch("dataset_impl.python_edu.load_dataset") as load:
            for size in (0, -1, True, 1.5, "10"):
                with self.assertRaises(ValueError):
                    load_function_dataset("python-edu", Tokenizer(), dataset_size=size)
            load.assert_not_called()

    def test_dataset_size_stops_processing(self):
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

        def tracked_spans(text, tree=None):
            visited.append(text)
            if text.startswith("def b"):
                raise AssertionError("Read past requested usable example count")
            return function_spans(text, tree)

        with (
            patch("dataset_impl.python_edu.load_dataset", return_value=raw),
            patch("dataset_impl.python_edu.function_spans", new=tracked_spans),
        ):
            # A fresh cache key keeps this test about generator early-stopping,
            # not about whether an earlier run already cached the same rows.
            dataset = load_function_dataset(f"python-edu-{uuid.uuid4()}", Tokenizer(), dataset_size=1)
            self.assertEqual(len(dataset), 1)
            self.assertFalse(any(text.startswith("def b") for text in visited))

    def test_config_dataloaders(self):
        config = load_config(Path(__file__).with_name("config.yaml"), ProjectConfig)
        raw = Dataset.from_dict({"text": [f'def f{i}():\n    """Return {i}."""\n    return {i}\n' for i in range(100)]})
        config.data.dataloader.num_workers = 0
        config.data.dataset.dataset_size = 50
        with patch("dataset_impl.python_edu.load_dataset", return_value=raw):
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
