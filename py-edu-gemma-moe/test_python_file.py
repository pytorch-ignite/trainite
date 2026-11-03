"""Offline checks: python -m unittest test_python_file.py."""

import unittest
import uuid
from unittest.mock import patch

from datasets import Dataset

from dataset_impl.python_file import FileCausalLMTransform, extract_file, load_file_dataset


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 1) for i in ids if not (skip_special_tokens and i == self.eos_token_id))


SOURCE = 'import math\n\n\ndef area(r):\n    """Circle area."""\n    return math.pi * r * r\n'


class ExtractFileTest(unittest.TestCase):
    def test_keeps_files_whose_imports_are_allowlisted(self):
        self.assertEqual(extract_file(SOURCE), SOURCE)
        self.assertEqual(extract_file("def f():\n    return 1\n"), "def f():\n    return 1\n")

    def test_rejects_relative_and_unlisted_imports(self):
        self.assertIsNone(extract_file("from . import x\n"))
        self.assertIsNone(extract_file("from .local import thing\n"))
        self.assertIsNone(extract_file("import django\n"))

    def test_rejects_unusable_input(self):
        self.assertIsNone(extract_file("def broken(\n"))
        self.assertIsNone(extract_file("   \n\n"))
        self.assertIsNone(extract_file(None))  # type: ignore[arg-type]


class FileCausalLMTransformTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.transform = FileCausalLMTransform(self.tokenizer, max_length=256)

    def test_predicts_every_token_with_no_prompt_mask(self):
        item = self.transform({"text": SOURCE})
        text_ids = self.tokenizer.encode(SOURCE)
        # EOS is appended and becomes the final target; nothing is masked.
        self.assertEqual(item.train_input_ids.tolist(), text_ids)
        self.assertEqual(item.train_label_ids.tolist(), text_ids[1:] + [self.tokenizer.eos_token_id])
        self.assertTrue((item.train_label_ids != -100).all())
        self.assertEqual(item.train_input_ids.shape, item.train_label_ids.shape)
        self.assertEqual(item.attention_mask.tolist(), [1] * len(text_ids))

    def test_eval_prompt_is_a_prefix_of_the_file(self):
        item = self.transform({"text": SOURCE})
        prefix = item.eval_input_ids.tolist()
        self.assertGreaterEqual(len(prefix), 1)
        self.assertEqual(prefix, item.train_input_ids.tolist()[: len(prefix)])
        self.assertEqual(item.source, self.tokenizer.decode(prefix))

    def test_rejects_files_over_max_length(self):
        long_text = "x = 1\n" + "# padding\n" * 100
        with self.assertRaises(ValueError):
            self.transform({"text": long_text})

    def test_rejects_empty_rows(self):
        with self.assertRaises(ValueError):
            self.transform({"text": ""})
        with self.assertRaises(ValueError):
            self.transform({})


class LoadFileDatasetTest(unittest.TestCase):
    def test_filters_by_length_and_allowlist(self):
        short = "import os\n\nprint(os.getcwd())\n"
        too_short = "x = 1\n"
        too_long = "y = '" + "z" * 300 + "'\n"
        relative = "from . import thing\n"
        raw = Dataset.from_dict({"text": [short, too_short, too_long, relative]})
        path = f"python-edu-file-{uuid.uuid4()}"
        with patch("dataset_impl.python_file.load_dataset", return_value=raw) as load:
            dataset = load_file_dataset(path, Tokenizer(), max_length=128, min_length=8)
            load.assert_called_once_with(path, split="train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.column_names, ["text"])
        self.assertEqual(dataset[0]["text"], short)
        FileCausalLMTransform(Tokenizer(), max_length=128)(dataset[0])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            load_file_dataset("p", Tokenizer(), dataset_size=0)
        with self.assertRaises(ValueError):
            load_file_dataset("p", Tokenizer(), max_length=1)
        with self.assertRaises(ValueError):
            load_file_dataset("p", Tokenizer(), max_length=64, min_length=128)

    def test_cache_key_does_not_collide_with_function_level(self):
        raw = Dataset.from_dict({"text": [SOURCE]})
        with patch("dataset_impl.python_file.load_dataset", return_value=raw):
            first = load_file_dataset("same-path", Tokenizer(), max_length=256)
            second = load_file_dataset("same-path", Tokenizer(), max_length=128)
        self.assertNotEqual(first.cache_files, second.cache_files)
        self.assertIn("file-level", first.cache_files[0]["filename"])


if __name__ == "__main__":
    unittest.main()
