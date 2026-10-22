"""Offline tokenizer checks: python -m unittest test_tokenizer.py."""

import tempfile
import unittest
from unittest.mock import patch

from preprocessors.python_bpe import load_python_bpe_tokenizer

CORPUS = [
    {"text": "def add(a, b):\n    return a + b\n\n"},
    {"text": "def greet(name):\n    return 'hello ' + name\n\n"},
] * 50


class PythonBpeTokenizerTest(unittest.TestCase):
    def test_trains_once_and_roundtrips(self):
        text = "def add(a, b):\n    return a + b\n"
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("datasets.load_dataset", return_value=CORPUS) as load_dataset:
                tokenizer = load_python_bpe_tokenizer("tiny/corpus", vocab_size=300, cache_dir=cache_dir)
            load_dataset.assert_called_once_with("tiny/corpus", split="train")
            self.assertEqual(tokenizer.eos_token_id, tokenizer.pad_token_id)
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)

            with patch("datasets.load_dataset", side_effect=AssertionError("cache miss retrained")):
                cached = load_python_bpe_tokenizer("tiny/corpus", vocab_size=300, cache_dir=cache_dir)
            self.assertEqual(cached.encode(text, add_special_tokens=False), ids)


if __name__ == "__main__":
    unittest.main()
