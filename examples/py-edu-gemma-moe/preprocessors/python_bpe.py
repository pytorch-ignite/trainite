"""Byte-level BPE tokenizer trained on this experiment's raw corpus."""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def load_python_bpe_tokenizer(
    dataset_path: str,
    vocab_size: int = 8192,
    split: str = "train",
    cache_dir: str = "outputs/tokenizers",
    min_frequency: int = 2,
) -> PreTrainedTokenizerFast:
    """Load a byte-level BPE tokenizer, training it once on the dataset text.

    Training happens on the raw ``text`` column before functions are filtered by
    token length, so the corpus is not biased by the tokenizer it produces. The
    cache file is keyed by dataset, split, and vocabulary size, so repeat runs
    reuse the same token ids instead of silently retraining.

    The pre-tokenizer is the standard GPT-2 regex, matching the earlier run so
    its results stay comparable; a code-aware pre-tokenizer measured only 1.7%
    fewer tokens per function on this corpus, which is not worth a second arm.

    Special tokens follow the earlier runs: ``<eos>`` is trained in and ``<pad>``
    appended, so ``len(tokenizer) == vocab_size + 1`` and the embedding has one
    extra row.
    """
    stem = f"{dataset_path.replace('/', '__')}-{split.replace('/', '__')}-{vocab_size}-{min_frequency}"
    tokenizer_file = Path(cache_dir) / f"{stem}.json"
    if not tokenizer_file.exists():
        from datasets import load_dataset

        corpus = (
            sample["text"] for sample in load_dataset(dataset_path, split=split) if isinstance(sample["text"], str)
        )

        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # Only <eos> is trained in; <pad> is appended below, as the reference run
        # did. The embedding therefore has vocab_size + 1 rows and
        # pad_token_id == vocab_size.
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        tokenizer_file.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_file))

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        eos_token="<eos>",
        pad_token="<pad>",
    )
