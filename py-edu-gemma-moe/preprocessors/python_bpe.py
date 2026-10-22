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
    reuse the same token ids instead of retraining.

    Byte-level BPE round-trips arbitrary source text, including indentation and
    unseen characters, without unknown tokens.
    """
    tokenizer_file = (
        Path(cache_dir)
        / f"{dataset_path.replace('/', '__')}-{split.replace('/', '__')}-{vocab_size}-{min_frequency}.json"
    )
    if not tokenizer_file.exists():
        from datasets import load_dataset

        corpus = (
            sample["text"] for sample in load_dataset(dataset_path, split=split) if isinstance(sample["text"], str)
        )
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
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
        pad_token="<eos>",
    )
