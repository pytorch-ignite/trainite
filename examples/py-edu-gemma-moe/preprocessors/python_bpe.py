"""Byte-level BPE tokenizer trained on this experiment's raw corpus."""

from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers
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
    cache file is keyed by dataset, split, vocabulary size, and tokenizer recipe,
    so repeat runs reuse the same token ids and a recipe change retrains instead
    of silently reusing a stale file.
    """
    tokenizer_file = (
        Path(cache_dir)
        / f"{dataset_path.replace('/', '__')}-{split.replace('/', '__')}-{vocab_size}-{min_frequency}-code.json"
    )
    if not tokenizer_file.exists():
        from datasets import load_dataset

        corpus = (
            sample["text"] for sample in load_dataset(dataset_path, split=split) if isinstance(sample["text"], str)
        )

        tokenizer = Tokenizer(models.BPE(unk_token=None))

        # Keep identifiers whole, isolate digits, and group whitespace runs so
        # Python indentation becomes a single token.
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                # A plain string is a literal in tokenizers; the regex must be wrapped.
                pre_tokenizers.Split(pattern=Regex(r"\d| +|\n+|[^\s\d\w]+|\w+"), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Special tokens are trained in, so vocab_size == len(tokenizer) and every
        # id, including <pad>, fits the embedding.
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<eos>"],
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
