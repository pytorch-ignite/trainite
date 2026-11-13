"""Byte-level BPE tokenizer trained on this experiment's raw corpus."""

from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# Qwen2.5-Coder's pre-tokenizer, copied verbatim from its tokenizer.json. On this
# corpus it needs 1.7% fewer tokens per function than GPT-2's regex, so Python
# indentation and operators stay grouped instead of splitting per byte.
CODE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


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

    Special tokens follow the earlier runs: ``<eos>`` is trained in and ``<pad>``
    appended, so ``len(tokenizer) == vocab_size + 1`` and the embedding has one
    extra row.
    """
    # The "-code" suffix keeps tokenizers fitted with the old split-every-space
    # recipe (`...-8192-2.json`) from being reused.
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
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=Regex(CODE_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
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
