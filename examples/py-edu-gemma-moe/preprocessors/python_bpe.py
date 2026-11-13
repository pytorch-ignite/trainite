"""Byte-level BPE tokenizer trained on this experiment's raw corpus."""

from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# Qwen2.5-Coder's pre-tokenizer, copied verbatim from its tokenizer.json. On this
# corpus it needs 3.4% fewer tokens per function than GPT-2's regex (and 30% fewer
# than splitting every space), so the "code" arm uses it.
CODE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _pre_tokenizer(recipe: str) -> pre_tokenizers.PreTokenizer:
    if recipe == "gpt2":
        # Control arm: the built-in GPT-2 regex earlier runs used.
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    if recipe == "code":
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=Regex(CODE_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    raise ValueError(f"Unknown tokenizer recipe {recipe!r}; expected 'gpt2' or 'code'.")


def load_python_bpe_tokenizer(
    dataset_path: str,
    vocab_size: int = 8192,
    split: str = "train",
    cache_dir: str = "outputs/tokenizers",
    min_frequency: int = 2,
    recipe: str = "code",
) -> PreTrainedTokenizerFast:
    """Load a byte-level BPE tokenizer, training it once on the dataset text.

    Training happens on the raw ``text`` column before functions are filtered by
    token length, so the corpus is not biased by the tokenizer it produces. The
    cache file is keyed by dataset, split, vocabulary size, and tokenizer recipe,
    so repeat runs reuse the same token ids and a recipe change retrains instead
    of silently reusing a stale file.

    ``recipe`` selects only the pre-tokenizer: ``"gpt2"`` is the baseline this
    experiment started from, ``"code"`` is the code-aware one. Everything else
    (vocabulary size, special tokens, trainer, corpus) is shared, so the two arms
    differ by exactly one factor. Special tokens follow the earlier runs: ``<eos>``
    is trained in and ``<pad>`` appended, so ``len(tokenizer) == vocab_size + 1``
    and the embedding has one extra row.
    """
    tokenizer_file = (
        Path(cache_dir)
        / f"{dataset_path.replace('/', '__')}-{split.replace('/', '__')}-{vocab_size}-{min_frequency}-{recipe}.json"
    )
    if not tokenizer_file.exists():
        from datasets import load_dataset

        corpus = (
            sample["text"] for sample in load_dataset(dataset_path, split=split) if isinstance(sample["text"], str)
        )

        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = _pre_tokenizer(recipe)
        tokenizer.decoder = decoders.ByteLevel()

        # Only <eos> is trained in; <pad> is appended below, as the reference run
        # did, so the control arm is that recipe exactly. The embedding therefore
        # has vocab_size + 1 rows and pad_token_id == vocab_size.
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
