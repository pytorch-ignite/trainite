# String-Reverse Example

A simple sequence-to-sequence task where a Transformer learns to reverse strings composed of characters. This example provides a complete, standalone template for training models on synthetic symbolic tasks.

## Getting Started

This example was generated using the Trainite CLI:

```bash
uv run trainite init examples/string-reverse --model transformer --dataset string-reverse --tokenizer char -y
```

All files in this directory are self-contained and do not depend on the `trainite` library. You can modify them freely for your own experiments.

## Files

| File | Description |
|---|---|
| `config.yaml` | Hyperparameters including model dimensions, sequence lengths, and tokenizer settings |
| `config.py` | Pydantic classes for type-safe configuration loading |
| `tokenizers/char.py` | Character-level tokenizer with encode/decode and alphabet presets |
| `models/transformer.py` | Transformer architecture with positional encoding and multi-head attention |
| `dataset/string-reverse.py` | Dataset logic for generating token sequences and their reversals |
| `trainer.py` | Full training loop with automated metric logging and checkpointing |
| `main.py` | Main entry point to load config, build tokenizer, and start training |
| `utils.py` | Shared utilities including the `instantiate()` factory |

## Running Training

From the project root (`trainite/`):

```bash
cd examples/string-reverse
uv run python main.py config.yaml
```

### Expected Output

```text
2026-05-15 16:21:43 [INFO] trainer: starting run in output/transformer__string-reverse/...
2026-05-15 16:21:44 [INFO] trainer: Evaluating on training set...
2026-05-15 16:21:44 [INFO] trainer: Evaluating on validation set...
2026-05-15 16:21:44 [INFO] trainer: epoch=1 train_loss=3.3850 train_acc=0.0461 val_loss=3.3971 val_acc=0.0500
...
```

## Task Description

The model is trained to reverse a sequence of characters from a defined alphabet.

- **Input**: A character sequence, e.g. `['a', 'b', 'c', 'd']`
- **Output**: The reversed sequence, e.g. `['d', 'c', 'b', 'a']`
- **Vocabulary**: Characters defined in the tokenizer's `alphabet` config (supports presets like `@alpha`, `@digits`, `@alphanumeric`).
- **Variable Length**: The dataset supports sequences of varying lengths, using a padding token (ID 0) to align batches.

## Configuration

The task is highly customizable via `config.yaml`. The modular architecture uses `_target_` paths to allow you to swap your own models, datasets, or tokenizers without changing the trainer code.

The tokenizer is the single source of truth for vocabulary — the model's embedding size and the dataset's token range are both derived from it at runtime.

Example for a variable-length alphanumeric task:

```yaml
tokenizer:
  _target_: "tokenizers.char.build_tokenizer"
  alphabet: "@alphanumeric"

model:
  _target_: "models.transformer.build_transformer_model"
  hidden_size: 64

dataset:
  _target_: "dataset.string-reverse.build_string_reverse_dataloaders"
  min_seq_len: 4
  max_seq_len: 32
  fixed_length: false
```
