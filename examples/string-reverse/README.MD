# String-Reverse Example

A simple sequence-to-sequence task where a Transformer learns to reverse strings composed of characters. This example provides a complete, standalone template for training models on synthetic symbolic tasks.

## Getting Started

This example was generated using the Trainite CLI:

```bash
uv run trainite init examples/string-reverse --model transformer --dataset string-reverse -y
```

All files in this directory are self-contained and do not depend on the `trainite` library. You can modify them freely for your own experiments.

## Files

| File | Description |
|---|---|
| `config.yaml` | Hyperparameters including model dimensions, variable sequence lengths, and alphabet shortcuts |
| `config.py` | Pydantic classes for type-safe configuration loading |
| `models/transformer.py` | Transformer architecture with positional encoding and multi-head attention |
| `dataset/string-reverse.py` | Dataset logic for generating character sequences and their reversals |
| `trainer.py` | Full training loop with automated metric logging and checkpointing |
| `main.py` | Main entry point to load config and start training |

## Running Training

From the project root (`trainite/`):

```bash
cd examples/string-reverse
uv run python main.py config.yaml
```

### Expected Output

```text
2026-05-15 16:21:43 [INFO] trainer: starting run in outputs/transformer__string-reverse/...
2026-05-15 16:21:44 [INFO] trainer: Evaluating on training set...
2026-05-15 16:21:44 [INFO] trainer: Evaluating on validation set...
2026-05-15 16:21:44 [INFO] trainer: epoch=1 train_loss=3.3850 train_acc=0.0461 val_loss=3.3971 val_acc=0.0500
...
```

## Task Description

The model is trained to reverse a sequence of characters from a defined alphabet.

- **Input**: A character sequence, e.g. `['a', 'b', 'c', 'd']`
- **Output**: The reversed sequence, e.g. `['d', 'c', 'b', 'a']`
- **Vocabulary**: Characters defined in the `alphabet` config string (supports shortcuts like `@alpha`, `@digits`).
- **Variable Length**: The dataset supports sequences of varying lengths, using a padding token (Index 0) to align batches.

## Configuration

The task is customizable via `config.yaml`. The `_target_` paths point to the model and dataset builder functions, so you can swap either component without changing the trainer code.

`dataset.vocab_size` is derived from `dataset.alphabet` and is included as a reference for the model configuration. By default, `model.vocab_size` uses `${dataset.vocab_size}` so the embedding/output layers match the dataset vocabulary.
For out-of-distribution experiments with larger alphabets or reserved tokens, increase `model.vocab_size` above `dataset.vocab_size`.

Example for a variable-length alphanumeric task:

```yaml
model:
  _target_: "models.transformer.build_transformer_model"
  vocab_size: ${dataset.vocab_size}
  hidden_size: 64

dataset:
  _target_: "dataset.string-reverse.build_string_reverse_dataloaders"
  alphabet: "@alphanumeric"
  vocab_size: 36 # derived from alphabet; keep as a reference for model size
  min_seq_len: 4
  max_seq_len: 32
  fixed_length: false
```
