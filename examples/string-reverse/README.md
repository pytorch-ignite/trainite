# String-Reverse Example

A simple sequence-to-sequence task where a Transformer learns to reverse a string of integers.

## How This Was Generated

This example was scaffolded using the Trainite CLI:

```bash
uv run trainite init examples/string-reverse --model transformer --dataset string-reverse
```

This will prompt you for the output directory and run name. To skip prompts and use defaults, add the `-y` flag:

```bash
uv run trainite init examples/string-reverse --model transformer --dataset string-reverse -y
```

All files are self-contained with zero `trainite` imports — you can edit them freely.

## Files

| File | Description |
|---|---|
| `config.yaml` | Hyperparameters (epochs, batch size, model dimensions, etc.) |
| `config.py` | Pydantic config classes for type-safe configuration |
| `model.py` | Encoder-style Transformer with positional encoding and multi-head attention |
| `dataset.py` | Generates random integer sequences and their reversals |
| `trainer.py` | Full training loop using PyTorch-Ignite with metrics and checkpointing |
| `main.py` | Entrypoint — loads config, builds trainer, runs training |

## Running

From the project root (`trainite/`):

```bash
cd examples/string-reverse
uv run python main.py config.yaml
```

### Expected Output

```
[INFO] trainer: starting run in output/transformer__string-reverse/...
[INFO] trainer: Evaluating on training set...
[INFO] trainer: Evaluating on validation set...
[INFO] trainer: epoch=1 train_loss=3.5801 train_acc=0.0315 val_loss=3.5680 val_acc=0.0371
...
```

## Monitoring with TensorBoard

```bash
uv run tensorboard --logdir output
```

Then open http://localhost:6006 in your browser.

## Configuration

The default config runs 3 epochs as a quick smoke test. To train for longer, edit `config.yaml`:

```yaml
trainer:
  epochs: 50

dataset:
  train_size: 2048
```

## Task Description

- **Input**: A random sequence of integers, e.g. `[5, 12, 3, 28, 7]`
- **Output**: The reversed sequence, e.g. `[7, 28, 3, 12, 5]`
- **Vocabulary**: Integers from `0` to `vocab_size - 1` (default: 32)
- **Sequence Length**: Configurable via `seq_len` (default: 8)
