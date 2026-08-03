# String Reversal Example: Exploring Small Transformers with Trainite

A small decoder-only Transformer experiment that learns to reverse character strings:

```text
input:  abc123
target: 321cba
```

The example is designed for studying how sequence length, model depth, and model width affect exact string reversal.

The base tranite project which was used and modified here for the use case was generated using `trainite==0.1.0` with command

```bash
trainite init string_reversal --dataset string-reverse
```

## Task and data

`StringReverseDataset` generates random strings from a configurable character set. The default uses letters and digits (`@alphanumeric`) with lengths from 1 to 16.

The decoder receives a prompt followed by the expected completion:

```text
<BOS> abc123 <SEP> 321cba <EOS>
```

Only completion tokens contribute to the training loss. The dataset is automatically split into training, validation, and test sets using the ratios in `config.yaml`.

Important data settings:

```yaml
preprocessor:
  charset: "@alphanumeric"

data:
  dataset:
    per_seq_size: 10000
    charset: "@alphanumeric"
    min_seq_len: 1
    max_seq_len: 16
    seq_len: null
    seed: 42
  test_ratio: 0.1
  val_ratio: 0.1
```

Set `seq_len` for a fixed-length experiment; otherwise use `min_seq_len` and `max_seq_len`.

## Metrics

Training reports three teacher-forced metrics:

- `loss`: cross-entropy over target tokens.
- `token_accuracy`: accuracy across individual target tokens.
- `exact_accuracy`: fraction of samples where every target token is correct.

After training, the best checkpoint is loaded and evaluated on the test split using normal autoregressive generation:

- `ar_exact_match_acc`: fraction of generated strings exactly matching their targets.
- `ar_token_acc`: character accuracy, counting missing and extra characters as errors.

Autoregressive metrics are the primary end-to-end result. Teacher-forced metrics are useful for diagnosing optimization and token-level learning.

## Model and training

The model is a decoder-only Transformer with causal attention, rotary positional embeddings, and a character-level tokenizer. The main capacity controls are:

```yaml
model:
  hidden_size: 64
  num_layers: 2
  num_heads: 2
  feedforward_dim: 128
  dropout: 0.1
  max_seq_len: 128
```

`hidden_size` must be divisible by `num_heads`, and the resulting head dimension must be even.

Validation loss drives `ReduceLROnPlateau`. By default, training stops when validation exact accuracy reaches 1.0 or the learning rate reaches its configured floor:

```yaml
scheduler:
  metric_name: loss
  mode: min
  patience: 3
  factor: 0.5
  min_lr: 0.000001

trainer:
  stop_on_perfect_acc: true
  stop_on_lr_floor: true
  early_stopping_patience: null
```

## Run the example

From this directory:

```bash
uv sync
uv run python main.py config.yaml
```

Training outputs are written below `outputs/<run_name>/<timestamp>/`. Each run contains its resolved configuration, console log, TensorBoard data, and model checkpoints.

View TensorBoard metrics with:

```bash
uv run tensorboard --logdir outputs
```

## Run sweeps

The sweep varies fixed sequence lengths across three seeds. Available modes are:

- `depth`: 6 or 8 layers with hidden size 32 and 2 heads.
- `width`: hidden/head pairs `(16, 1)`, `(32, 2)`, and `(64, 4)` with 4 layers.
- `full`: the depth and width cross-product.

Preview runs without training:

```bash
uv run python sweep.py depth --dry-run
```

Run locally with TensorBoard:

```bash
uv run python sweep.py depth --logger tensorboard
```

Run with ClearML and a two-hour limit per configuration:

```bash
clearml-init
uv run python sweep.py depth --logger clearml --time-budget 7200
```

Replace `depth` with `width` or `full` as needed. Without `--time-budget`, a run has no wall-clock limit.

## Files

- `config.yaml` — experiment configuration.
- `main.py` — single-run entry point.
- `sweep.py` — depth and width sweeps.
- `trainer.py` — training, metrics, generation, and checkpoints.
- `models/transformer.py` — decoder-only Transformer.
- `datasets/string_reverse.py` — generated reversal dataset.
- `preprocessors/char_tokenizer.py` — configurable character tokenizer.
