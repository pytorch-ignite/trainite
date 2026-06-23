# Trainite Project: string_reverse

Welcome to your new Trainite-generated training project!

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `dataset/`: Handles data loading and preprocessing.
- `trainer.py`: Defines the training and evaluation logic. You can override `train_step` or `eval_step` here.
- `config.py`: Contains Pydantic models for configuration validation. If you add new parameters to `config.yaml`, update the models here.
- `utils.py`: Shared utilities for configuration and logging.

## Getting Started

1. **Install Dependencies**:
   Ensure you have PyTorch, PyTorch-Ignite, and Pydantic installed.
   ```bash
   uv sync
   ```

2. **Run Training**:
   ```bash
   uv run python main.py config.yaml
   ```

3. **Monitor Progress**:
   Training logs and checkpoints are saved to the directory specified in `config.yaml` (default: `outputs/`). You can use TensorBoard to visualize metrics:
   ```bash
   uv run tensorboard --logdir outputs
   ```

## Components

### Model: transformer
# Transformer model

This is the model that learns the string-reverse task.

It is a decoder-only Transformer: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**: `input_ids` with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

The model is small and standard:

1. **Embedding**
   Converts token IDs into vectors.
2. **Positional encoding**
   Adds token order information.
3. **Transformer blocks**
   Repeated attention + feedforward layers.
4. **Final projection**
   Maps hidden states back to vocabulary logits.

## Practical notes

### Padding
Padding ID is `0`. The model ignores padded positions in attention.

### Sequence length
`max_seq_len` should be at least as long as your longest input sequence.

### Hidden size and heads
`hidden_size` must be divisible by `num_heads`.

### Positional encoding
`hidden_size` must be even because the positional encoding uses sine and cosine pairs.

## Config knobs

### `hidden_size`
Size of the token embeddings and hidden states.

Bigger values usually give more capacity, but cost more memory and time.

### `num_layers`
How many Transformer blocks to stack.

More layers = more capacity, slower training.

### `num_heads`
How many attention heads to use.

Choose a value that divides `hidden_size` cleanly.

### `feedforward_dim`
Size of the feedforward layer inside each block.

A common choice is `2x` to `4x` of `hidden_size`.

### `dropout`
Dropout rate used in attention, feedforward layers, and positional encoding.

### `max_seq_len`
Maximum sequence length supported by positional encoding.

## Minimal config example

```yaml
model:
  _target_: trainite.models.transformer.build_transformer_model
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  max_seq_len: 64
```

## When to change this file

Edit `trainite/models/transformer.py` if you want to:

- make the model wider or deeper
- swap attention or feedforward behavior
- change how padding is handled
- add caching, rotary embeddings, or other sequence features

## Good starting rule

If the dataset is tiny, start with a small model:

- `hidden_size=64`
- `num_layers=2`
- `num_heads=2`
- `feedforward_dim=128`

If training is stable and the model underfits, scale up from there.


### Dataset: string_reverse
# String Reverse dataset

This dataset gives you a very small task: feed in a random character sequence, predict the reversed sequence.

It is useful because it is easy to understand, fast to generate, and good for checking whether the model and training loop are wired correctly.

## What each sample looks like

Each example is built as a single prompt + answer sequence:

```text
<bos> abc <eos> cba <eos>
```

Training uses teacher forcing:

- the input contains both the prompt and the answer
- the labels mask out the prompt part with `-100`
- loss is only computed on the reversed answer

So the model is not asked to copy the input. It is asked to read the input, then produce the reversed text.

## What the dataset returns

Each item is a dictionary:

```python
{
    "input_ids": Tensor,
    "labels": Tensor,
}
```

Use `collate_fn` to pad a batch:

- inputs are padded with `0`
- labels are padded with `-100`

## Tokenizer

The dataset uses a simple character tokenizer with these special tokens:

- `0` = `<pad>`
- `1` = `<bos>`
- `2` = `<eos>`
- `3` = `<unk>`

It supports printable ASCII characters by default.

## Config knobs

### `per_seq_size`
How many unique examples to generate per sequence length bucket.

### `charset`
Which characters the random strings can use.

Built-in presets:

- `@universal` — all printable ASCII + space
- `@alpha` — letters only
- `@digits` — numbers only
- `@alphanumeric` — letters and numbers

You can also pass a custom string, for example `"abc123"`.

### `seq_len`
Use one fixed sequence length for every sample.

### `min_seq_len` / `max_seq_len`
Use a random length in this range.

Do not set `seq_len` together with `min_seq_len` or `max_seq_len`.

### `seed`
Controls dataset generation so runs are repeatable.

## Minimal config example

```yaml
data:
  train:
    dataset:
      _target_: trainite.datasets.string_reverse.build_string_reverse_dataset
      per_seq_size: 512
      charset: "@alpha"
      min_seq_len: 2
      max_seq_len: 12
```

## When to change this file

Edit `trainite/datasets/string_reverse.py` if you want to:

- change the tokenization rules
- add or remove allowed characters
- make the task longer or harder
- change how padding or labels are built

## Good starting rule

If you only want to test the pipeline, keep the defaults and just change:

- `per_seq_size`
- `charset`
- `min_seq_len` / `max_seq_len`


### Trainer: decoder-trainer
# DecoderTrainer

`DecoderTrainer` is the default training loop for this project.

It is built on PyTorch-Ignite and already handles the things you usually want on day one:

- training and evaluation loops
- optimizer setup
- learning-rate warmup + decay
- checkpoints
- early stopping
- TensorBoard logging

## What it expects

`DecoderTrainer` works with a model that returns logits shaped like:

```python
(batch, seq_len, vocab_size)
```

and a batch that looks like:

```python
{
    "input_ids": Tensor,
    "labels": Tensor,
}
```

It uses cross-entropy loss and ignores label value `-100`.

## What happens during training

For each batch:

1. move batch to device
2. run forward pass
3. compute cross-entropy loss
4. backpropagate
5. clip gradients if `grad_clip_norm` is set
6. step optimizer

At the end of each epoch it also runs evaluation on:

- training data
- validation data, if present
- test data, if present

## Logging and checkpoints

`DecoderTrainer` saves a run directory like this:

```text
output/<run_name>/<timestamp>/
```

Inside it you get:

- `config.yaml`
- `output.log`
- `last_*` checkpoint
- `best_*` checkpoint if validation exists
- TensorBoard logs

## Config knobs

### `epochs`
How many full passes over the training data to run.

### `log_every_steps`
How often to print training updates.

### `grad_clip_norm`
If set, gradients are clipped before the optimizer step.

This can help when training becomes unstable.

## What to tweak first

If you are just getting started, the usual first changes are:

- `epochs`
- `log_every_steps`
- `grad_clip_norm`
- optimizer settings in `config.yaml`

## If you want to customize behavior

Edit `trainite/trainers/decoder_trainer.py` if you want to:

- change how loss is computed
- add gradient accumulation
- add new metrics
- change checkpoint rules
- change validation frequency
- change logging behavior

The main places to look are:

- `_train_step`
- `_eval_step`
- `_attach_metrics`
- `_attach_handlers`

## Minimal config example

```yaml
trainer:
  epochs: 10
  log_every_steps: 20
  grad_clip_norm: 1.0
```

## Good starting rule

If the run fails or looks unstable, check these first:

- input and label shapes match
- `vocab_size` is large enough for the dataset
- `max_seq_len` is long enough
- learning rate is not too high


## Customization

### Dataset Configuration

You can configure your data in two ways in `config.yaml`:

#### Option 1: Explicit Splits
Define exactly what goes into each split. Use this for established datasets with fixed splits.
```yaml
data:
  train:
    dataset:
      _target_: ...
    dataloader:
      batch_size: 32
      shuffle: true
  val:
    dataset:
      _target_: ...
    dataloader:
      batch_size: 32
```

#### Option 2: Automatic Splitting
Define a single dataset and split ratios. Trainite will split it randomly (with a fixed seed).
```yaml
data:
  dataset:
    _target_: ...
  train_ratio: 0.8
  val_ratio: 0.2
  dataloader:
    batch_size: 32
```
*Note: In Option 2, the training split defaults to `shuffle: true`, while others default to `shuffle: false`.*

### Adding Configuration Parameters
1. Add the parameter to `config.yaml`.
2. Update the corresponding Pydantic model in `config.py` to include the new field.

## Design Philosophy
Trainite follows a "Cookiecutter, not framework" approach. The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
