# Trainite Project: string_reversal

Welcome to your new Trainite-generated training project!

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `datasets/`: Handles data loading and preprocessing.
- `trainer.py`: Defines the training and evaluation logic. You can override `_train_step` or `_eval_step` here.
- `config.py`: Contains base Pydantic models for configuration structure validation.
- `utils.py`: Shared utilities for configuration and logging.

## Experiment-specific trainer changes

This example started from Trainite's generated decoder-training project, but its trainer was customized for controlled model depth and width experiments. The model and data pipeline still follow the generated project structure; the changes below affect evaluation, scheduling, stopping, and experiment reporting.

### Sequence-level exact accuracy

The generated trainer reports loss and token accuracy. This example additionally reports `exact_accuracy`: a prediction is correct only when every non-ignored target token in the sequence is correct.

Token accuracy can look high while a model still gets most complete reversals wrong, especially for longer strings. Exact accuracy is therefore the more meaningful teacher-forced metric for this task. It is reported for the training, validation, and test splits.

### Autoregressive test evaluation

Teacher-forced metrics evaluate all target positions using the correct preceding tokens. That does not measure how errors compound when the model generates a reversal itself.

After loading the best checkpoint, the test step therefore also generates every test prediction autoregressively and reports:

- `ar_exact_match_acc`: fraction of decoded predictions exactly equal to their target strings.
- `ar_token_acc`: character accuracy over decoded strings. Missing and extra characters count as errors.
- A small set of prompt, target, and prediction examples.

These are the primary end-to-end task metrics. Teacher-forced metrics remain useful for diagnosing optimization and token-level learning.

### Validation-driven learning-rate reduction

The generated trainer's fixed warmup and linear-decay schedule was replaced with `ReduceLROnPlateau`. At the end of each validation pass, the scheduler monitors the configured metric—validation loss by default—and reduces the learning rate when that metric stops improving.

This is better suited to depth and width sweeps because different model sizes may converge at different rates. A fixed schedule tied to the requested run length can favor one model size or prematurely decay another.
Also the experimentations are time based instead of epoch based, so the standard warmup and decay schedule is not suitable for this project. The `ReduceLROnPlateau` scheduler is more flexible and can adapt to the model's performance during training.

The scheduler is configured under `scheduler` in `config.yaml`:

```yaml
scheduler:
  metric_name: loss
  mode: min
  patience: 3
  factor: 0.5
  min_lr: 0.000001
```

### Experiment stopping rules

Two optional convergence rules were added:

- `stop_on_perfect_acc`: stop when validation exact accuracy reaches `1.0`.
- `stop_on_lr_floor`: stop after the scheduler reaches `scheduler.min_lr`.

Both are enabled in the default configuration. Ordinary `early_stopping_patience` is disabled by default so it does not terminate training before `ReduceLROnPlateau` has a chance to lower the learning rate. It can still be enabled when a strict no-improvement limit is desired.

These rules let sweep runs use a generous epoch limit while avoiding unnecessary work after convergence or exhaustion of the configured learning-rate schedule.

### Comparable training evaluation

At each epoch, training metrics are evaluated on the same number of batches as the validation loader rather than on the entire, usually larger, training split. This reduces evaluation overhead and keeps the amount of data used for train-versus-validation monitoring comparable.

### Experiment logging

Iteration loss uses the same `trainer.log_every_steps` interval for TensorBoard and ClearML. Evaluation metrics are logged once per epoch, and autoregressive metrics are logged after testing. This keeps backend behavior comparable while avoiding needless per-iteration ClearML traffic during sweeps.

For ClearML, `output.project` is the shared project name and `output.run_name` identifies an individual task. This groups depth and width runs under one experiment project instead of creating a separate ClearML project for every run.

## Getting Started

You can set up and run this project using either **uv** (recommended for speed) or standard **pip**.

### Option A: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

1. **Install uv** (if you don't have it):
   - **macOS/Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows**:
     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - Or via pip: `pip install uv`

2. **Install Dependencies & Set Up Virtual Environment**:

   ```bash
   uv sync
   ```

3. **Run Training**:

   ```bash
   uv run python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   uv run tensorboard --logdir outputs
   ```

### Option B: Using standard pip & venv

If you prefer standard Python tools, you can create a virtual environment and use `pip`:

1. **Create and Activate a Virtual Environment**:
   - **macOS/Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install Dependencies**:

   ```bash
   pip install -e .
   ```

3. **Run Training**:

   ```bash
   python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   tensorboard --logdir outputs
   ```

## Running Sweeps

Sweep modes are `depth`, `width`, and `full`. For the final results, each run used a two-hour (7,200-second) wall-clock budget.

```bash
# Preview the runs without training
uv run python sweep.py depth --dry-run

# Run locally with TensorBoard
uv run python sweep.py depth --logger tensorboard

# Run the final ClearML sweep with a two-hour budget per run
uv run python sweep.py depth --logger clearml --time-budget 7200
```

Change `depth` to `width` or `full` to run a different sweep. Without `--time-budget`, runs have no wall-clock limit.

````

## Experiment Logging and Checkpoints

The default `logger: tensorboard` stores metrics and checkpoints locally. To use ClearML instead:

1. Run `clearml-init` to configure your ClearML server and credentials.
2. Set `logger: clearml` in `config.yaml`.

ClearML runs save checkpoints locally during training, avoiding remote uploads when `best.pt` or `last.pt` changes.
After a run and its test evaluation complete successfully, the final checkpoint files are uploaded once as ClearML
artifacts. Failed or interrupted runs retain their checkpoints locally without uploading them.

## Components

### Model: transformer

# Transformer model

It is a decoder-only Transformer: given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
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

`max_seq_len` represents the precomputed cache size for RoPE. Inputs longer than this will fall back to slower on-the-fly calculations.

### Hidden size and heads

`hidden_size` must be divisible by `num_heads`.

### Rotary positional encoding

`head_dim` (which is `hidden_size // num_heads`) must be even because RoPE rotates pairs of dimensions.

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

Dropout rate used in attention and feedforward layers.

### `max_seq_len`

Precomputed cache size for the rotary position embeddings.

## Minimal config example

```yaml
model:
  _target_: models.transformer.TransformerModel
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  max_seq_len: 64
````

## When to change this file

Edit `models/transformer.py` if you want to:

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

Each dataset item is a dictionary of raw strings:

```python
{
    "source": "abc",
    "target": "cba",
}
```

## Tokenizer

This dataset yields raw text strings (`source` and `target`). These strings are processed by the project's preprocessor before being passed to the model.

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
      _target_: datasets.string_reverse.StringReverseDataset
      per_seq_size: 512
      charset: "@alpha"
      min_seq_len: 2
      max_seq_len: 12
```

## When to change this file

Edit `datasets/string_reverse.py` if you want to:

- change the tokenization rules
- add or remove allowed characters
- make the task longer or harder
- change how padding or labels are built

## Good starting rule

If you only want to test the pipeline, keep the defaults and just change:

- `per_seq_size`
- `charset`
- `min_seq_len` / `max_seq_len`

### Trainer: decoder_trainer

# Trainer

`Trainer` (registered as `decoder-trainer`) is the default training loop for this project.

It is built on PyTorch-Ignite and already handles the things you usually want on day one:

- training and evaluation loops
- optimizer setup
- learning-rate warmup + decay
- checkpoints
- early stopping
- TensorBoard logging

## What it expects

`Trainer` works with a model that returns logits shaped like:

```python
(batch, seq_len, vocab_size)
```

and a batch that looks like:

```python
{
    "input_ids": Tensor,
    "labels": Tensor,
    "attention_mask": Tensor,  # Optional
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

At the end of the training run, it runs a final evaluation on the test data, if present.

## Logging and checkpoints

`Trainer` saves a run directory like this:

```text
outputs/<run_name>/<timestamp>/
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

### `early_stopping_patience`

Number of epochs to wait for validation loss improvement before stopping training early. Set to `null` to disable early stopping.

### `inference_every_epochs`

Run qualitative generation/inference tests on the model every N epochs. Set to `null` to disable.

### `inference_num_samples`

Number of random prompt samples to generate and log during the evaluation inference phase.

### `max_inference_new_tokens`

Maximum number of new tokens to generate per sample.

## What to tweak first

If you are just getting started, the usual first changes are:

- `epochs`
- `log_every_steps`
- `grad_clip_norm`
- `early_stopping_patience`
- optimizer settings in `config.yaml`

## If you want to customize behavior

Edit `trainer.py` if you want to:

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

### Preprocessor: char_tokenizer

# Char Tokenizer

`CharTokenizer` is a simple character-level tokenizer with a hardcoded universal vocabulary.

## Features

- Character-to-ID mapping of printable ASCII characters plus space.
- Special tokens:
  - `<PAD>` (ID 0)
  - `<BOS>` (ID 1)
  - `<SEP>` (ID 2)
  - `<EOS>` (ID 3)
  - `<UNK>` (ID 4)
- Supports padding, truncation, and PyTorch tensor generation.

## Config knobs

### `_target_`

Must be set to `preprocessors.char_tokenizer.CharTokenizer` in the template config.

## Minimal config example

```yaml
preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer
```

## Understanding `utils.py`

`utils.py` contains the bootstrapping and setup logic that links your configurations to executable code. It primarily acts as the file where all the helper functions are located that are being used in `trainer.py`. It handles parsing of dynamic `_target_` paths from `config.yaml`, builds your models and datasets, manages dataset splitting, and attaches standard PyTorch-Ignite event handlers (such as checkpointing, early stopping, learning rate scheduling, and logging).

Since the generated code in `utils.py` belongs to you, you can modify it directly to customize checkpoint rules, adjust early stopping metrics, or add new loggers.

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
  test_ratio: 0.1
  val_ratio: 0.1
  dataloader:
    batch_size: 32
```

_Note: In Option 2, the training split defaults to `shuffle: true`, while others default to `shuffle: false`._

### Adding Configuration Parameters

1. Add the parameter to `config.yaml`.
2. (Optional) Update the corresponding Pydantic model in `config.py` if you want strict validation and IDE autocomplete for custom fields.

## Design Philosophy

The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
