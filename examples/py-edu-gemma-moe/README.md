# Trainite Project: py-edu-gemma-moe

> Generated with Trainite v0.2.0 using:
> ```bash
> trainite init py-edu-gemma-moe --model gemma4-moe gemma4-dense --dataset python-edu --trainer decoder-trainer --run-name gemma4_moe__py_edu_15m
> ```

Welcome to your new Trainite-generated training project!

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `dataset_impl/`: Contains the selected dataset and transform implementations.
- `trainer.py`: Defines the training and evaluation logic. You can override `_train_step` or `_eval_step` here.
- `config.py`: Contains base Pydantic models for configuration structure validation.
- `utils.py`: Shared utilities for configuration and logging.

## Getting Started

You can set up and run this project using either **uv** (recommended for speed) or standard **pip**.

### Option A: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

1. **Install uv** (if you don't have it):
   * **macOS/Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   * **Windows**:
     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * Or via pip: `pip install uv`

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
   * **macOS/Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * **Windows**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

   > [!NOTE]
   > `pip install -e .` only installs this project's dependencies — it does **not**
   > build the project into an installable Python package. This project is meant
   > to be run from source via `python main.py config.yaml`.
   >
   > To make it installable as a real package, restructure your code into a single
   > package — e.g., move `models/`, `dataset_impl/`, `preprocessors/` and the
   > top-level modules (`main.py`, `config.py`, `trainer.py`, `utils.py`) into a
   > `src/<your_package>/` directory with `__init__.py` files — then edit
   > `pyproject.toml` to replace the `[tool.setuptools]` block with:
   >
   > ```toml
   > [tool.setuptools.packages.find]
   > where = ["src"]
   > include = ["<your_package>*"]
   > ```
   >
   > and run `pip install -e .` again. See the
   > [setuptools package discovery docs](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)
   > for other layouts.

3. **Run Training**:
   ```bash
   python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   tensorboard --logdir outputs
   ```

### Option C: Cloud Training (SkyPilot)

To launch this experiment on cloud GPUs (AWS, GCP, Azure, Lambda, etc.) using [SkyPilot](https://docs.skypilot.ai/):

1. **Initialize SkyPilot configuration** (if not already generated):
   ```bash
   trainite add:sky
   ```
   This generates `sky.yaml` and adds `skypilot` to `pyproject.toml`.

2. **Update dependencies & verify cloud**:
   ```bash
   pip install -e .    # or: uv sync
   sky check
   ```

3. **Launch on Cloud**:
   ```bash
   sky launch sky.yaml
   ```

> [!TIP]
> Trainite installs the base `skypilot` package. Running `sky check` will detect your active cloud credentials and provide the exact one-line command to install your specific cloud provider's driver if needed (e.g. `pip install "skypilot[aws]"` or `pip install "skypilot[kubernetes]"`).
>
> You can also run jobs on your own on-premises servers or custom clusters via SSH by following SkyPilot's [Existing Machines guide](https://docs.skypilot.co/en/latest/reservations/existing-machines.html).

## Experiment Logging and Checkpoints

The default `logger: tensorboard` stores metrics and checkpoints locally. To use ClearML instead:

1. Run `clearml-init` to configure your ClearML server and credentials.
2. Set `logger: clearml` in `config.yaml`.

ClearML runs still keep checkpoints in the local output directory. By default, `ClearMLSaver` also uploads them to
ClearML's file server with `output_uri=True`.

To change checkpoint storage, edit the `ClearMLSaver` call in `trainer.py`:

- Use a storage URI such as `s3://bucket/path` to upload somewhere else.
- Use `None` to leave the destination to `CLEARML_DEFAULT_OUTPUT_URI` or
  `sdk.development.default_output_uri` in `clearml.conf`.

To disable checkpoint uploads while retaining ClearML metric logging, pass `output_uri=False` to `ClearMLLogger` in
`utils.py` and use `output_uri=None` for `ClearMLSaver` in `trainer.py`.

## Components

### Models: gemma4_moe, gemma4_dense
#### gemma4_moe
# Gemma 4 MoE model

A compact text-only, decoder-only Gemma 4 mixture-of-experts model for training from scratch.
It alternates sliding-window and global attention. Each block combines a dense
gated MLP with routed experts, while only the top experts run for each token.

## Helpful reference

This implementation was developed with
[rwightman/gemma4_pytorch_codex](https://github.com/rwightman/gemma4_pytorch_codex)
as an architecture reference.

## What goes in / out

- `input_ids`: `(batch, sequence)` token IDs
- `attention_mask`: optional `(batch, sequence)` padding mask
- output: `(batch, sequence, vocab_size)` logits

## Minimal config example

```yaml
model:
  _target_: models.gemma4_moe.Gemma4MoEModel
  hidden_size: 64
  num_layers: 12
  num_attention_heads: 4
  num_key_value_heads: 2
  dense_intermediate_size: 128
  expert_dim: 64
  num_experts: 4
  top_k: 2
  head_dim: 16
  layer_pattern: sssssg
  sliding_window: 64
  global_num_key_value_heads: 2
  global_head_dim: 16
  global_key_equals_value: true
  rope_theta: 10000.0
  global_rope_theta: 1000000.0
  rotary_fraction: 1.0
  global_rotary_fraction: 0.25
  attention_dropout: 0.0
loss:
  _target_: models.gemma4_moe.GemmaMoECausalLMLoss
  aux_loss_coef: 0.01
  ignore_index: -100
```

`layer_types` (one entry per layer) and `layer_pattern` are mutually exclusive.
`layer_pattern` is a compact string with `s` = sliding and `g` = global
(e.g. `sssssg` = 5x sliding + 1x global), repeated to fill `num_layers`
(`len(pattern)` must divide `num_layers`).
Keep `top_k <= num_experts`.

`attention_dropout` defaults to 0.0, matching Gemma. It applies only during training,
never during evaluation. Local/global RoPE defaults match the dense model.

Training adds the router balancing loss to cross-entropy. Evaluation reports
cross-entropy alone, so validation loss measures language modeling quality.


#### gemma4_dense
# Gemma 4 Dense model

It is a decoder-only causal Transformer model based on Google DeepMind's Gemma 4 architecture: given a sequence of token IDs, it predicts the next token at every position.

## Helpful reference

This implementation was developed with
[rwightman/gemma4_pytorch_codex](https://github.com/rwightman/gemma4_pytorch_codex)
as an architecture reference.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

1. **Embedding**
   Converts token IDs into vectors, scaled by $\sqrt{\text{hidden\_size}}$.
2. **Alternating Attention Blocks**
   Alternates between local sliding-window attention (100% RoPE with $\theta=10{,}000$) and global full attention (25% RoPE with $\theta=1{,}000{,}000$).
3. **Q/K Normalization**
   Applies RMSNorm per head before RoPE to stabilize training dynamics and prevent softmax saturation.
4. **Sandwich Normalization**
   Applies RMSNorm before and after both Attention and MLP sub-layers.
5. **Gated GeLU MLP**
   GeLU feed-forward network with `tanh` approximation and no biases (`DenseMLP`).
6. **Learnable Layer Scale**
   Scales each block's output with a learnable scalar parameter.
7. **Final RMSNorm & Projection**
   Normalizes the final representations and projects to vocabulary logits (with optional logit soft-capping and weight tying).

## Config knobs

### `hidden_size`
Total embedding and hidden dimension size.

### `num_layers`
Total number of transformer layers to stack.

### `num_attention_heads`
Number of query attention heads. Must be divisible by `num_key_value_heads`.

### `num_key_value_heads`
Number of key/value heads for Grouped-Query Attention (GQA).

### `dense_intermediate_size`
Hidden dimension of the gated GeLU feed-forward sub-layer.

### `head_dim`
Dimension of each attention head.

### `sliding_window`
Context window length for local sliding-window attention layers (default: 512).

### `layer_pattern`
Compact string with `s` (sliding) and `g` (global) alternating attention layers (e.g. `sg` or `sssssg`), repeated to fill `num_layers` (default: `"sg"`). Mutually exclusive with `layer_types`.

### `layer_types`
Optional explicit tuple of layer types (`sliding` or `global`), one entry per layer. Mutually exclusive with `layer_pattern`.

### `rope_theta`
Base frequency $\theta$ for local sliding-window layers (default: 10000.0).

### `global_rope_theta`
Base frequency $\theta$ for global full-attention layers (default: 1000000.0 for long-range scaling).

### `rotary_fraction`
Fraction of head dimensions rotated in local sliding layers (default: 1.0 for 100% rotation).

### `global_rotary_fraction`
Fraction of head dimensions rotated in global full layers (default: 0.25 for 25% partial rotation).

### `global_num_key_value_heads`
Number of key/value heads for global layers (default: 2).

### `global_head_dim`
Dimension of attention heads in global layers (default: 16).

### `global_key_equals_value`
Whether key projection is reused as value projection in global attention layers (default: true).

### `tie_word_embeddings`
Whether to tie the output projection weights to the input embedding table (default: true).

### `attention_dropout`
Attention dropout probability during training (default: 0.0, matching Gemma). Disabled during evaluation.

### `final_logit_softcap`
Optional $\tanh$-based soft-capping bound for logits to prevent overconfidence (default: null).

## Minimal config example

```yaml
model:
  _target_: models.gemma4_dense.Gemma4DenseModel
  hidden_size: 64
  num_layers: 2
  num_attention_heads: 4
  num_key_value_heads: 2
  dense_intermediate_size: 128
  head_dim: 16
  layer_pattern: sg
  sliding_window: 512
  rope_theta: 10000.0
  global_rope_theta: 1000000.0
  rotary_fraction: 1.0
  global_rotary_fraction: 0.25
  global_key_equals_value: true
  attention_dropout: 0.0
  tie_word_embeddings: true
  final_logit_softcap: null
```


### Dataset: python_edu
# Python-Edu dataset

This built-in dataset loads the [`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned) dataset from Hugging Face Datasets.

Python-Edu samples contain a `text` field with Python source code. The built-in transform uses this field as the causal language-modeling sequence.

## Configure the dataset

The dataset contains a `train` split. Trainite uses `DataWithAutoSplit` to automatically create the `train`, `val`, and `test` splits.

```yaml
data:
  dataset:
    path: Avelina/python-edu-cleaned
```

The default split ratios are:

* **train:** 80%
* **val:** 10%
* **test:** 10%

The ratios can be configured through `test_ratio` and `val_ratio`.

`path` is passed to Hugging Face `datasets.load_dataset`.

## Configure the transform

The transform can be configured with:

```yaml
data:
  transform:
    max_length: 128
```

`max_length` controls the maximum sequence length passed to the tokenizer.

The transform creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

## Hugging Face authentication

The dataset is loaded and cached by Hugging Face Datasets. Do not put Hugging Face access tokens directly in `config.yaml`.

If authentication is required, authenticate through the Hugging Face CLI or the appropriate environment configuration.


### Trainer: decoder_trainer
# Trainer

`Trainer` (registered as `decoder-trainer`) is the default training loop for this project.

It is built on PyTorch-Ignite and already handles the things you usually want on day one:

- training and evaluation loops
- optimizer setup
- mixed precision (AMP: float32, fp16, bf16) via Hugging Face Accelerate
- multi-GPU distributed data parallel (DDP) support via Hugging Face Accelerate
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

### `precision`
Mixed precision training mode: `"float32"` (default), `"fp16"`, or `"bf16"`.
Managed automatically via Hugging Face Accelerate.

- `"float32"`: Standard full precision.
- `"fp16"`: Half precision with dynamic loss scaling (recommended for NVIDIA GPUs with Tensor Cores).
- `"bf16"`: Bfloat16 mixed precision without loss scaling (recommended for Ampere+ GPUs or modern CPUs/TPUs).

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

## Distributed Training (Multi-GPU)

`Trainer` uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) to scale seamlessly across multiple GPUs without code changes.

Launch distributed training using `accelerate launch`:

```bash
# Launch directly specifying the number of processes:
accelerate launch --num_processes=4 main.py config.yaml

# Or configure your environment interactively once:
accelerate config
# And launch with saved settings:
accelerate launch main.py config.yaml
```

Accelerate automatically handles DDP wrapping, data partitioning (distributed samplers), and metric synchronization across all processes. Checkpoints and evaluation logs are automatically isolated to the main process (rank 0).

For further reading, see:
- [Launching Distributed Code](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [Accelerate Command Line Reference](https://huggingface.co/docs/accelerate/package_reference/cli)

## What to tweak first

If you are just getting started, the usual first changes are:

- `epochs`
- `precision`
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
  precision: bf16  # float32, fp16, or bf16
  log_every_steps: 20
  grad_clip_norm: 1.0
```

## Good starting rule

If the run fails or looks unstable, check these first:

- input and label shapes match
- `vocab_size` is large enough for the dataset
- `max_seq_len` is long enough
- learning rate is not too high


### Preprocessor: gpt2_tokenizer
# GPT-2 Tokenizer

This project uses the fast GPT-2 tokenizer from Hugging Face Transformers. Its byte-level BPE vocabulary handles
general text more efficiently than the character tokenizer.

GPT-2 has no dedicated padding token, so `load_gpt2_tokenizer` reuses its end-of-text token for padding. Trainite's
attention masks still distinguish padding from real end-of-text tokens.

## Minimal config

```yaml
preprocessor:
  _target_: preprocessors.gpt2_tokenizer.load_gpt2_tokenizer
```

Edit `preprocessors/gpt2_tokenizer.py` if you need another pretrained tokenizer. Keep in mind that a larger vocabulary
also makes Trainite's embedding and output layers larger.


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
*Note: In Option 2, the training split defaults to `shuffle: true`, while others default to `shuffle: false`.*

### Adding Configuration Parameters
1. Add the parameter to `config.yaml`.
2. (Optional) Update the corresponding Pydantic model in `config.py` if you want strict validation and IDE autocomplete for custom fields.

### Grid Search / Hyperparameter Sweeps
Trainite supports grid search through the optional sweep section in `config.yaml`. Specify the parameters to explore using the same nested structure as the main configuration, with a list of candidate values for each parameter.

For example, to try three learning rates:
```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0003

sweep:
  optimizer:
    lr: [0.0001, 0.0003, 0.001]
```

Trainite generates a training run for each combination of sweep parameter values. Parameters not specified in sweep retain their base configuration values.

Run the sweep using the usual command:
`uv run python main.py config.yaml`

## Design Philosophy
The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
