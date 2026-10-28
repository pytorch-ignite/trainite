# Trainite Project: py-edu-gemma-moe

> Generated with Trainite v0.1.0 using:
> ```bash
> trainite init py-edu-gemma-moe --model gemma4-moe --dataset python-edu --trainer decoder-trainer --run-name gemma4_moe__python_edu
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

## Metrics

Train/validation/test evaluators report `loss` (cross-entropy), `token_accuracy`,
and `perplexity = exp(loss)`, weighted over supervised body tokens and EOS.
Prompt and padding labels are ignored. Perplexity excludes the MoE auxiliary loss;
checkpoint selection and early stopping still use validation cross-entropy.

Training logs include smoothed `batch_loss` (total objective), `lm_loss`
(cross-entropy only), and unweighted `aux_loss` (router balancing). The objective
is `lm_loss + aux_loss_coef * aux_loss`. Balanced routing has auxiliary loss 1,
though this value alone does not establish useful expert specialization.

`router/layer_N/expert_M` logs the smoothed share of top-k routing assignments
received by each expert. Shares sum to 1 per layer; uniform usage is
`1 / num_experts`. Persistently near-zero usage suggests an unused expert.
These statistics cover all nonpadding input tokens, including prompts, and are
combined across ranks weighted by real-token counts. They describe training
routing, not validation routing. Smoothed values are not whole-epoch counts.

Every `benchmark_every_epochs` epochs the trainer also greedily completes fixed
validation prompts and logs `bench/compile_rate` (generated source parses) and
`bench/body_rate` (it also defines the prompted function with a body beyond the
docstring). Generated code is never executed during training.

After training, `test()` restores the best checkpoint, runs the held-out loader,
then generates and executes solutions for HumanEval+ mini (EvalPlus). It logs
`humaneval/base_pass_at_1` and `humaneval/plus_pass_at_1`, and leaves
`humaneval_samples.jsonl` plus EvalPlus' `*_eval_results.json` in the run
directory for inspection. The mini suite keeps all 164 HumanEval tasks but uses a
reduced set of extra tests, so it is much faster than full HumanEval+ while still
catching clear correctness failures. Generated code runs in subprocesses with
timeouts, but this is not a security sandbox: only run it with trusted models and
data. Disable it with `humaneval_enabled: false`; tune `humaneval_max_new_tokens`
and `humaneval_batch_size` in the trainer config. EvalPlus is installed as a
project dependency.

All metrics are written to TensorBoard or ClearML. Console epoch summaries include
perplexity, separate training loss components, expert usage, and benchmark rates.
Run the offline checks with `uv run python -m unittest test_metrics.py test_tokenizer.py test_humaneval.py`.

## Experiment Logging and Checkpoints

The default `logger: tensorboard` stores metrics and checkpoints locally. To use ClearML instead:

1. Run `clearml-init` to configure your ClearML server and credentials.
2. Set `logger: clearml` in `config.yaml`.

ClearML runs log metrics and keep checkpoints in the local output directory during training. After training, the best
checkpoint is uploaded once as a ClearML artifact with `wait_on_upload=True`. PyTorch auto-connect is disabled, so
checkpoints are not registered as `Imported by <run>` models.

Uploads are best-effort: if one fails, the error is logged and the local checkpoint is kept. Artifacts are stored at the
task's output URI, which comes from `CLEARML_DEFAULT_OUTPUT_URI` or `sdk.development.default_output_uri` in
`clearml.conf`. Set it to an `s3://` or `gs://` path if the worker's local disk will not persist.

## Components

### Models: gemma4_moe
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


### Dataset: function completion

Run training normally from this project directory:

```bash
uv run python main.py config.yaml
```

Check the actual dataset and configured tokenizer before training:

```bash
uv run python test_real_dataset.py
```

This explicitly loads `train[:1000]`, filters functions, retains up to 100 usable
examples, checks all selected examples and train/validation/test batches, and
prints a prompt/completion pair. It requires network access or a populated Hugging
Face cache; the initial dataset download can exceed the requested source slice.
To exercise the full source split and retain up to 1,000 usable functions:

```bash
uv run python test_real_dataset.py --source-split train --dataset-size 1000
```

Set `data.dataset.dataset_size` to a positive integer (for example, `10000`)
to keep the first N usable function examples after extraction and token-length
filtering, before Trainite splits them. The default `null` keeps all usable
functions; if fewer than N survive filtering, all available examples are kept.
Indexing stops immediately once N usable functions are collected, including in
the middle of a source file. Results are cached through Hugging Face
`Dataset.from_generator`. Initial source downloads can still cover the full dataset.

All dataset processing lives in `dataset_impl/python_edu.py`. Trainite instantiates
`load_function_dataset` through the dataset `_target_`; it loads the existing
Python-Edu raw `text` column through the original Hugging Face dataset loader,
then indexes individual raw functions using a cached generator. No separate
preparation script, JSONL export, or bucket configuration is required. Hugging
Face handles its normal download and processing cache.

Each generator row is a validated prompt/completion pair: the prompt contains the
file's top-level imports followed by the function signature and existing
docstring, and the completion is the remaining function body. Imports preserve
source text and order, including aliases and multiline imports; nested or
conditional imports are not lifted out of their original scope. The generator extends slices over trailing backslash continuations and drops
only examples that still do not parse, so extraction errors happen before
training. Functions that read a sibling function, class, or module-level
variable from the same file are skipped: the prompt carries imports but not
that context, and HumanEval prompts are self-contained. `PythonEduTransform`
only tokenizes the precomputed strings and masks the prompt labels. The loader
checks token lengths directly to exclude oversized functions before batching; it
does not instantiate or call the transform. Only top-level, undecorated functions
with nonempty docstrings and remaining bodies are included. Async functions and
multiline signatures are supported. Source is parsed, never executed. Missing
docstrings, invalid Python, inline suites, and examples exceeding 512 tokens
including EOS are skipped. Imports count toward the token limit. Dataset and
transform `max_length` values must match.

Trainite's existing automatic split logic divides the extracted function examples
into 80/10/10 train/validation/test sets using the configured seed. Ratios are
controlled by `data.val_ratio` and `data.test_ratio`; the dataset loader does not
split data. Functions from the same source file or duplicate functions can
therefore appear in different splits.
The transform masks prompt labels with -100, supervises body tokens and EOS, and
supplies only the prompt at inference.

Bodies may still depend on globals, helpers, or conditional imports absent from the prompt; these
are training examples, not standalone correctness benchmarks. The shared Trainite
raw-code pretraining component remains unchanged.

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


### Preprocessor: python_bpe
# Byte-level BPE Tokenizer

The default preprocessor trains a byte-level BPE tokenizer on the raw `text` column of the configured dataset, once,
and caches it under `outputs/tokenizers/<dataset>-<split>-<vocab>-<min_frequency>.json`. Later runs reuse that file, so token ids stay
stable. Training runs before functions are filtered by token length, so the corpus is not biased by the tokenizer it
produces.

An 8192-token code vocabulary keeps the output layer small, which matters for memory: logits scale linearly with
vocabulary size. Byte-level BPE round-trips indentation and arbitrary characters without unknown tokens. Padding uses
a separate `<pad>` token so the end-of-sequence embedding stays trainable and EOS can be predicted.

## Minimal config

```yaml
preprocessor:
  _target_: preprocessors.python_bpe.load_python_bpe_tokenizer
  vocab_size: 8192
  cache_dir: outputs/tokenizers
```

`dataset_path` and `split` are injected from `data.dataset`, so the tokenizer always trains on the corpus the run
actually uses. Delete the cache file to retrain after changing the corpus or merge settings. The pretrained GPT-2
tokenizer remains available as `preprocessors.gpt2_tokenizer.load_gpt2_tokenizer` (50257 tokens, no training).


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

## Design Philosophy
The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
