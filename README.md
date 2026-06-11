# Trainite

**Trainite** is a cookiecutter project generator for training language models using [PyTorch-Ignite](https://github.com/pytorch/ignite). 

Instead of wrapping your code in a runtime framework, Trainite generates a clean, standalone, zero-dependency project folder containing standard PyTorch code that you fully own.

### Concretely, Trainite generates:

```
my-experiment/
├── config.yaml     # YAML configuration (edit hyperparameters here)
├── config.py       # Pydantic validation models (IDE autocomplete)
├── models/         # Model architecture templates (fully editable)
├── datasets/       # Dataset and tokenization templates (fully editable)
├── trainer.py      # Training loop built on PyTorch-Ignite (fully editable)
├── utils.py        # Config loading and instantiation helpers
├── main.py         # Project entrypoint (runs training)
├── pyproject.toml  # Isolated package dependencies
└── README.md       # Documentation tailored to your selected components
```

### Dynamic configuration (`config.yaml`):

Swap components (like models, optimizers, or datasets) purely from the YAML configuration using dotted import paths (`_target_`):

```yaml
model:
  _target_: models.transformer.TransformerModel
  hidden_size: 64
  num_layers: 2
  num_heads: 2
  feedforward_dim: 128
  dropout: 0.1
  max_seq_len: 128

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001

data:
  dataset:
    _target_: datasets.string_reverse.StringReverseDataset
    per_seq_size: 256
    min_seq_len: 1
    max_seq_len: 16
  dataloader:
    batch_size: 32
```

**Trainite is explicitly not:**
- A replacement for HuggingFace Trainer or Accelerate.
- A tool for training large models on massive corpora.
- A framework that forces you into its runtime abstractions.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
  - [Initialize a Project](#initialize-a-project)
  - [Run Training](#run-training)
  - [Monitor & Iterate](#monitor--iterate)
- [Built-in Components](#built-in-components)
- [Configuration](#configuration)
- [Customizing Your Project](#customizing-your-project)
- [Examples](#examples)
- [Development](#development)
  - [Architecture Overview](#architecture-overview)
  - [Adding a Built-in Component](#adding-a-built-in-component)
  - [Testing](#testing)
  - [Contributing](#contributing)

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/pytorch-ignite/trainite.git
cd trainite

# Install the package in editable mode
pip install -e .

# Generate a project
trainite init my-experiment --model transformer --dataset string-reverse

# Train
cd my-experiment
python main.py config.yaml
```

---

## Installation

Requires Python >= 3.10. Clone the repository first:

```bash
git clone https://github.com/pytorch-ignite/trainite.git
cd trainite
```

Install using one of the following methods:

### Option 1: Using `uv` (Recommended)

If you use [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

This installs dependencies into a local `.venv` and manages execution. Run CLI commands prefixed with `uv run`:
```bash
uv run trainite init
```

### Option 2: Standard `pip`

Create and activate your virtual environment, then install in editable mode:

```bash
pip install -e .
```

This registers the `trainite` command globally in your environment. You can run CLI commands directly:
```bash
trainite init
```

---

## Usage

> [!NOTE]
> If you are using the `uv` workflow, prefix commands with `uv run` (e.g., `uv run trainite init` or `uv run python main.py config.yaml`). If you are using standard `pip`/`venv`, run them directly (e.g., `trainite init` or `python main.py config.yaml`).

### Initialize a Project

```bash
trainite init <project-name> --model <model> --dataset <dataset>
```

You can pass all options as flags:

```bash
trainite init my-experiment --model transformer --dataset string-reverse
```

Or run it interactively — Trainite will prompt you step by step:

```bash
trainite init
```

### Run Training

Your generated project is a standalone application with no runtime dependency on Trainite. Navigate to it and run:

```bash
cd my-experiment
python main.py config.yaml
```

### Monitor & Iterate

- **TensorBoard**: `tensorboard --logdir outputs`
- **Edit architecture**: Open `models/transformer.py` and modify the model.
- **Edit hyperparameters**: Open `config.yaml` and change learning rates, batch sizes, data splits, etc.

Training outputs are organized by run:

```
outputs/
└── transformer__string-reverse/
    └── 20260611_1430/
        ├── output.log          # Training logs
        ├── config.yaml         # Exact config used for this run
        ├── best_checkpoint_*.pt
        ├── last_checkpoint_*.pt
        └── tensorboard/        # TensorBoard event files
```



## Built-in Components

### Models

| Name | Architecture | Description |
|---|---|---|
| `transformer` | Decoder-only Transformer | Causal LM with rotary position embeddings (RoPE) and multi-head attention. |

### Datasets

| Name | Task | Description |
|---|---|---|
| `string-reverse` | Reverse a random string | Synthetic, CPU-generatable. No downloads required. Ships with a character-level tokenizer. |

### Trainers

| Name | Description |
|---|---|
| `pretrainer` | Standard supervised training (next-token prediction). Includes LR warmup + linear decay, checkpointing, early stopping, TensorBoard logging, and inference sample logging. |

---

## Configuration

All configuration lives in `config.yaml`. Every key maps to a validated Pydantic model in `config.py`, giving you type checking, clear error messages, and IDE autocomplete.

### Data Configuration

Trainite supports two ways to define your data:

**Option 1: Automatic splitting** — define one dataset and let Trainite split it:

```yaml
data:
  dataset:
    _target_: datasets.string_reverse.StringReverseDataset
    per_seq_size: 256
    charset: "@alpha"
    min_seq_len: 1
    max_seq_len: 16
  train_ratio: 0.8
  val_ratio: 0.1
  dataloader:
    batch_size: 32
    shuffle: true
```

**Option 2: Explicit splits** — define each split independently:

```yaml
data:
  train:
    dataset:
      _target_: datasets.string_reverse.StringReverseDataset
      per_seq_size: 1000
    dataloader:
      batch_size: 32
      shuffle: true
  val:
    dataset:
      _target_: datasets.string_reverse.StringReverseDataset
      per_seq_size: 200
    dataloader:
      batch_size: 32
```

### The `_target_` Key

Components are specified via `_target_`, a dotted import path that is resolved at runtime. This allows you to swap models, datasets, optimizers, or collate functions purely from config:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

---

## Customizing Your Project

Since the generated code is yours, you can override at any layer without touching the others:

| What to change | How |
|---|---|
| Model architecture | Edit `models/<model_name>.py`, or add a new file and update `_target_` in config |
| Dataset | Edit `datasets/<dataset_name>.py`, or add a new file and update `_target_` in config |
| Train step | Override `_train_step()` in `trainer.py` |
| Eval step | Override `_eval_step()` in `trainer.py` |
| Metrics | Add Ignite metrics in `_attach_metrics()` in `trainer.py` |
| Handlers | Attach Ignite event handlers to `self.engine` in `trainer.py` |
| Config fields | Add fields to the Pydantic models in `config.py` and corresponding keys in `config.yaml` |

---

## Examples

Working examples are in the [`examples/`](examples/) directory:

| Example | Description |
|---|---|
| [`string_reverse`](examples/string_reverse/) | Train a decoder-only transformer to reverse strings. Demonstrates the full pipeline: data generation, training, evaluation, and inference logging. |

Each example is a standalone project — `cd` into it, install dependencies, and run `python main.py config.yaml`.

---

# Development

Everything below is for contributors and developers working on Trainite itself.

---

## Architecture Overview

When a user runs `trainite init`, the following happens:

```
trainite init my-experiment --model transformer --dataset string-reverse
│
├─ 1. Registry lookup
│     Look up ModelSpec, DatasetSpec, TrainerSpec from the registry
│
├─ 2. Config assembly
│     Instantiate Pydantic config defaults for each component
│     Inject collate function target from ModelSpec into DataConfig
│     Build a complete ProjectConfig and dump it to config.yaml
│
├─ 3. Template copying
│     Copy implementation files (model, dataset, trainer, utils, main, config)
│     from the trainite package into the target project directory
│
├─ 4. Import rewriting
│     Replace internal imports (e.g. "trainite.config" → "config")
│     so the generated code is fully standalone
│
└─ 5. Dependency generation
      Generate pyproject.toml with only the deps needed for the
      selected components
```

The generated project has **zero runtime dependency** on the `trainite` package.


## Adding a Built-in Component

To add a new built-in model (the process is similar for datasets and trainers):

1. **Write the implementation** in `trainite/models/<name>.py`. The model class should accept all hyperparameters as `__init__` kwargs.

2. **Create a Pydantic config class** in `trainite/config/model.py`:

   ```python
   class MyModelConfig(ComponentConfig):
       target: str = Field(default="trainite.models.my_model.MyModel", alias="_target_")
       hidden_size: int = Field(default=64, gt=0)
       # ... other hyperparameters
   ```

3. **Register it** in `trainite/config/registry.py`:

   ```python
   MODEL_SPECS["my-model"] = ModelSpec(
       name="my_model",
       implementation_path=Path("trainite/models/my_model.py"),
       config_cls=MyModelConfig,
       implementation_symbol="MyModel",
       builder_symbol="MyModel",
       # ...
   )
   ```

4. **(Optional)** Add a readme template at `trainite/templates/components/models/my_model.md`.

5. **Add tests** in `tests/models/my_model_test.py`.

## Testing

Run the full test suite:

```bash
uv run pytest
```

Tests are organized by component:

```
tests/
├── cli_test.py                     # CLI and code generation
├── utils_test.py                   # Config loading and instantiation
├── config/
│   └── base_test.py                # Config validation
├── datasets/
│   └── string_reverse_test.py      # Dataset and tokenizer
├── models/
│   └── transformer_test.py         # Model forward pass and generation
└── trainers/
    └── pretrainer_test.py          # Training loop integration
```

## Contributing

### Setup

```bash
git clone https://github.com/pytorch-ignite/trainite.git
cd trainite
uv sync --dev
uv run pre-commit install
```

### Code Quality

Pre-commit hooks run automatically on every commit and enforce:

- **Ruff** — linting and formatting
- **Pyrefly** — type checking
- Trailing whitespace, YAML validity, TOML validity, no debug statements

To run checks manually:

```bash
uv run pre-commit run --all-files
```

### Pull Requests

- Branch from `main`
- Write tests for new functionality
- Ensure all CI checks pass (`uv run pytest` + pre-commit)
