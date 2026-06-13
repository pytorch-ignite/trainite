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

### Dynamic Configuration

Swap components (like models, optimizers, or datasets) dynamically via config using dotted import paths (`_target_`):

```yaml
model:
  _target_: models.transformer.TransformerModel
  hidden_size: 64

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
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
  - [Testing](#testing)

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

### Swapping Components with `_target_`

Components (such as models, datasets, optimizers, or collation functions) are specified via the `_target_` key. This key holds a dotted import path resolved at runtime, allowing you to swap components directly from config without altering training scripts:

```yaml
model:
  _target_: models.transformer.TransformerModel
  hidden_size: 64

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

For detailed configuration parameters and data splitting options (e.g., automatic splitting vs. explicit splits) specific to your selected components, refer to the generated `README.md` inside your initialized project.

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
