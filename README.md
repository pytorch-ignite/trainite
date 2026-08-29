# Trainite

**Trainite** gives you a complete, working training project as your starting point. The goal of generating a training project is to provide a clean, standalone codebase for **learning, rapid prototyping, and experimenting** without being constrained by framework abstractions. *(Note: Code generation is deterministic and template-based, not AI-based).*

Built on [PyTorch-Ignite](https://github.com/pytorch/ignite), the generated code is yours — read it, modify it, extend it however your research needs.

### Concretely, Trainite generates:

```
my-experiment/
├── config.yaml     # YAML configuration (edit hyperparameters here)
├── config.py       # Pydantic validation models (IDE autocomplete)
├── models/         # Model architecture templates
├── dataset_impl/   # Dataset and transform implementations
├── preprocessors/  # Tokenizer and preprocessing templates
├── trainer.py      # Training loop built on PyTorch-Ignite
├── utils.py        # Config loading and instantiation helpers
├── main.py         # Project entrypoint (runs training)
├── pyproject.toml  # Isolated package dependencies
└── README.md       # Documentation tailored to your selected components
```

### Dynamic Configuration

Swap components (like models, optimizers, or datasets) dynamically via config using dotted import paths (`_target_`):

```yaml
model:
  _target_: models.rope_transformer.RoPETransformerModel
  hidden_size: 64

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

**Trainite is explicitly not:**

- A replacement for HuggingFace Trainer.
- A tool for training large models on massive corpora.
- A framework that forces you into its runtime abstractions.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Initialize a Project](#initialize-a-project)
  - [Run Training](#run-training)
  - [Monitor & Iterate](#monitor--iterate)
  - [Cloud Training with SkyPilot](#cloud-training-with-skypilot)
- [Built-in Components](#built-in-components)
- [Configuration](#configuration)
- [Customizing Your Project](#customizing-your-project)
- [Examples](#examples)

---

## Installation

Requires Python >= 3.10.

From [pip](https://pypi.org/project/trainite/):

```bash
pip install trainite
```

From source (using `uv`):

```bash
git clone https://github.com/pytorch-ignite/trainite.git
cd trainite
uv sync
source .venv/bin/activate
```

From source (using `pip`):

```bash
git clone https://github.com/pytorch-ignite/trainite.git
cd trainite
pip install -e .
```

---

## Quickstart

Once Trainite is installed, generate and run your first experiment project:

```bash
# 1. Initialize a starter project
trainite init my-experiment --model rope-transformer --dataset string-reverse

# 2. Navigate into the project and install dependencies
cd my-experiment
pip install -e .

# 3. Run training
python main.py config.yaml
```

---

## Usage

> [!NOTE]
> If you are using the `uv` workflow, prefix commands with `uv run` (e.g., `uv run trainite init` or `uv run python main.py config.yaml`). If you are using standard `pip`/`venv`, run them directly (e.g., `trainite init` or `python main.py config.yaml`).

### Initialize a Project

You can generate a starter project by passing configuration options as command-line flags:

```bash
trainite init my-experiment --model rope-transformer --dataset string-reverse
```

Or run it interactively (Trainite will walk you through the options step-by-step):

```bash
trainite init
```

Interactive prompt preview:

*(Note: Trainite supports multi-model selection; you can select multiple models to include in your starter project.)*

```text
? Project directory: my-experiment
? Model(s): rope-transformer
? Dataset: string-reverse
? Trainer: decoder-trainer
? Output directory: outputs
? Run name: rope-transformer__string-reverse
? Enable SkyPilot cloud training (generate sky.yaml)? No

 Generated config.yaml
 Generated models/rope_transformer.py
 Generated dataset_impl/string_reverse.py
 Generated dataset_impl/transformed.py
 Generated trainer.py
 Generated utils.py
 Generated main.py
 Generated config.py
 Generated preprocessors/char_tokenizer.py
 Generated README.md
 Generated pyproject.toml
```

### Run Training

Your generated project is a standalone application with no runtime dependency on Trainite. Navigate to it and run:

```bash
cd my-experiment
python main.py config.yaml
```

### Monitor & Iterate

- **TensorBoard**: `tensorboard --logdir outputs`
- **ClearML**: Run `clearml-init`, then set `logger: clearml` in `config.yaml`.
- **Edit architecture**: Open `models/transformer.py` and modify the model.
- **Edit hyperparameters**: Open `config.yaml` and change learning rates, batch sizes, data splits, etc.

With ClearML enabled, Trainite keeps checkpoints in the run directory and uploads them to ClearML's default file
server. Generated projects document how to use another storage URI, defer to ClearML configuration, or keep
checkpoints local only.

Training outputs are organized by run:

```
outputs/
└── rope-transformer__string-reverse/
    └── 20260611_1430/
        ├── output.log          # Training logs
        ├── config.yaml         # Exact config used for this run
        ├── best_checkpoint_*.pt
        ├── last_checkpoint_*.pt
        └── tensorboard/        # TensorBoard event files
```

### Cloud Training with SkyPilot

You can enable [SkyPilot](https://docs.skypilot.ai/) cloud training in two ways:

1. **During project creation**:
   ```bash
   trainite init my-experiment --sky
   ```
2. **In an existing project**:
   ```bash
   cd my-experiment
   trainite add:sky
   ```

This creates `sky.yaml` in your project and adds `skypilot` to `pyproject.toml`. To launch on any cloud (AWS, GCP, Azure, Lambda, Kubernetes, etc.):

```bash
# 1. Update environment
pip install -e .    # or: uv sync

# 2. Verify cloud credentials
sky check

# 3. Launch training on cloud GPU
sky launch sky.yaml
```

> [!TIP]
> Trainite installs the base `skypilot` package. Running `sky check` will detect your active cloud credentials and provide the exact one-line command to install your specific cloud provider's driver if needed (e.g. `pip install "skypilot[aws]"` or `pip install "skypilot[kubernetes]"`).
>
> You can also run jobs on your own on-premises servers or custom clusters via SSH by following SkyPilot's [Existing Machines guide](https://docs.skypilot.co/en/latest/reservations/existing-machines.html).

## Built-in Components

### Models

| Name                | Architecture             | Description                                                                |
| ------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `basic-transformer` | Decoder-only Transformer | Standard causal LM with absolute positional embeddings.                    |
| `rope-transformer`  | Decoder-only Transformer | Standard causal LM with rotary positional embeddings (RoPE).               |

### Preprocessors

| Name   | Description                                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `char` | Character-level tokenizer mapping a charset to integer IDs, with reserved special tokens (`<PAD>`, `<BOS>`, `<SEP>`, `<EOS>`, `<UNK>`). |
| `gpt2` | Fast GPT-2 byte-level BPE tokenizer from Hugging Face Transformers; reuses the end-of-text token for padding. |

### Datasets

| Name             | Task                    | Description                                        |
| ---------------- | ----------------------- | -------------------------------------------------- |
| `string-reverse` | Reverse a random string | Synthetic, CPU-generatable. No downloads required. |
| `hugging-face`   | User-defined             | Loads a Hub dataset and leaves the task-specific transform for you. |
| `counting`       | Count the number of occurrences of a target token in a sequence.           | Synthetic, CPU-generatable. No downloads required. |

### Trainers

| Name              | Description                                                                                                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `decoder-trainer` | Standard supervised training (next-token prediction). Includes LR warmup + linear decay, checkpointing, early stopping, TensorBoard logging, and inference sample logging. |

---

## Configuration

All configuration lives in `config.yaml`. The top-level blocks map to base validation schemas in `config.py` (e.g. `ModelConfig`, `DatasetConfig`). This ensures that your configuration has the correct overall structure and valid keys, while allowing you to pass custom hyperparameters to your local components.

### Swapping Components with `_target_`

Components (such as models, datasets, optimizers, or collation functions) are specified via the `_target_` key. This key holds a dotted import path resolved at runtime, allowing you to swap components directly from config without altering training scripts:

```yaml
model:
  _target_: models.rope_transformer.RoPETransformerModel
  hidden_size: 64

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

For detailed configuration parameters and data splitting options (e.g., automatic splitting vs. explicit splits) specific to your selected components, refer to the generated `README.md` inside your initialized project.

---

## Customizing Your Project

Since the generated code is yours, you can override at any layer without touching the others:

| What to change     | How                                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------- |
| Model architecture | Edit `models/<model_name>.py`, or add a new file and update `_target_` in config         |
| Dataset            | Edit `dataset_impl/<dataset_name>.py`, or add a new file and update `_target_` in config |
| Train step         | Override `_train_step()` in `trainer.py`                                                 |
| Eval step          | Override `_eval_step()` in `trainer.py`                                                  |
| Metrics            | Add Ignite metrics in `_attach_metrics()` in `trainer.py`                                |
| Handlers           | Attach Ignite event handlers to `self.trainer` in `trainer.py`                           |
| Config fields      | Add fields to the Pydantic models in `config.py` and corresponding keys in `config.yaml` |

---

## Examples

Working examples are in the [`examples/`](examples/) directory:

| Example                                      | Description                                                                                                                                        |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`string_reversal`](examples/string_reversal/) | Train a decoder-only transformer to reverse strings. Demonstrates the full pipeline: data generation, training, evaluation, and inference logging. |
| [`counting`](examples/counting/)            | Train a decoder-only transformer to count the number of occurrences of a target token in a sequence. |

Each example is a standalone project — `cd` into it, install dependencies, and run `python main.py config.yaml`.

---

> For contributor and development docs, see [CONTRIBUTING.md](CONTRIBUTING.md).
