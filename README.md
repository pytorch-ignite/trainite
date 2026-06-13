# Trainite

**Trainite** is a cookiecutter toolbox for training language models with PyTorch-Ignite. Generate a clean, working project — model, dataset, config, trainer, entrypoint — and get out of the way. Zero to running training loop in minutes.

---

## Quickstart

```bash
git clone <repo-url>
cd trainite_prototype
uv sync

# Initialize a project from built-in components
uv run trainite init my-experiment --model transformer --dataset string-reverse --trainer pretrainer

cd my-experiment
uv run python main.py config.yaml
```

---

## Design Overview

### Cookiecutter, Not Framework
`trainite init` generates real, readable Python files. You own them. No hidden imports, no sealed abstractions. Modify anything — model, dataset, train step, metrics — without touching the rest.

### Registry System
Centralized registry (`trainite/config/registry.py`) manages built-in components via `ComponentSpec`:

| Field | Purpose |
|---|---|
| `implementation_path` | Source file used as template |
| `config_cls` | Pydantic class for hyperparameters |
| `builder_symbol` | Class name or factory function (e.g., `TransformerModel`) |
| `readme_template_path` | Per-component documentation template |
| `collate_fn_symbol` | (Datasets only) Collate function for DataLoader |

Registry enables CLI to dynamically discover and compose any combination of components.

### Dynamic Config Generation
When `trainite init` runs, it uses `inspect.getsource(cls)` to extract Pydantic model source code and inlines it into a local `config.py`. Result: fully typed, IDE-friendly configuration with zero runtime dependency on the `trainite` library.

### Template Adaptation
The CLI rewrites imports on the fly: `from trainite.config import ...` → `from config import ...`, and normalizes builder function references so `main.py` stays generic across architectures.

### Dataset Configuration Styles

Trainite supports two ways to define your data:

#### Option 1: Explicit Splits
Manually define every split. Each split gets its own dataset instance and dataloader config.
```yaml
data:
  train:
    dataset: { _target_: ..., size: 1000 }
    dataloader: { batch_size: 32, shuffle: true }
  val:
    dataset: { _target_: ..., size: 200 }
    dataloader: { batch_size: 32 }
```

#### Option 2: Automatic Splitting
Define one dataset and let Trainite handle the math. Useful for quick prototyping.
```yaml
data:
  dataset: { _target_: ..., size: 1200 }
  test_ratio: 0.1
  val_ratio: 0.1
  dataloader: { batch_size: 32 } # Applied to all splits
```

**Hierarchy:**

```
ProjectConfig
├── model: ComponentConfig          (TransformerModelConfig)
├── optimizer: OptimizerConfig      (torch.optim.AdamW, lr, etc.)
├── data: DataConfig
│   ├── train: SplitConfig
│   │   ├── dataset: ComponentConfig   (StringReverseDatasetConfig)
│   │   └── dataloader: DataLoaderConfig
│   ├── val: SplitConfig | None
│   └── test: SplitConfig | None
├── trainer: TrainerConfig          (PreTrainerConfig)
├── output: OutputConfig            (root, run_name)
├── seed: int
└── device: str                     ("auto" | "cpu" | "cuda")
```

`ComponentConfig` uses `_target_` key for late binding: the string `"module.path.ClassName"` is resolved via `importlib`, enabling config-driven instantiation without hardcoded imports.

```python
# trainite/utils.py
def instantiate(config: ComponentConfig, **kwargs) -> Any:
    target_path = config._target_    # "trainite.models.transformer.TransformerModel"
    target_symbol = get_target(target_path)
    return target_symbol(**config_params, **kwargs)
```

`dump_config(config, path)` writes the full resolved config back to YAML for reproducibility.

---
### Generated project (`trainite init`)
```
<project>/
├── config.yaml     # YAML config — edit this
├── config.py       # Generated Pydantic models (typed, IDE-friendly)
├── models/         # Model architecture (your copy, editable)
├── datasets/       # Dataset + collate (your copy, editable)
├── trainer.py      # Training loop (your copy, editable)
├── utils.py        # Config helpers
├── main.py         # Entrypoint
├── pyproject.toml  # Dependencies
└── README.md       # Tailored to selected components
```

---

## Extending

| Layer | How |
|---|---|
| Model | Edit `models/{model_name}.py`, add new architecture class |
| Dataset | Edit `datasets/{dataset_name}.py`, add new Dataset subclass |
| Train step | Override `Pretrainer._train_step` or set `train_step` in config |
| Metrics | Add Ignite metrics in `_attach_metrics` |
| Handlers | Attach Ignite event handlers to `trainer.engine` |
| Config | Add fields to Pydantic model + YAML — explicit, no magic |

---

## Getting Started

### 1. Installation

Requires [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone <repo-url>
cd trainite_prototype
uv sync
```

### 2. Initialize a Project

Create a new, isolated training project using the CLI:

```bash
# General syntax: uv run trainite init <project-name> --model <model> --dataset <dataset>
uv run trainite init my-experiment --model transformer --dataset string-reverse
```

### 3. Run Training

Your generated project is a standalone application. Navigate to it and run:

```bash
cd my-experiment
uv run python main.py config.yaml
```

### 4. Monitor & Iterate

- **TensorBoard**: `uv run tensorboard --logdir outputs`
- **Edit Code**: Open `my-experiment/models/transformer.py` to change architecture.
- **Edit Config**: Open `my-experiment/config.yaml` to change learning rates or data splits.

---
