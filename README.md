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
| `builder_symbol` | Factory function (e.g., `build_transformer_model`) |
| `readme_template_path` | Per-component documentation template |
| `collate_fn_symbol` | (Datasets only) Collate function for DataLoader |

Registry enables CLI to dynamically discover and compose any combination of components.

### Dynamic Config Generation
When `trainite init` runs, it uses `inspect.getsource(cls)` to extract Pydantic model source code and inlines it into a local `config.py`. Result: fully typed, IDE-friendly configuration with zero runtime dependency on the `trainite` library.

### Template Adaptation
The CLI rewrites imports on the fly: `from trainite.config import ...` → `from config import ...`, and normalizes builder function references so `main.py` stays generic across architectures.

---

## Components

### String Reverse Dataset

Synthetic seq2seq task: given a random string, predict its reverse. Great for testing whether a model can learn structural transformation.

- **Tokenizer**: Character-level (`CharTokenizer`), covers all printable ASCII. `<pad>`, `<bos>`, `<eos>`, `<unk>` special tokens.
- **Format**: Prompt is `<bos> <sequence> <eos>`, target is `<sequence> <eos> <reversed> <eos>`. Prompt tokens masked from loss — model only learns on the reversed portion.
- **Collation**: Dynamic padding. Inputs pad with `0`, labels pad with `-100`.

| Config Key | Default | Notes |
|---|---|---|
| `size` | 256 | Number of samples |
| `charset` | `"@alpha"` | `"@alpha"`, `"@digits"`, `"@universal"`, or custom string |
| `min_seq_len` / `max_seq_len` | 1 / 16 | Variable-length mode (or use `seq_len` for fixed) |
| `seed` | 42 | Reproducibility |

### Transformer Model

Standard decoder-only Transformer. Sinusoidal positional encoding, multi-head causal self-attention via `F.scaled_dot_product_attention`, GELU feedforward.

- **Attention**: Single QKV projection, chunked into heads. Causal mask by default; combines with padding mask when present.
- **Scaling**: Embeddings multiplied by `√hidden_size` before positional encoding (per the original paper).
- **Vocab**: Auto-resolved from dataset if not provided in config. Must be ≥ dataset vocab size.

| Config Key | Default | Notes |
|---|---|---|
| `hidden_size` | 64 | Embedding dimension |
| `num_layers` | 2 | Transformer block count |
| `num_heads` | 2 | Must divide `hidden_size` |
| `feedforward_dim` | 128 | FFN hidden dimension |
| `dropout` | 0.1 | Applied in attn, FFN, pos encoding |
| `max_seq_len` | 128 | Positional encoding buffer size |

### PreTrainer

Batteries-included training loop on PyTorch-Ignite. Handles everything you don't want to write: engine wiring, checkpointing, early stopping, LR scheduling, and TensorBoard logging.

- **LR schedule**: 10% warmup (0 → peak), 90% linear decay (peak → 0).
- **Checkpointing**: Saves best model by validation accuracy + last model every epoch.
- **Early stopping**: Patience of 3 epochs on validation loss.
- **Metrics**: Masked cross-entropy loss, exact-match sequence accuracy.

| Config Key | Default | Notes |
|---|---|---|
| `epochs` | 3 | Training epochs |
| `log_every_steps` | 10 | Logging interval |
| `grad_clip_norm` | None | Max gradient norm (disabled if None) |

Output: `output/<run_name>/<timestamp>/` — config snapshot, log, best/last checkpoints, TensorBoard events.

---

## Config System (`trainite/config/`)

Configuration is YAML → Pydantic → validated. Every component has a dedicated Pydantic model.

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

`ComponentConfig` uses `_target_` key for late binding: the string `"module.path.builder_function"` is resolved via `importlib`, enabling config-driven instantiation without hardcoded imports.

```python
# trainite/utils.py
def instantiate(config: ComponentConfig, **kwargs) -> Any:
    target_path = config._target_    # "trainite.models.transformer.build_transformer_model"
    target_symbol = get_target(target_path)
    return target_symbol(**config_params, **kwargs)
```

`dump_config(config, path)` writes the full resolved config back to YAML for reproducibility.

---

## Templates (`trainite/templates/`)

Per-component documentation templates live in `trainite/templates/components/`. During `trainite init`, the CLI reads them and fills placeholders (`{{project_name}}`, `{{model_docs}}`, etc.) into the generated `README.md`.

```
trainite/templates/
├── components/
│   ├── datasets/string_reverse.md
│   ├── models/transformer.md
│   └── trainers/pretrainer.md
└── project/
    └── README.md          # Template for generated project README
```

The project README template (`README.md`) uses `{{model_docs}}`, `{{dataset_docs}}`, `{{trainer_docs}}` placeholders that get replaced with content from the component templates. This means generated projects get tailored documentation reflecting exactly which components were selected.

---

## Project Structure

### Library source
```
trainite/
├── cli/            # CLI: init command, template adaptation, import rewriting
├── config/         # Pydantic schemas, ComponentSpec registry, config I/O
├── datasets/       # Dataset implementations (string_reverse, ...)
├── models/         # Model implementations (transformer, ...)
├── trainers/       # Training loop (pretrainer)
├── templates/      # Per-component docs, project README template
├── main.py         # Library entrypoint (for direct prototyping)
└── utils.py        # get_target, instantiate helpers
```

### Generated project (`trainite init`)
```
<project>/
├── config.yaml     # YAML config — edit this
├── config.py       # Generated Pydantic models (typed, IDE-friendly)
├── models/         # Model architecture (your copy, editable)
├── dataset/        # Dataset + collate (your copy, editable)
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
| Model | Edit `model.py`, add new architecture class + builder |
| Dataset | Edit `dataset.py`, add new Dataset subclass + collate |
| Train step | Override `Pretrainer._train_step` or set `train_step` in config |
| Metrics | Add Ignite metrics in `_attach_metrics` |
| Handlers | Attach Ignite event handlers to `trainer.engine` |
| Config | Add fields to Pydantic model + YAML — explicit, no magic |

---

## Installation

```bash
git clone <repo-url>
cd trainite_prototype
uv sync
```

Dependencies: PyTorch, PyTorch-Ignite, Pydantic, PyYAML, OmegaConf, TensorBoard.
