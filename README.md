# Trainite

**Trainite** is a cookiecutter-style toolbox for training language models with PyTorch-Ignite. It is designed for researchers who want to go from zero to a running training loop in minutes without fighting a framework's abstractions.

---
## Design Overview
### 1. The Registry System (`ComponentSpec`)
Trainite uses a centralized registry (`trainite/config/registry.py`) to manage its built-in components. Each component (Model, Dataset, Trainer) is defined by a `ComponentSpec`:

- **`implementation_path`**: Path to the source file used as a template.
- **`config_cls`**: The Pydantic class defining the component's hyperparameters.
- **`builder_symbol`**: The name of the factory function (e.g., `build_model`) that instantiates the component.

This registry allows the CLI to dynamically discover and compose any combination of components during initialization.

### 2. Dynamic Config Generation
One of Trainite's core features is generating a **local, self-contained `config.py`** for each project. It achieves this using Python's `inspect` module:

- When `trainite init` runs, it looks up the `config_cls` for the selected model, dataset, and trainer.
- It uses **`inspect.getsource(cls)`** to extract the literal source code of these Pydantic models.
- It then concatenates these definitions into a single `config.py` file and wraps them in a master `ProjectConfig` class.

**Why?** This ensures the generated project has a fully typed, IDE-friendly configuration system without needing to import the `trainite` library at runtime.

### 3. Template Adaptation Logic
The CLI doesn't just copy files; it **refactors** them on the fly. In `trainite/cli/init.py`, the `_build_templates` function performs targeted string replacements:

- **Rewiring Imports**: Changes library-style imports (e.g., `from trainite.config import ...`) to local imports (`from config import ...`).
- **Symbol Normalization**: It renames specific builder functions to generic names like `build_model` or `build_dataloaders` so that `main.py` can remain generic across different architectures.

### 4. Configuration Composition
The generated `ProjectConfig` uses Pydantic's `Field(default_factory=...)` to compose the sub-configs:

```python
class ProjectConfig(BaseModel):
    model: TransformerModelConfig = Field(default_factory=TransformerModelConfig)
    dataset: StringReverseDatasetConfig = Field(default_factory=StringReverseDatasetConfig)
    # ...
```
This composition allows for a flat YAML structure that is both easy to read and strictly validated.

---

## Directory Structure

### Core Library (Prototype)
```text
trainite/
├── cli/            # CLI implementation & template adaptation logic
├── config/         # Pydantic schemas & the Component Registry
├── datasets/       # Implementation templates for datasets
├── models/         # Implementation templates for models
└── trainers/       # Base training logic (Ignite Engines)
```

### Generated Project Structure
```text
<project>/
├── config.yaml     # End-user hyperparameter file
├── config.py       # Generated Pydantic models (source-injected)
├── model.py        # Adapted model architecture
├── dataset.py      # Adapted dataset/dataloader logic
├── trainer.py      # Adapted Ignite training loop
└── main.py         # Static entrypoint
```

---

## Installation

```bash
git clone <repo-url>
cd trainite_prototype
uv sync
```

---

## Usage

### 1. Initialize a Project
```bash
uv run trainite init my-experiment --model transformer --dataset string-reverse
```

### 2. Run Training
```bash
cd my-experiment
uv run python main.py config.yaml
```

---

## Extending Trainite

Since everything is a local file, extending is easy:
- **Custom Model**: Modify `model.py`.
- **Custom Data**: Modify `dataset.py`.
- **Custom Metrics**: Attach new Ignite metrics in `trainer.py`'s `_attach_metrics`.
- **Custom Handlers**: Use Ignite's event system in `trainer.py`.

---

## Prototype specific files

- `main.py`: Sample entrypoint running the library version directly.
- `spec.md`: Design specification.
- `config.yaml`: Default configuration.
