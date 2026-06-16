# Contributing to Trainite

Thanks for your interest in contributing! This document covers the internal architecture, how to run tests, and the contribution workflow.

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

---

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

---

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
