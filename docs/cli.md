# CLI Guide

Trainite exposes a small command-line interface for scaffolding and extending standalone training projects.

For project structure and first-run setup, see [Home](index.md).  
For `config.yaml` details and training behavior, see [Training Guide](training.md).

## Command summary

```text
trainite [--help] [--version|-V]
trainite init [PROJECT_DIR] [OPTIONS]
trainite add:sky [--force]
```

## Global flags

- `--help`: Show available subcommands and command-specific usage.
- `--version` / `-V`: Print the Trainite version and repository URL.

Examples:

```bash
trainite --help
trainite --version
trainite -V
```

## `trainite init`

Generate a starter training project with local copies of model, dataset, trainer, and config code.

### Interactive mode

Run with no additional arguments:

```bash
trainite init
```

You will be prompted for:

1. Project directory
2. One or more model templates
3. The primary active model (when multiple models are selected)
4. Dataset template
5. Trainer template
6. Output root (`output.root` in `config.yaml`)
7. Run name (`output.run_name` in `config.yaml`)
8. Whether to generate `sky.yaml` for SkyPilot

### Non-interactive mode

Pass all options directly:

```bash
trainite init my-experiment \
  --model rope-transformer basic-transformer \
  --dataset string-reverse \
  --trainer decoder-trainer \
  --output-root outputs \
  --run-name rope_vs_basic \
  --sky
```

### Options

- `PROJECT_DIR` (positional): Output directory for the generated project (default: `my-cool-experiment`).
- `--model`: One or more model templates. At least one is required.
- `--dataset`: Dataset template.
- `--trainer`: Trainer template.
- `--output-root`: Value written to `output.root` in generated `config.yaml` (default: `outputs`).
- `--run-name`: Value written to `output.run_name` in generated `config.yaml` (default: `<first_model>__<dataset>`, with `-` replaced by `_`).
- `--sky`: Also generate `sky.yaml` and include SkyPilot dependency in generated `pyproject.toml`.
- `--force`: Overwrite starter files in a non-empty existing directory.

Available template choices:

- Models: `rope-transformer`, `basic-transformer`
- Datasets: `string-reverse`, `counting`, `hugging-face`, `wikitext`, `ultrachat-200k`, `python-edu`
- Trainers: `decoder-trainer`

### Multi-model scaffolding

You can pass multiple models in a single `--model` flag:

```bash
trainite init my-experiment \
  --model rope-transformer basic-transformer \
  --dataset counting \
  --trainer decoder-trainer
```

When multiple models are selected:

- Trainite scaffolds each model template into `models/` (for example, `models/rope_transformer.py` and `models/basic_transformer.py`).
- The **first** model is treated as the primary model for generated `config.yaml`.
- In interactive mode, Trainite asks which selected model should be the primary active model.

## `trainite add:sky`

Enable SkyPilot support in an **existing** Trainite-generated project.

Run this command from your project directory (the one containing `config.yaml` and `main.py`):

```bash
cd my-experiment
trainite add:sky
```

What it does:

1. Generates `sky.yaml` from Trainite's SkyPilot template.
2. Uses `project_name` from `config.yaml` (or falls back to the folder name) in the generated config.
3. Adds a `skypilot` dependency to `pyproject.toml` (if it is not already present).

Option:

- `--force`: Overwrite an existing `sky.yaml`.

If `config.yaml` or `main.py` is missing, the command exits with an error and asks you to run it from a valid Trainite experiment directory.
