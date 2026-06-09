# Trainite Project: {{project_name}}

Welcome to your new Trainite-generated training project!

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `datasets/`: Handles data loading and preprocessing.
- `trainer.py`: Defines the training and evaluation logic. You can override `train_step` or `eval_step` here.
- `config.py`: Contains Pydantic models for configuration validation. If you add new parameters to `config.yaml`, update the models here.
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

3. **Run Training**:
   ```bash
   python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   tensorboard --logdir outputs
   ```

## Components

### Model: {{model_name}}
{{model_docs}}

### Dataset: {{dataset_name}}
{{dataset_docs}}

### Trainer: {{trainer_name}}
{{trainer_docs}}

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
  train_ratio: 0.8
  val_ratio: 0.2
  dataloader:
    batch_size: 32
```
*Note: In Option 2, the training split defaults to `shuffle: true`, while others default to `shuffle: false`.*

### Adding Configuration Parameters
1. Add the parameter to `config.yaml`.
2. Update the corresponding Pydantic model in `config.py` to include the new field.

## Design Philosophy
Trainite follows a "Cookiecutter, not framework" approach. The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
