# Trainite Project: {{project_name}}

> Generated with Trainite v{{trainite_version}} using:
> ```bash
> {{recreation_command}}
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

## Experiment Logging and Checkpoints

The default `logger: tensorboard` stores metrics and checkpoints locally. To use ClearML instead:

1. Run `clearml-init` to configure your ClearML server and credentials.
2. Set `logger: clearml` in `config.yaml`.

ClearML runs still keep checkpoints in the local output directory. By default, `ClearMLSaver` also uploads them to
ClearML's file server with `output_uri=True`.

To change checkpoint storage, edit the `ClearMLSaver` call in `trainer.py`:

- Use a storage URI such as `s3://bucket/path` to upload somewhere else.
- Use `None` to leave the destination to `CLEARML_DEFAULT_OUTPUT_URI` or
  `sdk.development.default_output_uri` in `clearml.conf`.

To disable checkpoint uploads while retaining ClearML metric logging, pass `output_uri=False` to `ClearMLLogger` in
`utils.py` and use `output_uri=None` for `ClearMLSaver` in `trainer.py`.

## Components

### Models: {{model_name}}
{{model_docs}}

### Dataset: {{dataset_name}}
{{dataset_docs}}

### Trainer: {{trainer_name}}
{{trainer_docs}}

### Preprocessor: {{preprocessor_name}}
{{preprocessor_docs}}

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
