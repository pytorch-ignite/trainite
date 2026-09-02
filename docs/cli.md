# CLI Guide

Trainite provides a command-line interface (CLI) for creating and configuring training projects.

## Getting Help

To see the available commands:

```bash
uv run trainite --help
```

Trainite provides the following commands:

- `init` — Generate a starter training project.
- `add:sky` — Add a SkyPilot configuration to an existing Trainite project.

To see the options for a specific command:

```bash
uv run trainite init --help
```

```bash
uv run trainite add:sky --help
```

## Check the Version

To check the installed version of Trainite:

```bash
uv run trainite --version
```

## `trainite init`

The `init` command generates a starter training project.

### Interactive Mode

Run `init` without any options to start the interactive setup:

```bash
uv run trainite init
```

Trainite will prompt you for:

1. Project directory
2. Model or models
3. Dataset
4. Trainer
5. Output directory
6. Run name
7. Whether to enable SkyPilot cloud training

### Non-interactive Mode

You can provide the project configuration directly through command-line options.

For example:

```bash
uv run trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

This creates a starter training project in the `my-experiment` directory with the specified model, dataset, and trainer.

### Command Options

#### `--model`

Select the model or models to use.

Available models:

- `basic-transformer`
- `rope-transformer`

Example:

```bash
uv run trainite init my-experiment \
    --model rope-transformer
```

Multiple models can be selected:

```bash
uv run trainite init my-experiment \
    --model basic-transformer rope-transformer
```

#### `--dataset`

Select the dataset.

Available datasets:

- `string-reverse`
- `counting`
- `hugging-face`

Example:

```bash
uv run trainite init my-experiment \
    --model rope-transformer \
    --dataset counting
```

#### `--trainer`

Select the trainer.

Available trainer:

- `decoder-trainer`

Example:

```bash
uv run trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

#### `--output-root`

Specify the root directory for training outputs.

The default is `outputs`.

Example:

```bash
uv run trainite init my-experiment \
    --output-root experiment-outputs
```

#### `--run-name`

Specify a custom name for the training run.

Example:

```bash
uv run trainite init my-experiment \
    --run-name my-first-run
```

#### `--sky`

Enable SkyPilot cloud-training support and generate the `sky.yaml` configuration.

Example:

```bash
uv run trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer \
    --sky
```

#### `--force`

Overwrite existing starter files in the target project directory.

Use this when the target directory already contains generated starter files that you want to replace.

Example:

```bash
uv run trainite init my-experiment --force \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

## `trainite add:sky`

The `add:sky` command adds a SkyPilot configuration to an existing Trainite project.

Run it from inside an existing Trainite project:

```bash
uv run trainite add:sky
```

The project must contain:

```text
config.yaml
main.py
```

### Overwrite an Existing Configuration

If `sky.yaml` already exists, Trainite will not overwrite it by default.

Use `--force` to overwrite the existing configuration:

```bash
uv run trainite add:sky --force
```

### Using the Generated SkyPilot Configuration

After running `trainite add:sky`, update the project environment:

```bash
pip install -e .
```

or, with `uv`:

```bash
uv sync
```

Check that SkyPilot is available:

```bash
sky check
```

Launch the training job:

```bash
sky launch sky.yaml
```

View running jobs:

```bash
sky queue
```

View logs for a cluster:

```bash
sky logs <cluster_name>
```

Shut down a cluster when it is no longer needed:

```bash
sky down <cluster_name>
```

## Running a Generated Project

After creating a project with `trainite init`, move into the generated project:

```bash
cd my-experiment
```

Install the project:

```bash
pip install -e .
```

Run the training script with the generated configuration:

```bash
python main.py config.yaml
```

When using `uv`, you can run:

```bash
uv run python main.py config.yaml
```

## Command Reference

Use the built-in help to see the complete and current CLI options:

```bash
uv run trainite --help
```

```bash
uv run trainite init --help
```

```bash
uv run trainite add:sky --help
```
