# CLI Guide

Trainite provides a command-line interface (CLI) for creating and configuring training projects.

> The examples below assume Trainite is installed and available as the `trainite` command. If you are using `uv`, you can prefix the commands with `uv run`.

## Getting Help

To see all available commands:

```bash
trainite --help
```

To see help for the `init` command:

```bash
trainite init --help
```

To see help for the `add:sky` command:

```bash
trainite add:sky --help
```

## Check the Version

To check the installed version of Trainite:

```bash
trainite --version
```

You can also use:

```bash
trainite -V
```

## `trainite init`

The `init` command creates a new Trainite training project.

### Interactive Mode

Run `init` without any arguments:

```bash
trainite init
```

Trainite starts an interactive setup and prompts you for:

1. Project directory
2. Model(s)
3. Dataset
4. Trainer
5. Output directory
6. Run name
7. Whether to enable SkyPilot cloud training

You can select one or more models. If you select multiple models, Trainite asks you to choose which model should be the primary active model in `config.yaml`.

The default interactive choices include:

- Model: `rope-transformer`
- Dataset: `string-reverse`
- Trainer: `decoder-trainer`
- Output directory: `outputs`

### Non-interactive Mode

You can provide the configuration directly as command-line arguments.

For example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

This creates a project named `my-experiment`.

### Models

The currently available models are:

- `basic-transformer`
- `rope-transformer`

To select a model:

```bash
trainite init my-experiment \
    --model rope-transformer
```

Multiple models can be provided:

```bash
trainite init my-experiment \
    --model basic-transformer rope-transformer
```

### Datasets

The currently available datasets are:

- `string-reverse`
- `counting`
- `hugging-face`

For example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset counting
```

### Trainers

The currently available trainer is:

- `decoder-trainer`

For example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

### `--output-root`

Use `--output-root` to specify where training outputs should be stored.

The default value is `outputs`.

Example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer \
    --output-root experiment-outputs
```

### `--run-name`

Use `--run-name` to specify a custom name for the training run.

Example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer \
    --run-name my-first-run
```

### `--sky`

Use `--sky` to enable SkyPilot cloud-training support when creating a project.

Example:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer \
    --sky
```

This generates a `sky.yaml` configuration along with the project.

### `--force`

By default, Trainite does not overwrite an existing non-empty project directory.

Use `--force` when you want to overwrite generated starter files in the target directory:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer \
    --force
```

## `trainite add:sky`

The `add:sky` command adds SkyPilot support to an existing Trainite project.

Run the command from inside an existing Trainite experiment:

```bash
cd my-experiment
trainite add:sky
```

The project must contain:

```text
config.yaml
main.py
```

Trainite generates a `sky.yaml` configuration and adds the required SkyPilot dependency to the project's `pyproject.toml`.

### Force Overwrite

If `sky.yaml` already exists, Trainite will not overwrite it by default.

Use `--force` to replace the existing configuration:

```bash
trainite add:sky --force
```

## Running a Generated Project

After creating a project:

```bash
trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

Move into the generated project:

```bash
cd my-experiment
```

Install the project:

```bash
pip install -e .
```

Run the training:

```bash
python main.py config.yaml
```

If you are using `uv`, the generated project's training command can instead be run with:

```bash
uv run python main.py config.yaml
```

## SkyPilot Workflow

After using either:

```bash
trainite init my-experiment --sky
```

or:

```bash
cd my-experiment
trainite add:sky
```

install the project dependencies:

```bash
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

Check your SkyPilot setup:

```bash
sky check
```

Launch the training job:

```bash
sky launch sky.yaml
```

View running clusters and jobs:

```bash
sky queue
```

View logs for a cluster:

```bash
sky logs <cluster_name>
```

Stop and remove a cluster:

```bash
sky down <cluster_name>
```

## Command Summary

| Command | Purpose |
|---|---|
| `trainite --help` | Show all available CLI commands |
| `trainite --version` | Show the installed Trainite version |
| `trainite -V` | Show the installed Trainite version |
| `trainite init` | Start interactive project creation |
| `trainite init --help` | Show `init` options |
| `trainite init <project>` | Create a project non-interactively |
| `trainite init <project> --force` | Overwrite generated files in an existing project |
| `trainite init <project> --sky` | Create a project with SkyPilot support |
| `trainite add:sky` | Add SkyPilot support to an existing project |
| `trainite add:sky --force` | Overwrite an existing `sky.yaml` |
| `trainite add:sky --help` | Show `add:sky` options |

## Using `uv`

The commands above assume Trainite is installed and available directly as `trainite`.

If you are working from the Trainite repository with `uv`, prefix Trainite CLI commands with `uv run`.

For example:

```bash
uv run trainite --help
```

```bash
uv run trainite --version
```

```bash
uv run trainite init my-experiment \
    --model rope-transformer \
    --dataset string-reverse \
    --trainer decoder-trainer
```

The generated project's training command can also be run with:

```bash
uv run python main.py config.yaml
```

## CLI Documentation Validation

From the Trainite repository, build the documentation with:

```bash
uv run --group docs mkdocs build --strict
```
