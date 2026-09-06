# Trainite

Trainite generates self-contained PyTorch training projects from a small set of
tested building blocks. Choose a model, dataset, and trainer, then use the
generated project as a readable starting point for your experiment.

The generated code belongs to your project. It includes the model, data
pipeline, trainer, configuration, and declared dependencies needed to run the
experiment, so the project does not need Trainite at runtime.

## How it works

1. Run `trainite init` and select starter components.
2. Install the dependencies declared by the generated project.
3. Run `main.py` with the generated `config.yaml`.
4. Edit the local Python modules and configuration for your experiment.

Trainite currently provides these starting points:

- **Models:** basic Transformer and rotary-position Transformer
- **Datasets:** string reversal, counting, and Hugging Face datasets
- **Trainer:** decoder training powered by PyTorch-Ignite

The command-line interface supports an interactive setup as well as explicit
options for scripts and reproducible project generation.

[Create your first project](getting-started.md){ .md-button .md-button--primary }
