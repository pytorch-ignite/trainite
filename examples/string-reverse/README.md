# Trainite Project: string-reverse

Welcome to your new Trainite-generated training project! 

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `dataset/`: Handles data loading and preprocessing.
- `trainer.py`: Defines the training and evaluation logic. You can override `train_step` or `eval_step` here.
- `config.py`: Contains Pydantic models for configuration validation. If you add new parameters to `config.yaml`, update the models here.
- `utils.py`: Shared utilities for configuration and logging.

## Getting Started

1. **Install Dependencies**:
   Ensure you have PyTorch, PyTorch-Ignite, and Pydantic installed.
   ```bash
   uv sync
   ```

2. **Run Training**:
   ```bash
   uv run python main.py config.yaml
   ```

3. **Monitor Progress**:
   Training logs and checkpoints are saved to the directory specified in `config.yaml` (default: `outputs/`). You can use TensorBoard to visualize metrics:
   ```bash
   uv run tensorboard --logdir outputs
   ```

## Customization

### Modifying the Model
Edit the files in `models/`. You can change the architecture, add new layers, or change how the loss is calculated if it's part of the model.

### Using Your Own Data
Update the dataset factory and `Dataset` class in `dataset/`. In `config.yaml`, the `data` section allows you to configure `train`, `val` and `test` splits separately. Each split contains:
- `dataset`: Configuration for the dataset builder.
- `dataloader`: Parameters for the PyTorch DataLoader (batch size, shuffle, num_workers, etc.).

If you omit the `val` section, the trainer will fall back to using the `train` configuration for validation (with a warning).

If you omit the `test` section, the trainer will skip the testing phase after training (with a warning).

### Custom Training Logic
If you need to change the training loop (e.g., add gradient clipping, custom logging, or a different optimization step), edit `trainer.py`. You can override methods of the `PreTrainer` (or `RLTrainer`) class.

### Adding Configuration Parameters
1. Add the parameter to `config.yaml`.
2. Update the corresponding Pydantic model in `config.py` to include the new field.

## Design Philosophy
Trainite follows a "Cookiecutter, not framework" approach. The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
