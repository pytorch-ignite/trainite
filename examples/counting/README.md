# Reproducing "Knee-Deep in C-RASP: A Transformer Depth Hierarchy" Experiments

This example project reproduces the prefix-classification length-generalization sweep experiments on the alternating block language `L_k` described in the research paper:

> **Knee-Deep in C-RASP: A Transformer Depth Hierarchy**
> Andy Yang, Michaël Cadilhac, David Chiang
> *[arXiv:2506.16055](https://arxiv.org/pdf/2506.16055)*

The experimental setup and training configurations are replicated from the authors' official implementation repository:
* **GitHub Repository**: [pentagonalize/CRASP_depth](https://github.com/pentagonalize/CRASP_depth)

The experiments empirically verify the paper's theory predicting the depth required for transformers without positional encodings to length-generalize. Specifically, the theory states that a transformer of depth `d` without positional encodings can recognize language `L_k` up to `k = d + 2` block alternations, but cannot generalize beyond that (failing at `k = d + 3`). This establishes the theoretical generalization limit boundary at `k = d + 2`.

---

## Project Origin & Customization

This project was generated using the `trainite` CLI with the built-in counting dataset. *(Note: See the [main README](../../README.md) for Trainite installation instructions).*
```bash
trainite init --dataset counting
```

Following generation, the project was customized to align with the paper's experimental setup:
  1. **Model (`models/transformer.py`)**: Removed all positional encodings (Rotary Position Embeddings/RoPE) and set the output projection to support binary classification targets (`num_classes: 2`).
  2. **Trainer (`trainer.py`)**: Modified to evaluate sequence-level exact-match accuracy (`sequence_accuracy`) instead of token-level accuracy. It was also updated to terminate validation early if sequence accuracy reaches `1.0` (100%).
  3. **Sweeps (`sweep.py`)**: Custom script added to sweep across learning rates (`lr`), model dimensions (`dim`), depths (`depth`), and block alternations (`k`), appending results to `sweep.csv`.
  4. **Plotting (`plot_results.py`)**: Custom script added to parse `sweep.csv` and generate the transposed generalization heatmaps.

---

## The Counting Task (L_k Alternating Language)

The dataset generates random sequences from the alternating language `L_k` over the alphabet `{'a', 'b'}`:
* `L_1 = a^+`
* `L_{k+1} = L_k b^+` if `k` is odd
* `L_{k+1} = L_k a^+` if `k` is even

For prefix classification, the target labels indicate whether each character belongs to the final alternating block. The sequence transitions to target label `1` at the final switch index:
* **Example (k=1)**: `aaaaa` -> `11111`
* **Example (k=2)**: `aaabbb` -> `000111`
* **Example (k=3)**: `aaabbbaaa` -> `000000111`

Therefore, the model must learn to recognize `L_k` up to `k = d + 2` block alternations, but not beyond that. To keep the experiments manageable, for each model dimension `d`, we limit `k` to `d + 2`, `d + 3`, and `d + 4` in order to identify the model's generalization capabilities across different sequence lengths while keeping training manageable.

---

## Project Structure

* `config.yaml`: Baseline hyperparameters for a single training run.
* `main.py`: Entrypoint for single run. Run it with `python main.py config.yaml`.
* `sweep.py`: Outer loop hyperparameter sweep script.
* `plot_results.py`: Unified plotting script that generates heatmaps for every unique configuration and the aggregated best configurations.
* `models/transformer.py`: Causal transformer decoder without positional encodings.
* `datasets/counting.py`: Generator for `L_k` alternating sequences and prefix targets.
* `trainer.py`: Custom Ignite trainer loop tracking sequence exact match.
* `config.py`: Pydantic validation schemas.
* `utils.py`: Shared utilities for model building, dataloaders, and loggers.

---

## How to Run

### 1. Set Up Environment
Using **uv** (recommended):
```bash
uv sync
```

Or using **pip**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run a Baseline Experiment
Train a single model configuration defined in `config.yaml`:
```bash
uv run python main.py config.yaml
```

### 3. Run the Multi-Config Sweep
Execute the full sweep over model layers (`d = [1,...,10]`) and target blocks (`k = [d+2, d+3, d+4]`) for combinations of model dimensions (`256`, `512`) and learning rates (`1e-4`, `1e-5`):
```bash
uv run python sweep.py
```
This writes all results directly to `sweep.csv`.

### 4. Plot Heatmaps
Generate visual transposed generalization heatmaps (with model depth on the top x-axis and `k` on the left y-axis):
```bash
uv run python plot_results.py --input-csv sweep.csv
```
This yields individual heatmap charts (e.g. `sweep_heatmap_*_dim256_lr1e-4.png`) and the aggregated best configurations (`sweep_heatmap_*_best.png`), highlighting the theoretical limit boundary `k = d + 2` with a red outline.
