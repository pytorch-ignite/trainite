
## Hyperparameter Sweep

This project includes sweep support via [Optuna](https://optuna.org/). Edit `sweep.yaml` to configure your search, then run `python sweep.py`.

### Strategies

#### Grid Search (`strategy: "grid"`)
Exhaustively tries every combination of the listed values. Best for small, discrete search spaces.

```yaml
strategy: "grid"
direction: "maximize"
metric: "exact_accuracy"

parameters:
  model.num_heads: [1, 2, 4]
  model.num_layers: [1, 2, 4]
  optimizer.lr: [0.01, 0.001]
```
This runs 3 × 3 × 2 = 18 trials.

#### Random Search (`strategy: "random"`)
Samples random combinations from the parameter space. Supports both explicit lists and continuous ranges. Requires `n_trials`.

```yaml
strategy: "random"
direction: "minimize"
metric: "loss"
n_trials: 30

parameters:
  # Explicit list — picks randomly from these values
  model.num_layers: [2, 4, 6]

  # Continuous range — samples a float between low and high
  optimizer.lr:
    type: float
    low: 0.0001
    high: 0.01
    sample: log       # sample on log scale ("uniform" or "log", default: "uniform")

  # Integer range — samples an integer between low and high
  model.num_heads:
    type: int
    low: 1
    high: 8
```

#### Bayesian Optimization (`strategy: "tpe"`)
Uses Tree-structured Parzen Estimators to learn from previous trials and suggest better parameters. Most efficient for large search spaces. Requires `n_trials`. Same parameter format as random search.

```yaml
strategy: "tpe"
direction: "maximize"
metric: "exact_accuracy"
n_trials: 50

parameters:
  optimizer.lr:
    type: float
    low: 0.00001
    high: 0.01
    sample: log
  model.num_heads: [1, 2, 4, 8]
  model.num_layers:
    type: int
    low: 1
    high: 6
```

### Running a Sweep
```bash
uv run python sweep.py
```

### Results
Sweep results are saved under `outputs/<run_name>/sweep_<timestamp>/`.
Each trial gets its own subdirectory with checkpoints and logs.
A `sweep_summary.yaml` in the sweep directory contains the best parameters found and a summary of all trials.
