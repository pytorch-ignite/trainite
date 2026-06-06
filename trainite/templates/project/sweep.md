## Hyperparameter Sweep

This project includes sweep support via [Optuna](https://optuna.org/). Edit `sweep.yaml` to configure your search, then run `python sweep.py`.

### Example `sweep.yaml` Configuration

Here is a complete, copy-pasteable example of how `sweep.yaml` fits together:

```yaml
# 1. Search strategy and direction
strategy: "grid"          # Options: "grid", "random", "tpe", "brute_force", "cmaes", "gp", "nsgaii", "qmc"
direction: "maximize"     # "maximize" or "minimize"
metric: "exact_accuracy"  # The validation metric to optimize
n_trials: 20              # Number of trials (required for random, tpe, cmaes, gp, nsgaii, qmc)

# 2. Resuming a Sweep (Optional)
# To save and resume sweeps, define a SQLite storage database.
storage: "sweep_study.db"

# 3. Pruning (Early Stopping) (Optional)
# Automatically stop underperforming trials early to save compute resources.
# Pruning is disabled by default. If you define this block with 'enabled: false',
# do not include any other pruning fields.
pruning:
  enabled: true
  type: "median"          # Options: "median", "percentile", "hyperband", "successive_halving", "threshold", "patient", "wilcoxon", "nop"
  n_startup_trials: 5     # Run 5 trials fully to establish a baseline before pruning
  n_warmup_steps: 1       # Epochs to run in a trial before checking if it should prune

# 4. Search space parameters
parameters:
  # Explicit lists (compatible with all strategies)
  model.num_heads: [2, 4, 8]

  # Continuous float ranges (compatible with random/tpe strategies)
  optimizer.lr:
    type: float
    low: 0.0001
    high: 0.01
    sample: log           # "uniform" or "log"

  # Integer ranges (compatible with random/tpe strategies)
  model.num_layers:
    type: int
    low: 1
    high: 8
    step: 2               # optional step size
```

### Supported Search Strategies

*   **`grid`**: Exhaustively tries every combination of the parameters. Only explicit lists `[val1, val2, ...]` are supported. The total number of trials is computed automatically.
*   **`random`**: Samples random combinations from the parameter space. Supports lists and ranges.
*   **`tpe`**: Tree-structured Parzen Estimators. A Bayesian optimization algorithm that learns from previous trials to suggest better parameters.
*   **`brute_force`**: Suggests parameters using brute-force search.
*   **`cmaes`**: Covariance Matrix Adaptation Evolution Strategy. A derivative-free evolutionary algorithm for continuous search spaces.
*   **`gp`**: Gaussian Process-based Bayesian optimization.
*   **`nsgaii`**: Non-dominated Sorting Genetic Algorithm II. Often used for multi-objective optimization.
*   **`qmc`**: Quasi-Monte Carlo sampler using low-discrepancy sequences (e.g. Sobol/Halton sequences).

*Note: All strategies other than `grid` and `brute_force` require `n_trials` to be defined.*

### Supported Pruning Types (Early Stopping)

Pruning is disabled by default if the `pruning` block is omitted. If you explicitly define the `pruning` block with `enabled: false`, no other pruning parameters (such as `type`, `percentile`, `patience`, etc.) are permitted.

When `enabled: true`, you can choose one of the following types:
*   **`median`**: Prunes if the trial's metric at the current epoch is worse than the median of previous trials at that epoch.
*   **`percentile`**: Prunes if the trial's metric is in the bottom percentile.
    *   *Parameters*: `percentile` (float, e.g. `25.0`, required).
*   **`hyperband`**: An algorithm combining successive halving with adaptive resource allocation.
    *   *Parameters*: `min_resource` (int/str, default `"auto"`), `max_resource` (int/str, default `"auto"`), `reduction_factor` (int, default `4`), `bootstrap_count` (int, default `0`).
*   **`successive_halving`**: The base pruner algorithm used by Hyperband.
    *   *Parameters*: `min_resource` (int/str, default `"auto"`), `reduction_factor` (int, default `4`), `min_early_stopping_rate` (int, default `0`), `bootstrap_count` (int, default `0`).
*   **`threshold`**: Prunes if the metric falls outside specified bounds.
    *   *Parameters*: `lower` (float, optional), `upper` (float, optional). At least one must be set.
*   **`patient`**: Wraps another pruner and adds a patience window to delay the pruning decision.
    *   *Parameters*: `patience` (int, required), `min_delta` (float, default `0.0`), `wrapped_type` (default `"median"`).
*   **`wilcoxon`**: Uses Wilcoxon signed-rank test to prune trials based on performance distribution.
    *   *Parameters*: `p_threshold` (float, default `0.1`), `n_startup_steps` (int, default `2`).
*   **`nop`**: A no-op pruner. Does not prune any trials, but reports intermediate values.

### Running a Sweep

Run the sweep using the CLI:
```bash
# Run sequentially on the default device
uv run python sweep.py

# Specify custom configuration paths
uv run python sweep.py config.yaml --sweep sweep.yaml

# Run in parallel across multiple GPUs (one trial per GPU)
uv run python sweep.py --gpus 0,1

# Run with custom parallel count
uv run python sweep.py --gpus 0,1 --max-parallel 4
```

### Sweep Results

Sweep results are saved under `outputs/<run_name>/sweep_<timestamp>/`.
*   Each trial gets a descriptive subdirectory (e.g. `trial_0__lr=0.01_num_heads=2/`) with checkpoints and logs.
*   A `sweep_summary.yaml` in the sweep directory contains the best parameters found and a summary of all trials.
