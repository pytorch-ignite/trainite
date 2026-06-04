import logging
import shutil
from datetime import datetime
from pathlib import Path

import optuna
import yaml

from trainite.config import ProjectConfig, load_config
from trainite.config.sweep import SweepConfig, load_sweep_config
from trainite.sweep_utils import (
    apply_overrides,
    build_sampler,
    suggest_params,
    validate_sweep_params,
)
from trainite.trainers import PreTrainer


def objective(
    trial: optuna.Trial,
    base_config: ProjectConfig,
    sweep_config: SweepConfig,
    sweep_dir: Path,
) -> float:
    overrides = suggest_params(trial, sweep_config.parameters)
    run_config = apply_overrides(base_config, overrides)

    trial_name = f"trial_{trial.number}"
    run_config.output.root = str(sweep_dir)
    run_config.output.run_name = trial_name

    logger = logging.getLogger("sweep")
    logger.info("Trial %d | Parameters: %s", trial.number, overrides)

    trainer = PreTrainer(run_config)
    trainer.run()

    metric_value = trainer.val_evaluator.state.metrics[sweep_config.metric]
    logger.info(
        "Trial %d | %s = %.6f",
        trial.number,
        sweep_config.metric,
        metric_value,
    )
    return metric_value


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence Ignite internal logs
    logging.getLogger("ignite.engine").setLevel(logging.WARNING)
    logger = logging.getLogger("sweep")

    base_config = load_config("config.yaml")
    sweep_config = load_sweep_config("sweep.yaml")

    logger.info("Validating sweep parameters...")
    validate_sweep_params(base_config, sweep_config)
    logger.info("Validation passed.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(base_config.output.root)
        / base_config.output.run_name
        / f"sweep_{timestamp}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2("sweep.yaml", sweep_dir / "sweep.yaml")

    sampler = build_sampler(sweep_config, seed=base_config.seed)
    study = optuna.create_study(
        direction=sweep_config.direction,
        sampler=sampler,
        study_name=f"trainite-sweep-{timestamp}",
    )

    n_trials = sweep_config.n_trials
    if sweep_config.strategy == "grid":
        total = 1
        for val in sweep_config.parameters.values():
            if isinstance(val, list):
                total *= len(val)
        n_trials = total
        logger.info("Grid search: %d total combinations", n_trials)

    study.optimize(
        lambda trial: objective(trial, base_config, sweep_config, sweep_dir),
        n_trials=n_trials,
    )

    summary = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }
    summary_path = sweep_dir / "sweep_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    logger.info("Sweep complete!")
    logger.info(
        "Best trial: #%d with %s=%.6f",
        study.best_trial.number,
        sweep_config.metric,
        study.best_trial.value,
    )
    logger.info("Best parameters: %s", study.best_trial.params)
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
