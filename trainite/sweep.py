import argparse
import logging
import queue
import shutil
from datetime import datetime
from pathlib import Path

import optuna
import yaml
from ignite.engine import Events

from trainite.config import ProjectConfig, load_config
from trainite.config.sweep import SweepConfig, load_sweep_config
from trainite.sweep_utils import (
    allocate_gpu,
    apply_overrides,
    build_gpu_queue,
    build_pruner,
    build_sampler,
    build_trial_name,
    cleanup_trial,
    suggest_params,
    validate_sweep_params,
)
from trainite.trainers import PreTrainer


def objective(
    trial: optuna.Trial,
    base_config: ProjectConfig,
    sweep_config: SweepConfig,
    sweep_dir: Path,
    gpu_queue: queue.Queue,
) -> float:
    with allocate_gpu(gpu_queue):
        overrides = suggest_params(trial, sweep_config.parameters)
        run_config = apply_overrides(base_config, overrides)

        trial_name = build_trial_name(trial.number, overrides)
        run_config.output.root = str(sweep_dir)
        run_config.output.run_name = trial_name

        logger = logging.getLogger("sweep")
        logger.info("Trial %d | Parameters: %s", trial.number, overrides)

        trainer = PreTrainer(run_config)

        # Pruning hook
        if sweep_config.pruning.enabled and trainer.val_evaluator:

            @trainer.val_evaluator.on(Events.EPOCH_COMPLETED)
            def report_and_prune(engine):
                metric_val = engine.state.metrics.get(sweep_config.metric, 0.0)
                trial.report(metric_val, step=trainer.engine.state.epoch)
                if trial.should_prune():
                    if (
                        hasattr(trainer, "handlers")
                        and "tensorboard" in trainer.handlers
                    ):
                        trainer.handlers["tensorboard"].close()
                    raise optuna.exceptions.TrialPruned()

        try:
            trainer.run()
            metric_value = trainer.val_evaluator.state.metrics[sweep_config.metric]
        finally:
            cleanup_trial()

        logger.info(
            "Trial %d | %s = %.6f",
            trial.number,
            sweep_config.metric,
            metric_value,
        )
        return metric_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps using Optuna."
    )
    parser.add_argument(
        "config", nargs="?", default="config.yaml", help="Path to base config.yaml"
    )
    parser.add_argument(
        "--sweep", default="sweep.yaml", help="Path to sweep.yaml config"
    )
    parser.add_argument(
        "--gpus", default=None, help="Comma-separated GPU indices (e.g. '0,1')"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Number of concurrent trials (defaults to GPU count)",
    )
    args = parser.parse_args()

    if args.max_parallel is None:
        if args.gpus:
            args.max_parallel = len(args.gpus.split(","))
        else:
            args.max_parallel = 1

    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence Ignite internal logs
    logging.getLogger("ignite.engine").setLevel(logging.WARNING)
    logger = logging.getLogger("sweep")

    args = parse_args()

    base_config = load_config(args.config)
    sweep_config = load_sweep_config(args.sweep)

    logger.info("Validating sweep parameters...")
    validate_sweep_params(base_config, sweep_config)
    logger.info("Validation passed.")

    # Build infrastructure
    gpu_queue = build_gpu_queue(args.gpus)
    sampler = build_sampler(sweep_config, seed=base_config.seed)
    pruner = build_pruner(sweep_config.pruning)

    # Storage
    storage = f"sqlite:///{sweep_config.storage}" if sweep_config.storage else None

    # Sweep output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = (
        Path(base_config.output.root)
        / base_config.output.run_name
        / f"sweep_{timestamp}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.sweep, sweep_dir / "sweep.yaml")

    # Create study
    study = optuna.create_study(
        direction=sweep_config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=f"trainite-sweep-{timestamp}",
        load_if_exists=True,
    )

    # Determine trial count
    n_trials = sweep_config.n_trials
    if sweep_config.strategy == "grid":
        total = 1
        for val in sweep_config.parameters.values():
            if isinstance(val, list):
                total *= len(val)
        n_trials = total
        logger.info("Grid search: %d total combinations", n_trials)

    # Run
    study.optimize(
        lambda trial: objective(trial, base_config, sweep_config, sweep_dir, gpu_queue),
        n_trials=n_trials,
        n_jobs=args.max_parallel,
    )

    # Save summary
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        best_trial_data = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        }
    else:
        best_trial_data = None

    summary = {
        "best_trial": best_trial_data,
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
    if best_trial_data:
        logger.info(
            "Best trial: #%d with %s=%.6f",
            study.best_trial.number,
            sweep_config.metric,
            study.best_trial.value,
        )
        logger.info("Best parameters: %s", study.best_trial.params)
    else:
        logger.info("No trials completed successfully.")
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
