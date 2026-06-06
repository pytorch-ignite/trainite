import gc
import queue
from contextlib import contextmanager
from typing import Any

import optuna
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

from trainite.config.base import ProjectConfig
from trainite.config.sweep import ParameterRange, PruningConfig, SweepConfig


def _validate_path(config: BaseModel, path: str) -> None:
    """Walk a dot-notated path on a Pydantic model to verify it exists.

    Args:
        config: The root Pydantic model to traverse.
        path: Dot-notated path like "model.num_heads".

    Raises:
        ValueError: If any segment of the path does not exist.
    """
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise ValueError(
                f"Invalid parameter path '{path}': "
                f"'{part}' not found on {type(current).__name__}"
            )
        current = getattr(current, part)
        if not isinstance(current, BaseModel):
            raise ValueError(
                f"Invalid parameter path '{path}': "
                f"'{part}' is not a nested config object"
            )
    field_name = parts[-1]
    if field_name not in current.model_fields:
        is_extra_allowed = current.model_config.get("extra") == "allow"
        if not is_extra_allowed or not hasattr(current, field_name):
            available = list(current.model_fields.keys())
            if is_extra_allowed:
                extra_fields = [
                    k
                    for k in current.__dict__
                    if k not in current.model_fields and not k.startswith("_")
                ]
                available.extend(extra_fields)
            raise ValueError(
                f"Invalid parameter path '{path}': "
                f"'{field_name}' not found on {type(current).__name__}. "
                f"Available fields: {available}"
            )


def validate_sweep_params(
    base_config: ProjectConfig, sweep_config: SweepConfig
) -> None:
    """Validate all sweep parameter paths and values against the ProjectConfig schema.

    For each parameter in the sweep config, this function:
    1. Verifies the dot-notated path exists in the ProjectConfig model tree
    2. Attempts to set each candidate value and construct a full ProjectConfig to catch cross-field validation errors

    This runs BEFORE any training starts, so typos and invalid values are caught
    immediately rather than crashing mid-sweep.

    Raises:
        ValueError: If any parameter path is invalid or any value fails validation.
    """
    if sweep_config.pruning.enabled and sweep_config.pruning.type == "nop":
        import logging

        logging.getLogger("sweep").warning(
            "WARNING: Pruning is enabled ('enabled: true') but type is set to 'nop' "
            "(or not specified). No trials will be pruned. To enable actual early "
            "stopping, please specify a pruner type in sweep.yaml (e.g., type: 'median')."
        )

    for param_path, param_values in sweep_config.parameters.items():
        _validate_path(base_config, param_path)

        if isinstance(param_values, list):
            values_to_check = param_values
        elif isinstance(param_values, ParameterRange):
            # For ranges, validate the boundary values
            values_to_check: list[Any] = [param_values.low, param_values.high]
            if param_values.type == "int":
                values_to_check = [int(v) for v in values_to_check]
        else:
            raise ValueError(
                f"Parameter '{param_path}' has invalid definition type: "
                f"{type(param_values).__name__}"
            )

        for val in values_to_check:
            try:
                apply_overrides(base_config, {param_path: val})
            except Exception as e:
                raise ValueError(
                    f"Invalid value {val!r} for parameter '{param_path}': {e}"
                ) from e


def apply_overrides(config: ProjectConfig, overrides: dict[str, Any]) -> ProjectConfig:
    """Create a deep copy of the config with the given parameter overrides applied.

    Args:
        config: The base ProjectConfig to copy.
        overrides: Dict mapping dot-notated paths to new values.

    Returns:
        A new ProjectConfig with the overrides applied.
    """
    config_dict = OmegaConf.create(config.model_dump(by_alias=True))
    for path, value in overrides.items():
        OmegaConf.update(config_dict, path, value)
    return ProjectConfig.model_validate(OmegaConf.to_container(config_dict))


def cleanup_trial() -> None:
    """Release GPU memory and trigger garbage collection between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_pruner(pruning_config: PruningConfig) -> optuna.pruners.BasePruner:
    """Create the appropriate Optuna pruner from PruningConfig."""
    if not pruning_config.enabled:
        return optuna.pruners.NopPruner()
    return _build_raw_pruner(pruning_config.type, pruning_config)


def _build_raw_pruner(
    pruner_type: str, pruning_config: PruningConfig
) -> optuna.pruners.BasePruner:
    if pruner_type == "nop":
        return optuna.pruners.NopPruner()
    elif pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruning_config.n_startup_trials,
            n_warmup_steps=pruning_config.n_warmup_steps,
            interval_steps=pruning_config.interval_steps,
            n_min_trials=pruning_config.n_min_trials,
        )
    elif pruner_type == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=pruning_config.percentile,
            n_startup_trials=pruning_config.n_startup_trials,
            n_warmup_steps=pruning_config.n_warmup_steps,
            interval_steps=pruning_config.interval_steps,
            n_min_trials=pruning_config.n_min_trials,
        )
    elif pruner_type == "hyperband":
        min_res = pruning_config.min_resource
        if isinstance(min_res, str) and min_res.isdigit():
            min_res = int(min_res)
        max_res = pruning_config.max_resource
        if isinstance(max_res, str) and max_res.isdigit():
            max_res = int(max_res)

        return optuna.pruners.HyperbandPruner(
            min_resource=min_res if min_res == "auto" else int(min_res),
            max_resource=max_res if max_res == "auto" else int(max_res),
            reduction_factor=pruning_config.reduction_factor,
            bootstrap_count=pruning_config.bootstrap_count,
        )
    elif pruner_type == "successive_halving":
        min_res = pruning_config.min_resource
        if isinstance(min_res, str) and min_res.isdigit():
            min_res = int(min_res)
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_res if min_res == "auto" else int(min_res),
            reduction_factor=pruning_config.reduction_factor,
            min_early_stopping_rate=pruning_config.min_early_stopping_rate,
            bootstrap_count=pruning_config.bootstrap_count,
        )
    elif pruner_type == "threshold":
        return optuna.pruners.ThresholdPruner(
            lower=pruning_config.lower,
            upper=pruning_config.upper,
            n_warmup_steps=pruning_config.n_warmup_steps,
            interval_steps=pruning_config.interval_steps,
        )
    elif pruner_type == "wilcoxon":
        return optuna.pruners.WilcoxonPruner(
            p_threshold=pruning_config.p_threshold,
            n_startup_steps=pruning_config.n_startup_steps,
        )
    elif pruner_type == "patient":
        wrapped_t = pruning_config.wrapped_type
        if wrapped_t == "patient":
            wrapped_t = "median"
        inner = _build_raw_pruner(wrapped_t, pruning_config)
        return optuna.pruners.PatientPruner(
            wrapped_pruner=inner,
            patience=pruning_config.patience,
            min_delta=pruning_config.min_delta,
        )
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def build_gpu_queue(gpu_str: str | None) -> queue.Queue:
    """Create a queue of GPU device indices from a comma-separated string."""
    gpu_queue = queue.Queue()
    if gpu_str:
        for g in gpu_str.split(","):
            gpu_queue.put(g.strip())
    return gpu_queue


@contextmanager
def allocate_gpu(gpu_queue: queue.Queue):
    """Context manager that checks out a GPU from the queue and returns it when done."""
    gpu = gpu_queue.get() if not gpu_queue.empty() else None
    try:
        if gpu is not None:
            torch.cuda.set_device(f"cuda:{gpu}")
        yield gpu
    finally:
        if gpu is not None:
            gpu_queue.put(gpu)


def build_trial_name(trial_number: int, overrides: dict[str, Any]) -> str:
    """Build a descriptive trial directory name like 'trial_0__lr=0.01_num_heads=2'."""
    if not overrides:
        return f"trial_{trial_number}"
    suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
    return f"trial_{trial_number}__{suffix}"


def build_sampler(
    sweep_config: SweepConfig, seed: int | None = None
) -> optuna.samplers.BaseSampler:
    """Create the appropriate Optuna sampler for the given sweep strategy.

    Args:
        sweep_config: The sweep configuration specifying the strategy.
        seed: Optional random seed for reproducible sampling.

    Returns:
        An Optuna sampler instance.
    """
    if sweep_config.strategy == "grid":
        search_space: dict[str, list[Any]] = {}
        for key, val in sweep_config.parameters.items():
            if not isinstance(val, list):
                raise ValueError(
                    f"Grid strategy requires explicit lists, "
                    f"but got a range for '{key}'"
                )
            search_space[key] = val
        return optuna.samplers.GridSampler(search_space, seed=seed)
    elif sweep_config.strategy == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sweep_config.strategy == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sweep_config.strategy == "brute_force":
        return optuna.samplers.BruteForceSampler(seed=seed)
    elif sweep_config.strategy == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sweep_config.strategy == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    elif sweep_config.strategy == "nsgaii":
        return optuna.samplers.NSGAIISampler(seed=seed)
    elif sweep_config.strategy == "qmc":
        return optuna.samplers.QMCSampler(seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {sweep_config.strategy}")


def suggest_params(
    trial: optuna.Trial,
    parameters: dict[str, list[Any] | ParameterRange],
) -> dict[str, Any]:
    """Use an Optuna trial object to suggest values for each sweep parameter.

    Args:
        trial: The current Optuna trial.
        parameters: Dict mapping parameter paths to lists or ParameterRange objects.

    Returns:
        Dict mapping parameter paths to the suggested values for this trial.
    """
    suggestions: dict[str, Any] = {}
    for param_path, param_def in parameters.items():
        if isinstance(param_def, list):
            suggestions[param_path] = trial.suggest_categorical(param_path, param_def)
        elif isinstance(param_def, ParameterRange):
            log_val = param_def.sample == "log"
            if param_def.type == "int":
                suggestions[param_path] = trial.suggest_int(
                    param_path,
                    int(param_def.low),
                    int(param_def.high),
                    step=int(param_def.step) if param_def.step else 1,
                    log=log_val,
                )
            elif param_def.type == "float":
                kwargs: dict[str, Any] = {
                    "name": param_path,
                    "low": param_def.low,
                    "high": param_def.high,
                    "log": log_val,
                }
                if param_def.step is not None:
                    kwargs["step"] = param_def.step
                suggestions[param_path] = trial.suggest_float(**kwargs)
        else:
            raise ValueError(
                f"Unknown parameter definition type for '{param_path}': "
                f"{type(param_def).__name__}"
            )
    return suggestions
