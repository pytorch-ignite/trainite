from typing import Any

import optuna
from pydantic import BaseModel

from trainite.config.base import ProjectConfig
from trainite.config.sweep import ParameterRange, SweepConfig


def _resolve_path(config: BaseModel, path: str) -> tuple[BaseModel, str]:
    """Walk a dot-notated path on a Pydantic model to find the parent and field name.

    Args:
        config: The root Pydantic model to traverse.
        path: Dot-notated path like "model.num_heads".

    Returns:
        A tuple of (parent_model, field_name) where parent_model is the
        Pydantic model containing the target field.

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
        # If the model allows extra fields (like ComponentConfig), we also accept
        # fields that are already present on the instance.
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
    return current, field_name


def validate_sweep_params(
    base_config: ProjectConfig, sweep_config: SweepConfig
) -> None:
    """Validate all sweep parameter paths and values against the ProjectConfig schema.

    For each parameter in the sweep config, this function:
    1. Verifies the dot-notated path exists in the ProjectConfig model tree
    2. Attempts to set each candidate value to catch type/validation errors

    This runs BEFORE any training starts, so typos and invalid values are caught
    immediately rather than crashing mid-sweep.

    Raises:
        ValueError: If any parameter path is invalid or any value fails validation.
    """
    for param_path, param_values in sweep_config.parameters.items():
        parent, field_name = _resolve_path(base_config, param_path)

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
            test_parent = parent.model_copy(deep=True)
            try:
                setattr(test_parent, field_name, val)
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
    new_config = config.model_copy(deep=True)
    for path, value in overrides.items():
        parent, field_name = _resolve_path(new_config, path)
        setattr(parent, field_name, value)
    return new_config


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
