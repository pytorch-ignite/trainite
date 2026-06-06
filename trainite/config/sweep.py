from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class ParameterRange(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    type: Literal["int", "float"]
    low: float
    high: float
    sample: Literal["uniform", "log"] = "uniform"
    step: float | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ParameterRange":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.sample == "log" and self.low <= 0:
            raise ValueError("low must be positive when sample='log'")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be positive")
        return self


class PruningConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    enabled: bool = False
    type: Literal[
        "median",
        "percentile",
        "hyperband",
        "successive_halving",
        "threshold",
        "patient",
        "wilcoxon",
        "nop",
    ] = "nop"

    # Parameters for Median and Percentile pruners
    n_startup_trials: int = 5
    n_min_trials: int = 1

    # Parameters for Median, Percentile, and Threshold pruners
    n_warmup_steps: int = 0
    interval_steps: int = 1

    # PercentilePruner specific
    percentile: float | None = None

    # HyperbandPruner / SuccessiveHalvingPruner shared parameters
    min_resource: str | int = "auto"
    reduction_factor: int = 4
    bootstrap_count: int = 0

    # HyperbandPruner specific
    max_resource: str | int = "auto"

    # SuccessiveHalvingPruner specific
    min_early_stopping_rate: int = 0

    # ThresholdPruner specific
    lower: float | None = None
    upper: float | None = None

    # PatientPruner specific
    patience: int | None = None
    min_delta: float = 0.0
    wrapped_type: Literal[
        "median",
        "percentile",
        "hyperband",
        "successive_halving",
        "threshold",
        "wilcoxon",
        "nop",
    ] = "median"

    # WilcoxonPruner specific
    p_threshold: float = 0.1
    n_startup_steps: int = 2

    @model_validator(mode="after")
    def validate_pruner_constraints(self) -> "PruningConfig":
        if not self.enabled:
            customized_fields = self.model_fields_set - {"enabled"}
            if customized_fields:
                raise ValueError(
                    f"Pruning is disabled. The following parameters are not allowed: {', '.join(sorted(customized_fields))}. "
                    "Only 'enabled: false' is permitted."
                )
            return self

        if self.type == "percentile":
            if self.percentile is None:
                raise ValueError("percentile is required when type is 'percentile'")
            if not (0.0 < self.percentile < 100.0):
                raise ValueError("percentile must be between 0.0 and 100.0 exclusive")
        elif self.type == "patient":
            if self.patience is None:
                raise ValueError("patience is required when type is 'patient'")
            if self.patience <= 0:
                raise ValueError("patience must be positive")
            if self.min_delta < 0:
                raise ValueError("min_delta must be non-negative")
        elif self.type == "threshold":
            if self.lower is None and self.upper is None:
                raise ValueError(
                    "At least one of 'lower' or 'upper' must be specified when type is 'threshold'"
                )
        elif self.type == "wilcoxon":
            if not (0.0 < self.p_threshold < 1.0):
                raise ValueError("p_threshold must be between 0.0 and 1.0 exclusive")
            if self.n_startup_steps <= 0:
                raise ValueError("n_startup_steps must be positive")
        elif self.type in ("hyperband", "successive_halving"):
            if self.reduction_factor < 2:
                raise ValueError("reduction_factor must be at least 2")
            if self.bootstrap_count < 0:
                raise ValueError("bootstrap_count must be non-negative")
        return self


class SweepConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    strategy: Literal[
        "grid",
        "random",
        "tpe",
        "brute_force",
        "cmaes",
        "gp",
        "nsgaii",
        "qmc",
    ] = "grid"
    direction: Literal["maximize", "minimize"] = "maximize"
    metric: str = "exact_accuracy"
    n_trials: int | None = None
    storage: str | None = None
    pruning: PruningConfig = PruningConfig()
    parameters: dict[str, list[Any] | ParameterRange]

    @model_validator(mode="after")
    def validate_strategy_constraints(self) -> "SweepConfig":
        if self.strategy in ("grid", "brute_force"):
            for key, val in self.parameters.items():
                if isinstance(val, ParameterRange):
                    raise ValueError(
                        f"Parameter '{key}' uses a range, but strategy '{self.strategy}' "
                        f"requires explicit lists. Use [val1, val2, ...] instead."
                    )
        if self.strategy not in ("grid", "brute_force") and self.n_trials is None:
            raise ValueError(f"n_trials is required when strategy is '{self.strategy}'")
        if not self.parameters:
            raise ValueError("At least one parameter must be specified")
        return self


def load_sweep_config(path: str | Path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Sweep config file must contain a YAML mapping")
    return SweepConfig.model_validate(raw)
