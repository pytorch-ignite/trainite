from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class ParameterRange(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
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


class SweepConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    strategy: Literal["grid", "random", "tpe"] = "grid"
    direction: Literal["maximize", "minimize"] = "maximize"
    metric: str = "exact_accuracy"
    n_trials: int | None = None
    parameters: dict[str, list[Any] | ParameterRange]

    @model_validator(mode="after")
    def validate_strategy_constraints(self) -> "SweepConfig":
        if self.strategy == "grid":
            for key, val in self.parameters.items():
                if isinstance(val, ParameterRange):
                    raise ValueError(
                        f"Parameter '{key}' uses a range, but strategy 'grid' "
                        f"requires explicit lists. Use [val1, val2, ...] instead."
                    )
        if self.strategy in ("random", "tpe") and self.n_trials is None:
            raise ValueError(f"n_trials is required when strategy is '{self.strategy}'")
        if not self.parameters:
            raise ValueError("At least one parameter must be specified")
        return self


def load_sweep_config(path: str | Path) -> SweepConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Sweep config file must contain a YAML mapping")
    return SweepConfig.model_validate(raw)
