from __future__ import annotations

import importlib
from typing import Any


def instantiate(config: dict[str, Any] | Any, **kwargs) -> Any:
    """
    Instantiates a class or calls a function defined by a `_target_` key.

    If config is a Pydantic model, it is converted to a dict first.
    If config is already an object and doesn't have _target_, it is returned as is.
    """
    # Handle Pydantic models
    if hasattr(config, "model_dump"):
        config = config.model_dump()

    if not isinstance(config, dict) or "_target_" not in config:
        raise ValueError(
            "Config must be a dict with a '_target_' key or an already instantiated object."
        )

    # Work on a copy to avoid mutating the original config
    config_copy = config.copy()
    target_path = config_copy.pop("_target_")

    if not target_path:
        raise ValueError("The '_target_' key must not be empty.")

    try:
        module_path, symbol_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_symbol = getattr(module, symbol_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate target '{target_path}': {e}") from e

    # Merge YAML config with any runtime kwargs
    # kwargs take precedence over the config
    final_kwargs = {**config_copy, **kwargs}

    return target_symbol(**final_kwargs)
