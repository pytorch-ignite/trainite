from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf


def instantiate(config: DictConfig, **kwargs) -> Any:
    """
    Instantiates a class or calls a function defined by a `_target_` key
    using an OmegaConf DictConfig.
    """
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config

    if not isinstance(config_dict, dict) or "_target_" not in config_dict:
        raise ValueError("Config must be a dict containing a '_target_' key")

    # Work on a copy to avoid mutating the original config data
    params = config_dict.copy()
    target_path = params.pop("_target_")

    if not target_path:
        raise ValueError("The '_target_' key must not be empty.")

    try:
        # Split 'module.submodule.ClassName' into 'module.submodule' and 'ClassName'
        module_path, symbol_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_symbol = getattr(module, symbol_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate target '{target_path}': {e}") from e

    final_kwargs = {**params, **kwargs}

    return target_symbol(**final_kwargs)
