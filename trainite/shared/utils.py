import importlib
from pathlib import Path
from typing import Any, TypeVar

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_target(target_path: str) -> Any:
    """
    Gets a class or a function defined by a `_target_` key
    using an OmegaConf DictConfig.
    """
    if not target_path:
        raise ValueError("The '_target_' key must not be empty.")

    try:
        # Split 'module.submodule.ClassName' into 'module.submodule' and 'ClassName'
        module_path, symbol_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_symbol = getattr(module, symbol_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate target '{target_path}': {e}") from e

    return target_symbol


def instantiate(config: BaseModel, **kwargs) -> Any:
    """
    Instantiates a class or calls a function defined by a `_target_` key
    using an OmegaConf DictConfig.
    """
    if isinstance(config, BaseModel):
        config_dict = config.model_dump(by_alias=True, polymorphic_serialization=True)
    else:
        raise ValueError("Config must be an instance of BaseModel")

    if not isinstance(config_dict, dict) or "_target_" not in config_dict:
        raise ValueError("Config must be a dict containing a '_target_' key")

    # Work on a copy to avoid mutating the original config data
    params = config_dict.copy()
    target_path = params.pop("_target_")

    target_symbol = get_target(target_path)

    final_kwargs = {**params, **kwargs}

    return target_symbol(**final_kwargs)


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def dump_yaml(data: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def dump_config(config: BaseModel, path: str | Path) -> None:
    data = config.model_dump(by_alias=True, polymorphic_serialization=True)
    dump_yaml(data, path)


def load_config(path: str | Path, config_cls: type[T]) -> T:
    raw_conf = OmegaConf.load(path)
    return config_cls.model_validate(raw_conf)
