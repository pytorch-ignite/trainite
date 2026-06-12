import pytest

from trainite.config import ComponentConfig
from trainite.utils import get_target, instantiate


def cc(target: str | None = None, **kwargs: object) -> ComponentConfig:
    """Helper to create ComponentConfig with extra arguments without type errors."""
    if target:
        kwargs["_target_"] = target
    return ComponentConfig.model_validate(kwargs)


def test_get_target_success():
    target = get_target("trainite.utils.get_target")
    assert target is get_target


def test_get_target_empty_path():
    with pytest.raises(ValueError):
        get_target("")


def test_get_target_import_error():
    with pytest.raises(ImportError, match="Could not locate target"):
        get_target("trainite.nonexistent.Nope")


def test_instantiate_non_component_config():
    with pytest.raises(ValueError):
        instantiate("not_a_config")  # pyright: ignore[reportArgumentType]


def test_instantiate_import_error():
    config = cc("trainite.utils.Nope")
    with pytest.raises(ImportError, match="Could not locate target"):
        instantiate(config)


def test_instantiate_get_target():
    config = cc("trainite.utils.get_target", target_path="trainite.utils.get_target")
    target = instantiate(config)
    assert target is get_target


def test_instantiate_dict_with_kwargs():
    config = cc("builtins.dict", key="value")
    result = instantiate(config)
    assert result == {"key": "value"}


def test_instantiate_kwargs_override():
    config = cc("builtins.dict", key="value")
    result = instantiate(config, key="override")
    assert result == {"key": "override"}
