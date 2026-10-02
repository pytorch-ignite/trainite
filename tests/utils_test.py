import pytest

from pydantic import BaseModel, ConfigDict, Field


class MockComponent(BaseModel):
    model_config = ConfigDict(extra="allow")
    target: str = Field(alias="_target_")


from trainite.shared.utils import get_target, instantiate


def cc(target: str | None = None, **kwargs: object) -> MockComponent:
    """Helper to create MockComponent with extra arguments without type errors."""
    if target:
        kwargs["_target_"] = target
    return MockComponent.model_validate(kwargs)


def test_get_target_success():
    target = get_target("trainite.shared.utils.get_target")
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
    config = cc("trainite.shared.utils.Nope")
    with pytest.raises(ImportError, match="Could not locate target"):
        instantiate(config)


def test_instantiate_get_target():
    config = cc(
        "trainite.shared.utils.get_target",
        target_path="trainite.shared.utils.get_target",
    )
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


# ==========================================
# Grid Search Tests
# ==========================================

from trainite.shared.utils import load_grid_configs


class OptimizerConfig(BaseModel):
    lr: float = 0.001


class MockSweepConfig(BaseModel):
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    batch_size: int = 32
    notes: str = "default"


def test_load_grid_configs_no_sweep(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("optimizer:\n  lr: 0.01\nbatch_size: 16\nnotes: test\n")

    configs = load_grid_configs(config_file, MockSweepConfig)
    assert len(configs) == 1

    config, params = configs[0]
    assert config.optimizer.lr == 0.01
    assert config.batch_size == 16
    assert params == {}


def test_load_grid_configs_with_nested_sweep(tmp_path):
    config_file = tmp_path / "config_sweep.yaml"
    yaml_content = (
        "optimizer:\n"
        "  lr: 0.01\n"
        "batch_size: 16\n"
        "notes: test\n"
        "sweep:\n"
        "  optimizer:\n"
        "    lr: [0.01, 0.001]\n"
        "  batch_size: [16, 64]\n"
    )
    config_file.write_text(yaml_content)

    configs = load_grid_configs(config_file, MockSweepConfig)
    assert len(configs) == 4

    combos = {(c[0].optimizer.lr, c[0].batch_size) for c in configs}
    assert combos == {
        (0.01, 16),
        (0.01, 64),
        (0.001, 16),
        (0.001, 64),
    }


def test_load_grid_configs_sweep_typo(tmp_path):
    config_file = tmp_path / "config_typo.yaml"
    yaml_content = "optimizer:\n  lr: 0.01\nbatch_size: 16\nsweep:\n  optimizer:\n    lrr: [0.01, 0.001]\n"
    config_file.write_text(yaml_content)

    with pytest.raises(KeyError, match="Sweep key 'optimizer.lrr' does not exist"):
        load_grid_configs(config_file, MockSweepConfig)
