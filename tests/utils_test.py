import os
from pathlib import Path
from unittest import mock
import pytest
import yaml

from pydantic import BaseModel, ConfigDict, Field

from trainite.config.base import (
    DataConfigBase,
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    PreprocessorConfig,
    ProjectConfig,
    SplitConfig,
    TrainerConfig,
)
from trainite.shared.utils import create_accelerator, create_run_dir, get_target, instantiate, dump_config, load_config


class MockComponent(BaseModel):
    model_config = ConfigDict(extra="allow")
    target: str = Field(alias="_target_")


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


def test_load_grid_configs_rejects_empty_sweep_parameter(tmp_path):
    config_file = tmp_path / "config_empty_sweep.yaml"
    config_file.write_text("optimizer:\n  lr: 0.01\nsweep:\n  optimizer:\n    lr: []\n")

    with pytest.raises(
        ValueError,
        match="Sweep parameter 'optimizer.lr' must contain at least one value",
    ):
        load_grid_configs(config_file, MockSweepConfig)


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


def test_load_grid_configs_allows_null_base_value(tmp_path):
    class NullableConfig(BaseModel):
        optimizer: dict[str, float | None]

    config_file = tmp_path / "config_null.yaml"
    config_file.write_text("optimizer:\n  lr: null\nsweep:\n  optimizer:\n    lr: [0.01, 0.001]\n")

    configs = load_grid_configs(config_file, NullableConfig)

    assert len(configs) == 2
    assert [config.optimizer["lr"] for config, _ in configs] == [0.01, 0.001]


def test_load_grid_configs_rejects_flattened_sweep_keys(tmp_path):
    config_file = tmp_path / "config_flattened.yaml"
    yaml_content = "optimizer:\n  lr: 0.01\nbatch_size: 16\nsweep:\n  optimizer.lr: [0.01, 0.001]\n"
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="nested"):
        load_grid_configs(config_file, MockSweepConfig)


def test_load_grid_configs_sweep_typo(tmp_path):
    config_file = tmp_path / "config_typo.yaml"
    yaml_content = "optimizer:\n  lr: 0.01\nbatch_size: 16\nsweep:\n  optimizer:\n    lrr: [0.01, 0.001]\n"
    config_file.write_text(yaml_content)

    with pytest.raises(KeyError, match="Sweep key 'optimizer.lrr' does not exist"):
        load_grid_configs(config_file, MockSweepConfig)


def make_dummy_config(root_dir: Path, device: str | None = None) -> ProjectConfig:
    return ProjectConfig(
        project_name="test_proj",
        preprocessor=PreprocessorConfig(_target_="builtins.dict"),
        model=ModelConfig(_target_="builtins.dict"),
        output=OutputConfig(root=str(root_dir), run_name="test_run"),
        trainer=TrainerConfig(),
        data=DataConfigBase(
            train=SplitConfig(dataset=DatasetConfig(_target_="builtins.dict")),
            val=SplitConfig(dataset=DatasetConfig(_target_="builtins.dict")),
        ),
        device=device,
    )


def test_create_accelerator_cpu(tmp_path):
    cfg = make_dummy_config(tmp_path, device="cpu")
    accel = create_accelerator(cfg)
    assert accel.device.type == "cpu"
    assert os.environ.get("ACCELERATE_TORCH_DEVICE") is None


def test_create_accelerator_cuda_restores_env(tmp_path):
    cfg = make_dummy_config(tmp_path, device="cuda:1")
    with mock.patch("trainite.shared.utils.Accelerator") as mock_accel_cls:
        mock_accel = mock.MagicMock()

        def check_env(*args, **kwargs):
            assert os.environ.get("ACCELERATE_TORCH_DEVICE") == "cuda:1"
            return mock_accel

        mock_accel_cls.side_effect = check_env
        create_accelerator(cfg)
        mock_accel_cls.assert_called_once_with(cpu=False, mixed_precision="no")
        assert os.environ.get("ACCELERATE_TORCH_DEVICE") is None


def test_create_accelerator_distributed_guard(tmp_path):
    cfg = make_dummy_config(tmp_path, device="cuda:1")
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
        with pytest.raises(ValueError, match="Choose distributed GPUs through the launcher"):
            create_accelerator(cfg)


def test_create_run_dir(tmp_path):
    cfg = make_dummy_config(tmp_path, device="cpu")
    run_dir = create_run_dir(cfg)
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()


def test_loaded_list_config_can_be_dumped(tmp_path):
    source = tmp_path / "config.yaml"
    output = tmp_path / "output.yaml"
    source.write_text("_target_: builtins.dict\nlayer_types: [sliding, global]\n")

    config = load_config(source, MockComponent)
    dump_config(config, output)

    assert yaml.safe_load(output.read_text())["layer_types"] == ["sliding", "global"]
