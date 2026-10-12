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
