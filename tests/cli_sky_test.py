from pathlib import Path
import pytest
import tomlkit
import yaml
from trainite.cli.main import main


@pytest.fixture
def mock_trainite_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "mock-experiment"
    project_dir.mkdir()

    config_yaml = """project_name: test-sky-exp
model:
  _target_: models.rope_transformer.RoPETransformerModel
output:
  root: outputs
  run_name: test_run
"""
    (project_dir / "config.yaml").write_text(config_yaml)
    (project_dir / "main.py").write_text("# main entrypoint\n")

    pyproject_toml = """[project]
name = "test-sky-exp"
version = "0.1.0"
dependencies = [
    "torch",
    "pytorch-ignite",
]
"""
    (project_dir / "pyproject.toml").write_text(pyproject_toml)
    return project_dir


def test_sky_init_in_valid_project(mock_trainite_project: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(mock_trainite_project)

    main(argv=["add:sky"])

    sky_yaml = mock_trainite_project / "sky.yaml"
    assert sky_yaml.exists()

    with open(sky_yaml, "r") as f:
        data = yaml.safe_load(f)

    assert data["name"] == "test-sky-exp"
    assert data["workdir"] == "."
    assert data["resources"]["accelerators"] == "L4:1"
    assert data["resources"]["use_spot"] is False
    assert data["resources"]["disk_size"] == 50
    assert "uv sync" in data["setup"]
    assert "uv run python main.py config.yaml" in data["run"]

    # Check pyproject.toml updated
    pyproject_doc = tomlkit.parse((mock_trainite_project / "pyproject.toml").read_text())
    deps = pyproject_doc["project"]["dependencies"]
    assert "skypilot" in deps

    output = capsys.readouterr().out
    assert "Generated sky.yaml for 'test-sky-exp'" in output
    assert "Added 'skypilot' to pyproject.toml dependencies" in output


def test_sky_init_idempotent(mock_trainite_project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(mock_trainite_project)

    main(argv=["add:sky"])
    main(argv=["add:sky", "--force"])

    pyproject_doc = tomlkit.parse((mock_trainite_project / "pyproject.toml").read_text())
    deps = list(pyproject_doc["project"]["dependencies"])
    assert deps.count("skypilot") == 1


def test_sky_init_fails_outside_trainite_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    empty_dir = tmp_path / "empty-dir"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)

    with pytest.raises(SystemExit) as exc_info:
        main(argv=["add:sky"])

    assert exc_info.value.code == 1
    assert not (empty_dir / "sky.yaml").exists()

    err = capsys.readouterr().err
    assert "is not a valid Trainite project directory" in err
    assert "trainite add:sky" in err


def test_sky_init_overwrite_protection(mock_trainite_project: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(mock_trainite_project)

    main(argv=["add:sky"])

    with pytest.raises(SystemExit) as exc_info:
        main(argv=["add:sky"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    assert "Pass '--force' to overwrite it: trainite add:sky --force" in err
